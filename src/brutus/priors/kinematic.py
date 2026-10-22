#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Velocity moments of the Galactic components and their sky projection.

This module gives the mean velocity and the velocity covariance (the
velocity ellipsoid) of the thin disk, thick disk and halo at a
Galactocentric position, the solar velocity, and the projection of a
velocity ellipsoid to the observables: proper motions in Galactic
coordinates and radial velocity.

Velocities are in km/s and proper motions in mas/yr. Cylindrical
velocities are ordered ``(v_R, v_phi, v_z)``, with ``v_phi`` positive in
the direction of Galactic rotation. Observables are ordered
``(mu_l*, mu_b, v_r)``, where ``mu_l* = mu_l cos(b)``.
`~brutus.utils.math.pm_equatorial_to_galactic` converts Gaia proper
motions to this frame.

The disk dispersions are fits to Gaia-era data: the thin disk follows the
age, angular-momentum, metallicity and height relations of Sharma et al.
(2021), and the thick disk the high-[alpha/Fe] dispersions of Anguiano et
al. (2020) with the radial profile of Mackereth et al. (2019). The mean
rotation of both disks is the radial Jeans solution for the density of
`~brutus.priors.galactic.logn_disk`. The halo is the isotropic plus
radially anisotropic mixture of Lancaster et al. (2019).

See Also
--------
brutus.priors.galactic : Number densities of the same components

Examples
--------
>>> import numpy as np
>>> from brutus.priors.kinematic import (
...     thin_disk_velocity_moments, velocity_to_observables)
>>> from brutus.utils.math import galactic_to_galactocentric_cyl
>>>
>>> dists = np.array([0.5, 1.0, 2.0])  # kpc
>>> coord = (90.0, 30.0)  # (l, b) in degrees
>>> R, Z = galactic_to_galactocentric_cyl(dists, *coord)
>>> mean, cov = thin_disk_velocity_moments(R, Z, age=5.0, feh=0.0)
>>> obs_mean, obs_cov = velocity_to_observables(dists, coord, mean, cov)
"""

import os
from math import asinh, exp, radians, sqrt

import numpy as np
from numba import jit, prange
from scipy.interpolate import BSpline

__all__ = [
    "solar_velocity",
    "thin_disk_velocity_moments",
    "thick_disk_velocity_moments",
    "halo_velocity_moments",
    "velocity_to_observables",
]

# km/s per (kpc * mas/yr): one astronomical unit per Julian year.
K_KMS = 149597870.7 / (365.25 * 86400.0)

# Sgr A* proper motion (mu_l*, mu_b) in mas/yr (Reid & Brunthaler 2020).
PM_SGRA = (-6.411, -0.219)
# Solar motion toward the Galactic centre and in the direction of rotation
# relative to the circular velocity, in km/s (Schönrich et al. 2010).
U_SUN = 11.1
V_PEC_SUN = 12.24
# Shape of the circular velocity curve, v_c(R) / v_c(R_sun) against
# x = R / R_sun, from the measurements in rotation_curve.dat (Sofue & Kohno
# 2025; Eilers et al. 2019), each scaled by its own (R_sun, v_c,sun).
# Eilers et al. bins with errors above VC_MAX_ERR (km/s) are left out:
# those beyond 21 kpc have 6-29 km/s. One cubic P-spline is fitted to all
# points, weighted by their errors, in u = asinh(x / VC_CORE): linear in R
# near the centre and close to ln R beyond it, so the roughness penalty
# VC_PENALTY smooths over a fixed fraction of R (1-2 kpc near the Sun).
# The spline is odd in R, so v_c is proportional to R at the centre.
# Beyond the last point a slope penalty, ramped in over VC_FLAT, makes v_c
# flat. Knots are VC_KNOT apart in u, out to VC_END.
VC_SOURCES = ((8.178, 235.1), (8.122, 229.0))
VC_MAX_ERR = 5.0
VC_CORE = 0.3 / 8.2
VC_PENALTY = 1e4
VC_FLAT = 2.0 / 8.2
VC_KNOT = 0.05
VC_END = 40.0 / 8.2
_VC_DU = 0.002

# Thin-disk dispersion relations, (sigma_R, sigma_z) pairs (Sharma et al.
# 2021, eqs. 3-7 and Table 2): sigma at 10 Gyr, solar L_z, [Fe/H] = 0 and
# z = 0 in km/s; age exponent; L_z scale in kpc km/s; L_z weight; [Fe/H]
# slope per dex; |z| slope per kpc.
THIN_DISK_SIGMA0 = (39.4, 21.1)
THIN_DISK_BETA = (0.251, 0.441)
THIN_DISK_LAMBDA_L = (2300.0, 1130.0)
THIN_DISK_ALPHA_L = (0.09, 0.58)
THIN_DISK_GAMMA_FEH = (-0.19, -0.52)
THIN_DISK_GAMMA_Z = (0.12, 0.20)
# Radii in kpc where the thin-disk relations were fitted (Sharma et al. 2021).
# Beyond them the dispersions keep their values at the edges, reached
# through a quadratic blend within DISK_EDGE kpc of each edge, so that
# v_phi stays continuous.
THIN_DISK_R_RANGE = (3.0, 20.0)
DISK_EDGE = 1.0

# Thick-disk (high-[alpha/Fe]) dispersions (sigma_R, sigma_phi, sigma_z)
# in km/s (Anguiano et al. 2020, Table 1) and the e-folding length of the
# dispersions in kpc (Mackereth et al. 2019, h_sigma_R).
THICK_DISK_SIGMA = (64.68, 50.82, 43.60)
THICK_DISK_R_SIGMA = 15.0
# Radii in kpc of the Mackereth et al. (2019) sample. Beyond them the
# thick-disk dispersions keep their values at the edges, as for the thin
# disk.
THICK_DISK_R_RANGE = (4.0, 13.0)

# Halo mixture by Galactocentric radius (Lancaster et al. 2019, Table 2):
# centres of the radial bins in kpc; weight of the isotropic component;
# its radial and tangential dispersions; the dispersions of the two equal
# radial lobes, their separation from v_r = 0 and their mean v_phi; km/s.
# Lancaster et al. take v_phi < 0 for disc rotation, so the lobe v_phi here
# is their value with the sign changed.
HALO_R = (9.0, 16.15, 24.6, 48.965)
HALO_F_ISO = (0.52, 0.55, 0.55, 0.96)
HALO_SIGMA_R_ISO = (129.24, 122.03, 113.62, 95.21)
HALO_SIGMA_T_ISO = (114.27, 105.41, 96.72, 79.33)
HALO_SIGMA_R_LOBE = (109.93, 78.21, 62.91, 176.36)
HALO_SIGMA_T_LOBE = (58.18, 34.02, 29.14, 145.69)
HALO_DELTA_LOBE = (104.16, 98.60, 67.75, 82.87)
HALO_V_PHI_LOBE = (4.49, -11.54, -14.13, 27.39)


def solar_velocity(R_solar=8.2, pm_sgra=PM_SGRA, U_sun=U_SUN):
    r"""
    Return the velocity of the Sun relative to the Galactic centre.

    Sgr A* is taken to be at rest at the Galactic centre, so its proper
    motion is the reflex of the solar motion:

    .. math::
        V_\odot = -k R_\odot \mu_{l,\mathrm{Sgr\,A^*}}, \qquad
        W_\odot = -k R_\odot \mu_{b,\mathrm{Sgr\,A^*}}

    with :math:`k = 4.74047` km/s per (kpc mas/yr). This ties the solar
    velocity to ``R_solar``: use the same ``R_solar`` in the moment
    functions and in `velocity_to_observables`.

    Parameters
    ----------
    R_solar : float, optional
        Distance of the Sun from the Galactic centre in kpc. Default is 8.2.
    pm_sgra : tuple of float, optional
        Proper motion ``(mu_l*, mu_b)`` of Sgr A* in mas/yr. Default is
        ``(-6.411, -0.219)`` (Reid & Brunthaler 2020).
    U_sun : float, optional
        Solar velocity toward the Galactic centre in km/s. Default is 11.1
        (Schönrich et al. 2010).

    Returns
    -------
    v_sun : `~numpy.ndarray` of shape `(3,)`
        Solar velocity ``(U, V, W)`` in km/s, with U toward the Galactic
        centre, V in the direction of rotation and W toward the north
        Galactic pole.

    Raises
    ------
    ValueError
        If ``R_solar`` is not positive.
    """
    if not R_solar > 0:
        raise ValueError(f"R_solar must be positive, got {R_solar}")
    return np.array(
        [U_sun, -K_KMS * R_solar * pm_sgra[0], -K_KMS * R_solar * pm_sgra[1]]
    )


def _rotation_curve_table():
    """Return the shape of v_c and its slope per grid step on a grid
    uniform in u = asinh(x / VC_CORE), starting at u = 0."""
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), "rotation_curve.dat"))
    R, v, err, src = data[(data[:, 2] <= VC_MAX_ERR) | (data[:, 3] == 0)].T
    R_sun, vc_sun = (np.array(c)[src.astype(int)] for c in zip(*VC_SOURCES))
    u = np.arcsinh(R / R_sun / VC_CORE)
    weight = (vc_sun / err) ** 2

    # Odd cubic B-splines on knots symmetric about u = 0: coefficient j and
    # its mirror have opposite signs, and the middle one is 0.
    u_end = np.arcsinh(VC_END / VC_CORE)
    half = np.arange(0.0, u_end + VC_KNOT, VC_KNOT)
    knots = np.r_[[-half[-1]] * 3, -half[:0:-1], half, [half[-1]] * 3]
    n = len(knots) - 4
    h = n // 2
    odd = np.zeros((n, h))
    odd[h + 1 :] = np.eye(h)
    odd[:h] = -np.eye(h)[::-1]
    B = BSpline(knots, odd, 3)(u)

    # Curvature penalty everywhere; beyond the last point, a slope penalty
    # ramped in with a cosine weight, 1e4 times stronger once complete.
    greville = np.array([knots[i + 1 : i + 4].mean() for i in range(n)])
    mid = 0.5 * np.abs(greville[1:] + greville[:-1])
    u_last, u_flat = u.max(), np.arcsinh(
        (np.sinh(u.max()) * VC_CORE + VC_FLAT) / VC_CORE
    )
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip((mid - u_last) / (u_flat - u_last), 0, 1))
    D1, D2 = np.diff(np.eye(n), 1, axis=0), np.diff(np.eye(n), 2, axis=0)
    penalty = odd.T @ (D2.T @ D2 + 1e4 * D1.T @ (ramp[:, None] * D1)) @ odd
    coef = np.linalg.solve(
        B.T @ (weight[:, None] * B) + VC_PENALTY * penalty,
        B.T @ (weight * v / vc_sun),
    )
    spline = BSpline(knots, odd @ coef, 3)
    grid = np.arange(0.0, u_end, _VC_DU)
    norm = spline(np.arcsinh(1.0 / VC_CORE))
    # The slope is per grid step, the unit the Hermite interpolation uses.
    return spline(grid) / norm, spline.derivative()(grid) * _VC_DU / norm


_VC_SHAPE, _VC_SLOPE = _rotation_curve_table()


@jit(nopython=True, cache=True)
def _circular_velocity(R, R_solar, vc_solar):
    """Return v_c(R) and d ln v_c / d ln R.

    The shape is interpolated in _VC_SHAPE, against u = asinh(x / VC_CORE)
    with x = R / R_solar, with cubic Hermite polynomials whose slopes are
    _VC_SLOPE (per grid step), so v_c and its derivative are continuous.
    """
    if R <= 0.0:
        return 0.0, 1.0
    x = R / R_solar
    y = asinh(x / VC_CORE) / _VC_DU
    n = len(_VC_SHAPE) - 1
    if y >= n:
        return vc_solar * _VC_SHAPE[n], 0.0
    i = int(y)
    w = y - i
    p0, p1 = _VC_SHAPE[i], _VC_SHAPE[i + 1]
    m0, m1 = _VC_SLOPE[i], _VC_SLOPE[i + 1]
    w2, w3 = w * w, w * w * w
    s = (
        (2.0 * w3 - 3.0 * w2 + 1.0) * p0
        + (w3 - 2.0 * w2 + w) * m0
        + (3.0 * w2 - 2.0 * w3) * p1
        + (w3 - w2) * m1
    )
    ds_dy = (
        (6.0 * w2 - 6.0 * w) * p0
        + (3.0 * w2 - 4.0 * w + 1.0) * m0
        + (6.0 * w - 6.0 * w2) * p1
        + (3.0 * w2 - 2.0 * w) * m1
    )
    # du / d ln R = x / sqrt(x^2 + VC_CORE^2)
    return vc_solar * s, ds_dy / (_VC_DU * s) * x / sqrt(x * x + VC_CORE * VC_CORE)


@jit(nopython=True, cache=True)
def _logn_disk_dlnR(R, R_eff, R_scale):
    """Return d ln(nu) / d ln(R) of `~brutus.priors.galactic.logn_disk`,
    with the smoothed radius R_eff = sqrt(R^2 + R_smooth^2)."""
    return -R * R / (R_scale * R_eff)


@jit(nopython=True, cache=True)
def _jeans_vphi(var_R, var_phi, dlnvar_dlnR, dlnnu_dlnR, vc):
    """Mean rotation from the radial Jeans equation, clipped to [0, vc].

    Binney & Tremaine (2008), eq. 4.228, for a cylindrically aligned
    ellipsoid. The density gradient comes from the caller, so the rotation
    follows whatever density the caller uses.
    """
    vphi2 = vc * vc - var_phi + var_R * (1.0 + dlnnu_dlnR + dlnvar_dlnR)
    # Where the dispersions exceed what rotation can support, the equation
    # has no solution; near the centre, where the density gradient
    # vanishes, it gives rotation faster than circular.
    return min(sqrt(max(vphi2, 0.0)), vc)


@jit(nopython=True, cache=True)
def _thin_disk_sigma0(age, feh):
    """Return the Sharma et al. (2021) (sigma_R, sigma_z) of one stellar
    model at solar L_z and z = 0: sigma_0 f_tau f_[Fe/H].

    These factors depend only on the model, so a caller that evaluates many
    positions per model computes them once.
    """
    a = (age + 0.1) / 10.1
    return (
        THIN_DISK_SIGMA0[0]
        * a ** THIN_DISK_BETA[0]
        * (1.0 + THIN_DISK_GAMMA_FEH[0] * feh),
        THIN_DISK_SIGMA0[1]
        * a ** THIN_DISK_BETA[1]
        * (1.0 + THIN_DISK_GAMMA_FEH[1] * feh),
    )


@jit(nopython=True, cache=True)
def _soft_clamp(x, lo, hi, width):
    """Return x clamped to [lo, hi] and d(clamped x) / dx. Within ``width``
    of an edge the clamp is a quadratic blend, so both are continuous;
    further out it is the hard clamp. Needs hi - lo >= 2 width."""
    if x <= lo - width:
        return lo, 0.0
    if x < lo + width:
        t = x - lo + width
        return lo + t * t / (4.0 * width), t / (2.0 * width)
    if x <= hi - width:
        return x, 1.0
    if x < hi + width:
        t = hi + width - x
        return hi - t * t / (4.0 * width), t / (2.0 * width)
    return hi, 0.0


@jit(nopython=True, cache=True)
def _thin_disk_fL(L, L_solar, L_clamp):
    """Return the Sharma et al. (2021) f_L of sigma_R and of sigma_z, and
    d ln f_L(sigma_R) / d ln L_z, with L_z clamped by
    ``L_clamp`` = (L_lo, L_hi, width): L_z of circular orbits at the ends
    of THIN_DISK_R_RANGE, and DISK_EDGE times v_c at the Sun."""
    L_eff, dL = _soft_clamp(L, L_clamp[0], L_clamp[1], L_clamp[2])
    u2 = (L_eff / L_solar) ** 2
    a_R, a_z = THIN_DISK_ALPHA_L
    lam_R, lam_z = THIN_DISK_LAMBDA_L
    e_R = exp(-(L_eff - L_solar) / lam_R)
    f_R = (a_R * u2 + e_R) / (1.0 + a_R)
    f_z = (a_z * u2 + exp(-(L_eff - L_solar) / lam_z)) / (1.0 + a_z)
    dlnf_dlnL = (2.0 * a_R * u2 - (L_eff / lam_R) * e_R) / ((1.0 + a_R) * f_R)
    return f_R, f_z, dlnf_dlnL * dL * L / L_eff


@jit(nopython=True, cache=True)
def _thin_disk_radial(R, abs_z, sig_R0, vc, dlnvc, L, L_solar, L_clamp, dlnnu_dlnR):
    """Return (v_phi, var_R, var_phi, f_L of sigma_z) of the thin disk for
    f_L at ``L``.

    sigma_phi follows the epicyclic ratio
    sigma_phi^2 / sigma_R^2 = (1 + d ln v_c / d ln R) / 2, and
    d ln L_z / d ln R is taken as that of a circular orbit, 1 + d ln v_c /
    d ln R. Outside THIN_DISK_R_RANGE the gradient term fades out within
    DISK_EDGE, in R: a population that lags keeps an L_z below the
    clamp a little beyond 20 kpc, and the gradient would otherwise act
    there.
    """
    f_R, f_z, dlnf_R = _thin_disk_fL(L, L_solar, L_clamp)
    sig_R = sig_R0 * f_R * (1.0 + THIN_DISK_GAMMA_Z[0] * abs_z)
    var_R = sig_R * sig_R
    var_phi = 0.5 * var_R * (1.0 + dlnvc)
    fade = _soft_clamp(R, THIN_DISK_R_RANGE[0], THIN_DISK_R_RANGE[1], DISK_EDGE)[1]
    dlnvar_dlnR = 2.0 * dlnf_R * (1.0 + dlnvc) * fade
    vphi = _jeans_vphi(var_R, var_phi, dlnvar_dlnR, dlnnu_dlnR, vc)
    return vphi, var_R, var_phi, f_z


@jit(nopython=True, cache=True)
def _thin_disk_mean_L(R, abs_z, sig_R0, vc, dlnvc, L_solar, L_clamp, dlnnu_dlnR):
    """Return the mean L_z = R v_phi of a thin-disk population.

    Two steps of the fixed point L_z = R v_phi(L_z) from the circular
    L_z = R v_c. For R_solar = 8.2 kpc, 0.5 < R < 25 kpc, all ages and
    -1 < [Fe/H] < 0.5, f_L of sigma_R is then within 0.3% of the
    converged value at |z| < 1 kpc and 2.1% at |z| < 3 kpc. Where several
    L_z solve it (old, metal-poor stars beyond 11 kpc), the steps approach
    the largest, the solution connected to circular orbits.
    """
    L = R * vc
    for _ in range(2):
        L = (
            R
            * _thin_disk_radial(
                R, abs_z, sig_R0, vc, dlnvc, L, L_solar, L_clamp, dlnnu_dlnR
            )[0]
        )
    return L


@jit(nopython=True, cache=True)
def _thin_disk_moments(
    R, abs_z, sig_R0, sig_z0, vc, dlnvc, L_solar, L_clamp, dlnnu_dlnR
):
    """Return (v_phi, var_R, var_phi, var_z) of the thin disk.

    ``sig_R0`` and ``sig_z0`` come from ``_thin_disk_sigma0``. The
    dispersions take f_L at the population's mean L_z from
    ``_thin_disk_mean_L``; ``L_solar`` is R_solar v_c(R_solar), the L_z of
    a circular orbit at the Sun, and ``L_clamp`` is as in ``_thin_disk_fL``.
    """
    L = _thin_disk_mean_L(R, abs_z, sig_R0, vc, dlnvc, L_solar, L_clamp, dlnnu_dlnR)
    vphi, var_R, var_phi, f_z = _thin_disk_radial(
        R, abs_z, sig_R0, vc, dlnvc, L, L_solar, L_clamp, dlnnu_dlnR
    )
    sig_z = sig_z0 * f_z * (1.0 + THIN_DISK_GAMMA_Z[1] * abs_z)
    return vphi, var_R, var_phi, sig_z * sig_z


@jit(nopython=True, cache=True)
def _thick_disk_moments(R, vc, var_R0, var_phi0, var_z0, R_sigma, R_solar, dlnnu_dlnR):
    """Return (v_phi, var_R, var_phi, var_z) of the thick disk.

    Beyond THICK_DISK_R_RANGE the dispersions keep their edge values,
    blended in within DISK_EDGE of each edge.
    """
    R_eff, dR = _soft_clamp(R, THICK_DISK_R_RANGE[0], THICK_DISK_R_RANGE[1], DISK_EDGE)
    f = exp(-2.0 * (R_eff - R_solar) / R_sigma)
    var_R = var_R0 * f
    var_phi = var_phi0 * f
    dlnvar_dlnR = -2.0 * R / R_sigma * dR
    vphi = _jeans_vphi(var_R, var_phi, dlnvar_dlnR, dlnnu_dlnR, vc)
    return vphi, var_R, var_phi, var_z0 * f


@jit(nopython=True, cache=True)
def _halo_moments(R, Z):
    """Return the halo mixture at (R, Z): the weight of the isotropic
    component, its (var_R, var_phi, var_z, cov_Rz), and the lobe at
    v_r = +delta as (m_R, m_phi, m_z, var_R, var_phi, var_z, cov_Rz). The
    lobe at -delta has the same moments with m_R and m_z negated, and each
    lobe has weight (1 - f_iso) / 2.
    """
    r = sqrt(R * R + Z * Z)
    if r > 0.0:
        s, c = R / r, Z / r
    else:
        s, c = 1.0, 0.0
    # Linear interpolation between the bin centres, constant beyond them.
    n = len(HALO_R)
    i, w = 0, 0.0
    if r >= HALO_R[n - 1]:
        i, w = n - 2, 1.0
    elif r > HALO_R[0]:
        while r > HALO_R[i + 1]:
            i += 1
        w = (r - HALO_R[i]) / (HALO_R[i + 1] - HALO_R[i])

    def row(ys):
        return (1.0 - w) * ys[i] + w * ys[i + 1]

    f_iso = row(HALO_F_ISO)
    ss, cc, sc = s * s, c * c, s * c
    # v_R = v_r sin(theta) + v_theta cos(theta),
    # v_z = v_r cos(theta) - v_theta sin(theta), theta from the +z axis.
    var_r, var_t = row(HALO_SIGMA_R_ISO) ** 2, row(HALO_SIGMA_T_ISO) ** 2
    iso = (
        var_r * ss + var_t * cc,
        var_t,
        var_r * cc + var_t * ss,
        (var_r - var_t) * sc,
    )
    delta = row(HALO_DELTA_LOBE)
    var_r, var_t = row(HALO_SIGMA_R_LOBE) ** 2, row(HALO_SIGMA_T_LOBE) ** 2
    lobe = (
        delta * s,
        row(HALO_V_PHI_LOBE),
        delta * c,
        var_r * ss + var_t * cc,
        var_t,
        var_r * cc + var_t * ss,
        (var_r - var_t) * sc,
    )
    return f_iso, iso, lobe


@jit(nopython=True, cache=True)
def _store(mean, cov, m_R, m_phi, m_z, var_R, var_phi, var_z, cov_Rz):
    mean[0] = m_R
    mean[1] = m_phi
    mean[2] = m_z
    cov[0, 0] = var_R
    cov[1, 1] = var_phi
    cov[2, 2] = var_z
    cov[0, 2] = cov_Rz
    cov[2, 0] = cov_Rz


@jit(nopython=True, parallel=True, cache=True)
def _thin_disk_array(R, Z, age, feh, R_scale, R_smooth, R_solar, vc_solar):
    N = len(R)
    mean = np.zeros((N, 3))
    cov = np.zeros((N, 3, 3))
    lo, hi = THIN_DISK_R_RANGE
    L_solar = R_solar * _circular_velocity(R_solar, R_solar, vc_solar)[0]
    L_lo = lo * _circular_velocity(lo, R_solar, vc_solar)[0]
    L_hi = hi * _circular_velocity(hi, R_solar, vc_solar)[0]
    for i in prange(N):
        vc, dlnvc = _circular_velocity(R[i], R_solar, vc_solar)
        sig_R0, sig_z0 = _thin_disk_sigma0(age[i], feh[i])
        vphi, var_R, var_phi, var_z = _thin_disk_moments(
            R[i],
            abs(Z[i]),
            sig_R0,
            sig_z0,
            vc,
            dlnvc,
            L_solar,
            (L_lo, L_hi, DISK_EDGE * vc_solar),
            _logn_disk_dlnR(R[i], sqrt(R[i] * R[i] + R_smooth * R_smooth), R_scale),
        )
        _store(mean[i], cov[i], 0.0, vphi, 0.0, var_R, var_phi, var_z, 0.0)
    return mean, cov


@jit(nopython=True, parallel=True, cache=True)
def _thick_disk_array(
    R,
    var_R0,
    var_phi0,
    var_z0,
    R_sigma,
    R_scale,
    R_smooth,
    R_solar,
    vc_solar,
):
    N = len(R)
    mean = np.zeros((N, 3))
    cov = np.zeros((N, 3, 3))
    for i in prange(N):
        vc = _circular_velocity(R[i], R_solar, vc_solar)[0]
        vphi, var_R, var_phi, var_z = _thick_disk_moments(
            R[i],
            vc,
            var_R0,
            var_phi0,
            var_z0,
            R_sigma,
            R_solar,
            _logn_disk_dlnR(R[i], sqrt(R[i] * R[i] + R_smooth * R_smooth), R_scale),
        )
        _store(mean[i], cov[i], 0.0, vphi, 0.0, var_R, var_phi, var_z, 0.0)
    return mean, cov


@jit(nopython=True, parallel=True, cache=True)
def _halo_array(R, Z):
    N = len(R)
    weight = np.empty((N, 3))
    mean = np.zeros((N, 3, 3))
    cov = np.zeros((N, 3, 3, 3))
    for i in prange(N):
        f_iso, iso, lobe = _halo_moments(R[i], Z[i])
        weight[i, 0] = f_iso
        weight[i, 1] = weight[i, 2] = 0.5 * (1.0 - f_iso)
        _store(mean[i, 0], cov[i, 0], 0.0, 0.0, 0.0, *iso)
        _store(mean[i, 1], cov[i, 1], *lobe)
        m_R, m_phi, m_z = lobe[0], lobe[1], lobe[2]
        _store(
            mean[i, 2], cov[i, 2], -m_R, m_phi, -m_z, lobe[3], lobe[4], lobe[5], lobe[6]
        )
    return weight, mean, cov


def _check_radii(R):
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 1:
        raise ValueError(f"R must be 1-D, got shape {R.shape}")
    if not np.all(np.isfinite(R) & (R >= 0)):
        raise ValueError("R must be finite and non-negative")
    return R


def _check_heights(Z, R):
    Z = np.asarray(Z, dtype=np.float64)
    if Z.shape != R.shape:
        raise ValueError(f"Z must have the shape of R {R.shape}, got {Z.shape}")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z must be finite")
    return Z


def _check_positive(**values):
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}")


def _vc_solar(v_circ_solar, R_solar):
    """Return v_c at the Sun, after checking that it is positive."""
    if v_circ_solar is None:
        v_circ_solar = solar_velocity(R_solar)[1] - V_PEC_SUN
    v_circ_solar = float(v_circ_solar)
    if not (np.isfinite(v_circ_solar) and v_circ_solar > 0):
        raise ValueError(
            f"v_circ_solar must be finite and positive, got {v_circ_solar}"
        )
    return v_circ_solar


_FEH_MAX = -1.0 / min(THIN_DISK_GAMMA_FEH)


def thin_disk_velocity_moments(
    R,
    Z,
    age,
    feh,
    R_scale=2.6,
    R_smooth=2.0,
    R_solar=8.2,
    v_circ_solar=None,
):
    r"""
    Return the velocity ellipsoid of the thin disk.

    The radial and vertical dispersions follow Sharma et al. (2021, eqs.
    3-7 and Table 2):

    .. math::
        \sigma_v = \sigma_{0,v} \left(\frac{\tau + 0.1}{10.1}\right)^{\beta_v}
        f_{L_z}(L_z)\,(1 + \gamma_{\mathrm{[Fe/H]},v}\mathrm{[Fe/H]})
        (1 + \gamma_{z,v} |z|)

    with age :math:`\tau` in Gyr, :math:`|z|` in kpc and
    :math:`f_{L_z} = [\alpha_{L,v} (L_z/L_{z,\odot})^2
    + e^{-(L_z - L_{z,\odot})/\lambda_{L,v}}] / (1 + \alpha_{L,v})`. The
    :math:`L_{z,\odot} = R_\odot v_c(R_\odot)`. :math:`L_z` is the mean
    angular momentum of the population, :math:`R\,\bar{v}_\phi`, with
    :math:`\bar{v}_\phi` from the Jeans equation below (two steps of the
    fixed point from the circular :math:`L_z = R\,v_c`). The relations
    were fitted at 3 < R < 20 kpc. Outside those radii the dispersions
    keep their edge values, blended in within 1 kpc of each edge so that
    :math:`\bar{v}_\phi` stays continuous: :math:`L_z` is clamped to the
    circular :math:`L_z` at the edges, and the dispersion gradient in the
    Jeans equation fades out in R. The azimuthal dispersion
    follows the epicyclic ratio
    :math:`\sigma_\phi^2 / \sigma_R^2 = (1 + d\ln v_c / d\ln R) / 2`.

    The mean rotation is the radial Jeans solution (Binney & Tremaine 2008,
    eq. 4.228) for a cylindrically aligned ellipsoid and the density
    gradient of `~brutus.priors.galactic.logn_disk`:

    .. math::
        \bar{v}_\phi^2 = v_c^2 - \sigma_\phi^2
        + \sigma_R^2 \left(1 + \frac{\partial \ln \nu}{\partial \ln R}
        + \frac{\partial \ln \sigma_R^2}{\partial \ln R}\right)

    with the measured circular velocity :math:`v_c(R)`: one smoothing
    spline fitted to Sofue & Kohno (2025) and Eilers et al. (2019) together,
    each scaled to :math:`v_{c,\odot}` at :math:`R_\odot`
    (``rotation_curve.dat``). It smooths over 1-2 kpc near the Sun, goes
    to 0 as R at the centre and is flat beyond the last measurement at
    21 kpc. The mean rotation is
    clipped to :math:`[0, v_c]`: the Jeans equation has no
    solution where the right-hand side is negative, and near the centre,
    where the density gradient vanishes, it gives rotation faster than
    circular. The model is axisymmetric, so it does not describe the bar
    of the inner few kpc. The mean radial and vertical velocities are 0.

    Parameters
    ----------
    R, Z : array_like of shape `(N,)`
        Galactocentric cylindrical radius and height above the midplane in
        kpc.
    age : float or array_like of shape `(N,)`
        Stellar age in Gyr.
    feh : float or array_like of shape `(N,)`
        Metallicity [Fe/H].
    R_scale, R_smooth : float, optional
        Radial scale length and smoothing radius of the number density in
        kpc, as in `~brutus.priors.galactic.logn_disk`. Defaults are 2.6
        and 2.0.
    R_solar : float, optional
        Distance of the Sun from the Galactic centre in kpc. Default is 8.2.
    v_circ_solar : float, optional
        Circular velocity at ``R_solar`` in km/s, which scales the measured
        curve. Default is the solar V from ``solar_velocity(R_solar)``
        minus the peculiar 12.24 km/s of Schönrich et al. (2010):
        236.97 km/s for ``R_solar=8.2``. The shape of the curve is a
        function of ``R / R_solar``.

    Returns
    -------
    mean : `~numpy.ndarray` of shape `(N, 3)`
        Mean velocity ``(v_R, v_phi, v_z)`` in km/s.
    cov : `~numpy.ndarray` of shape `(N, 3, 3)`
        Velocity covariance in (km/s)^2.

    Raises
    ------
    ValueError
        If the positions are not finite 1-D arrays of one length, ``R`` or
        ``age`` is negative, ``feh`` is not below 1.92 (where a dispersion
        would vanish), ``age`` or ``feh`` does not broadcast to ``R``, a
        length scale is not positive, or ``v_circ_solar`` is not positive.
    """
    R = _check_radii(R)
    Z = _check_heights(Z, R)
    try:
        age, feh = (
            np.ascontiguousarray(
                np.broadcast_to(np.asarray(x, dtype=np.float64), R.shape)
            )
            for x in (age, feh)
        )
    except ValueError:
        raise ValueError(f"age and feh must broadcast to the shape of R {R.shape}")
    if not np.all(np.isfinite(age) & (age >= 0)):
        raise ValueError("age must be finite and non-negative")
    if not np.all(np.isfinite(feh) & (feh < _FEH_MAX)):
        raise ValueError(f"feh must be finite and below {_FEH_MAX:.2f}")
    _check_positive(R_scale=R_scale, R_smooth=R_smooth, R_solar=R_solar)
    return _thin_disk_array(
        R,
        Z,
        age,
        feh,
        float(R_scale),
        float(R_smooth),
        float(R_solar),
        _vc_solar(v_circ_solar, R_solar),
    )


def thick_disk_velocity_moments(
    R,
    sigma_solar=THICK_DISK_SIGMA,
    R_sigma=THICK_DISK_R_SIGMA,
    R_scale=2.0,
    R_smooth=2.0,
    R_solar=8.2,
    v_circ_solar=None,
):
    r"""
    Return the velocity ellipsoid of the thick disk.

    The dispersions do not depend on age (Mackereth et al. 2019 find the
    high-[alpha/Fe] age-velocity relations flat) or on height, and fall
    off with radius as

    .. math::
        \sigma_i(R) = \sigma_{i,\odot}\, e^{-(R - R_\odot)/R_\sigma}.

    The defaults are the high-[alpha/Fe] dispersions of Anguiano et al.
    (2020, Table 1) and the e-folding length of Mackereth et al. (2019).
    Beyond 4 < R < 13 kpc, the radii of the Mackereth et al. sample, the
    dispersions keep their edge values, blended in within 1 kpc of each
    edge. The mean
    rotation is the radial Jeans solution of
    `thin_disk_velocity_moments`, with
    :math:`\partial \ln \sigma_R^2 / \partial \ln R = -2R / R_\sigma`.
    With the defaults, :math:`\bar{v}_\phi = 191` km/s at the Sun;
    Anguiano et al. (2020) measure 191.8 km/s.

    Parameters
    ----------
    R : array_like of shape `(N,)`
        Galactocentric cylindrical radius in kpc.
    sigma_solar : tuple of float, optional
        Dispersions ``(sigma_R, sigma_phi, sigma_z)`` at the Sun in km/s.
        Default is ``THICK_DISK_SIGMA``, ``(64.68, 50.82, 43.60)``.
    R_sigma : float, optional
        E-folding length of the dispersions in kpc. Default is
        ``THICK_DISK_R_SIGMA``, 15.0.
    R_scale, R_smooth : float, optional
        Radial scale length and smoothing radius of the number density in
        kpc. Defaults are 2.0 and 2.0.
    R_solar, v_circ_solar : optional
        The rotation curve, as in `thin_disk_velocity_moments`.

    Returns
    -------
    mean : `~numpy.ndarray` of shape `(N, 3)`
        Mean velocity ``(v_R, v_phi, v_z)`` in km/s.
    cov : `~numpy.ndarray` of shape `(N, 3, 3)`
        Velocity covariance in (km/s)^2.

    Raises
    ------
    ValueError
        If ``R`` is not a finite, non-negative 1-D array, a dispersion is
        negative, a length scale is not positive, or ``v_circ_solar`` is not
        positive.
    """
    R = _check_radii(R)
    if len(sigma_solar) != 3 or min(sigma_solar) < 0:
        raise ValueError(
            f"sigma_solar must be 3 non-negative dispersions, got {sigma_solar}"
        )
    _check_positive(
        R_sigma=R_sigma,
        R_scale=R_scale,
        R_smooth=R_smooth,
        R_solar=R_solar,
    )
    var_R, var_phi, var_z = (float(s) ** 2 for s in sigma_solar)
    return _thick_disk_array(
        R,
        var_R,
        var_phi,
        var_z,
        float(R_sigma),
        float(R_scale),
        float(R_smooth),
        float(R_solar),
        _vc_solar(v_circ_solar, R_solar),
    )


def halo_velocity_moments(R, Z):
    r"""
    Return the three Gaussian components of the halo velocity distribution.

    The mixture is that of Lancaster et al. (2019, Table 2), fitted to Gaia
    DR2 blue horizontal branch stars at 4.9 < r < 67.9 kpc:

    - an isotropic component with zero mean, weight :math:`f_h`, radial
      dispersion :math:`\sigma_{r,h}` and tangential dispersion
      :math:`\sigma_{t,h}`;
    - two radial lobes (the Gaia-Enceladus debris) with weight
      :math:`(1 - f_h)/2` each, mean :math:`v_r = \pm\delta` and mean
      rotation :math:`\mu_{\phi}`, and dispersions :math:`\sigma_{r,r}`
      and :math:`\sigma_{t,r}`.

    The parameters are interpolated linearly in the spherical radius
    :math:`r` between the centres of the four radial bins and held
    constant beyond them. Each spherical ellipsoid is rotated to
    cylindrical coordinates with :math:`\sin\theta = R/r` and
    :math:`\cos\theta = Z/r`:

    .. math::
        \sigma_R^2 = \sigma_r^2 \sin^2\theta + \sigma_t^2 \cos^2\theta, \quad
        \sigma_z^2 = \sigma_r^2 \cos^2\theta + \sigma_t^2 \sin^2\theta, \quad
        \sigma_{Rz} = (\sigma_r^2 - \sigma_t^2) \sin\theta \cos\theta

    Parameters
    ----------
    R, Z : array_like of shape `(N,)`
        Galactocentric cylindrical radius and height above the midplane in
        kpc.

    Returns
    -------
    weight : `~numpy.ndarray` of shape `(N, 3)`
        Weights of the isotropic component and the lobes at
        :math:`v_r = +\delta` and :math:`-\delta`; each row sums to 1.
    mean : `~numpy.ndarray` of shape `(N, 3, 3)`
        Mean velocity ``(v_R, v_phi, v_z)`` of each component in km/s.
        ``velocity_to_observables(d, coord, mean[:, k], cov[:, k])``
        projects component ``k``.
    cov : `~numpy.ndarray` of shape `(N, 3, 3, 3)`
        Velocity covariance of each component in (km/s)^2.

    Raises
    ------
    ValueError
        If ``R`` and ``Z`` are not finite 1-D arrays of the same length,
        or ``R`` is negative.
    """
    R = _check_radii(R)
    return _halo_array(R, _check_heights(Z, R))


def _sightline(coord, v_sun):
    """Return the unit vectors (e_l, e_b, e_r) as rows, and e . v_sun.

    Heliocentric axes: x toward the Galactic centre, y toward l = 90 deg,
    z toward the north Galactic pole.
    """
    ell, b = radians(coord[0]), radians(coord[1])
    sl, cl, sb, cb = np.sin(ell), np.cos(ell), np.sin(b), np.cos(b)
    E = np.array([[-sl, cl, 0.0], [-sb * cl, -sb * sl, cb], [cb * cl, cb * sl, sb]])
    return E, E @ v_sun


@jit(nopython=True, cache=True)
def _project_moments(E, sun, inv_kd, eRx, eRy, m_R, m_phi, m_z, s_RR, s_pp, s_zz, s_Rz):
    """Project a cylindrical ellipsoid with no R-phi or phi-z covariance.

    ``E`` and ``sun`` come from ``_sightline``; ``(eRx, eRy)`` is the
    radial unit vector at the star. Returns the mean ``(mu_l*, mu_b, v_r)``
    and the covariance entries (00, 01, 02, 11, 12, 22).
    """
    # Row i of the map from (v_R, v_phi, v_z): E_i . (e_R, e_phi, e_z),
    # scaled by 1 / (k d) for the two proper motions. e_phi = (eRy, -eRx, 0)
    # is +y at the Sun.
    a0 = (E[0, 0] * eRx + E[0, 1] * eRy) * inv_kd
    b0 = (E[0, 0] * eRy - E[0, 1] * eRx) * inv_kd
    c0 = E[0, 2] * inv_kd
    a1 = (E[1, 0] * eRx + E[1, 1] * eRy) * inv_kd
    b1 = (E[1, 0] * eRy - E[1, 1] * eRx) * inv_kd
    c1 = E[1, 2] * inv_kd
    a2 = E[2, 0] * eRx + E[2, 1] * eRy
    b2 = E[2, 0] * eRy - E[2, 1] * eRx
    c2 = E[2, 2]
    # Rows of M @ Sigma for this Sigma.
    u0, v0, w0 = a0 * s_RR + c0 * s_Rz, b0 * s_pp, a0 * s_Rz + c0 * s_zz
    u1, v1, w1 = a1 * s_RR + c1 * s_Rz, b1 * s_pp, a1 * s_Rz + c1 * s_zz
    u2, v2, w2 = a2 * s_RR + c2 * s_Rz, b2 * s_pp, a2 * s_Rz + c2 * s_zz
    return (
        a0 * m_R + b0 * m_phi + c0 * m_z - sun[0] * inv_kd,
        a1 * m_R + b1 * m_phi + c1 * m_z - sun[1] * inv_kd,
        a2 * m_R + b2 * m_phi + c2 * m_z - sun[2],
        u0 * a0 + v0 * b0 + w0 * c0,
        u0 * a1 + v0 * b1 + w0 * c1,
        u0 * a2 + v0 * b2 + w0 * c2,
        u1 * a1 + v1 * b1 + w1 * c1,
        u1 * a2 + v1 * b2 + w1 * c2,
        u2 * a2 + v2 * b2 + w2 * c2,
    )


@jit(nopython=True, parallel=True, cache=True)
def _project_array(dists, E, sun, R_solar, mean, cov):
    N = len(dists)
    obs_mean = np.empty((N, 3))
    obs_cov = np.empty((N, 3, 3))
    for n in prange(N):
        d = dists[n]
        px = d * E[2, 0] - R_solar
        py = d * E[2, 1]
        R = sqrt(px * px + py * py)
        if R > 0.0:
            eRx, eRy = px / R, py / R
        else:
            eRx, eRy = -1.0, 0.0
        m0, m1, m2, c00, c01, c02, c11, c12, c22 = _project_moments(
            E,
            sun,
            1.0 / (K_KMS * d),
            eRx,
            eRy,
            mean[n, 0],
            mean[n, 1],
            mean[n, 2],
            cov[n, 0, 0],
            cov[n, 1, 1],
            cov[n, 2, 2],
            cov[n, 0, 2],
        )
        obs_mean[n, 0] = m0
        obs_mean[n, 1] = m1
        obs_mean[n, 2] = m2
        obs_cov[n, 0, 0] = c00
        obs_cov[n, 1, 1] = c11
        obs_cov[n, 2, 2] = c22
        obs_cov[n, 0, 1] = obs_cov[n, 1, 0] = c01
        obs_cov[n, 0, 2] = obs_cov[n, 2, 0] = c02
        obs_cov[n, 1, 2] = obs_cov[n, 2, 1] = c12
    return obs_mean, obs_cov


def velocity_to_observables(dists, coord, mean, cov, R_solar=8.2, v_sun=None):
    r"""
    Project a velocity ellipsoid to proper motions and radial velocity.

    A star at distance :math:`d` toward :math:`(l, b)` with Galactocentric
    velocity :math:`\mathbf{v}` has heliocentric velocity
    :math:`\mathbf{v} - \mathbf{v}_\odot` and observables

    .. math::
        \mu_{l*} = \frac{\hat{e}_l \cdot (\mathbf{v} - \mathbf{v}_\odot)}{k d}, \quad
        \mu_b = \frac{\hat{e}_b \cdot (\mathbf{v} - \mathbf{v}_\odot)}{k d}, \quad
        v_r = \hat{e}_r \cdot (\mathbf{v} - \mathbf{v}_\odot)

    with :math:`k = 4.74047` km/s per (kpc mas/yr). The map is linear in
    :math:`\mathbf{v}`, so a Gaussian velocity distribution stays Gaussian:
    with Jacobian :math:`M`, the covariance is :math:`M \Sigma M^T`. The
    Sun is in the Galactic midplane for this projection; its height
    changes the directions by less than ``Z_solar / R_solar`` radians.

    Parameters
    ----------
    dists : array_like of shape `(N,)`
        Heliocentric distances in kpc.
    coord : tuple of float
        Galactic coordinates ``(l, b)`` in degrees.
    mean : array_like of shape `(N, 3)`
        Mean velocity ``(v_R, v_phi, v_z)`` at each distance, in km/s.
    cov : array_like of shape `(N, 3, 3)`
        Velocity covariance at each distance, in (km/s)^2, with no
        ``(v_R, v_phi)`` or ``(v_phi, v_z)`` covariance, as returned by
        `thin_disk_velocity_moments`, `thick_disk_velocity_moments` and each
        component of `halo_velocity_moments`.
    R_solar : float, optional
        Distance of the Sun from the Galactic centre in kpc. Default is 8.2.
    v_sun : array_like of shape `(3,)`, optional
        Solar velocity ``(U, V, W)`` in km/s. Default is
        ``solar_velocity(R_solar)``.

    Returns
    -------
    obs_mean : `~numpy.ndarray` of shape `(N, 3)`
        Mean ``(mu_l*, mu_b, v_r)`` in (mas/yr, mas/yr, km/s).
    obs_cov : `~numpy.ndarray` of shape `(N, 3, 3)`
        Covariance of ``(mu_l*, mu_b, v_r)`` in the same units.

    Raises
    ------
    ValueError
        If a distance or ``R_solar`` is not positive, the shapes do not
        match, ``mean`` or ``cov`` is not finite, ``coord`` is not a finite
        ``(l, b)`` with ``|b| <= 90``, or ``cov`` is
        not symmetric and positive semi-definite or has ``(v_R, v_phi)`` or
        ``(v_phi, v_z)`` covariance.
    """
    dists = np.asarray(dists, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if dists.ndim != 1 or not np.all(dists > 0):
        raise ValueError("dists must be a 1-D array of positive distances")
    N = len(dists)
    if mean.shape != (N, 3) or cov.shape != (N, 3, 3):
        raise ValueError(
            f"mean and cov must have shapes ({N}, 3) and ({N}, 3, 3), "
            f"got {mean.shape} and {cov.shape}"
        )
    if np.any(cov[:, [0, 1, 1, 2], [1, 0, 2, 1]] != 0):
        raise ValueError("cov must have no (v_R, v_phi) or (v_phi, v_z) covariance")
    if np.any(cov[:, 0, 2] != cov[:, 2, 0]):
        raise ValueError("cov must be symmetric")
    var = np.diagonal(cov, axis1=1, axis2=2)
    if not (
        np.all(np.isfinite(cov))
        and np.all(var >= 0)
        and np.all(cov[:, 0, 2] ** 2 <= var[:, 0] * var[:, 2])
    ):
        raise ValueError("cov must be finite and positive semi-definite")
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean must be finite")
    if len(coord) != 2 or not (np.isfinite(coord[0]) and abs(coord[1]) <= 90):
        raise ValueError(
            f"coord must be (l, b) with finite l and |b| <= 90, got {coord}"
        )
    if not R_solar > 0:
        raise ValueError(f"R_solar must be positive, got {R_solar}")
    if v_sun is None:
        v_sun = solar_velocity(R_solar)
    v_sun = np.asarray(v_sun, dtype=np.float64)
    if v_sun.shape != (3,):
        raise ValueError(f"v_sun must have shape (3,), got {v_sun.shape}")
    E, sun = _sightline(coord, v_sun)
    return _project_array(dists, E, sun, float(R_solar), mean, cov)
