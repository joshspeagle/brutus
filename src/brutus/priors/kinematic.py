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

from math import exp, radians, sqrt, tanh

import numpy as np
from numba import jit, prange

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
# Slope of the circular velocity curve in km/s/kpc and the largest radius
# in kpc of the data it fits (Eilers et al. 2019); v_c is flat beyond it.
DVCIRC_DR = -1.7
VC_LINEAR_R_MAX = 25.0
# Radius in kpc of the taper that takes v_c to 0 at the Galactic centre.
VC_TAPER = 2.0

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
# Beyond them the dispersions keep their values at the edges.
THIN_DISK_R_RANGE = (3.0, 20.0)

# Thick-disk (high-[alpha/Fe]) dispersions (sigma_R, sigma_phi, sigma_z)
# in km/s (Anguiano et al. 2020, Table 1) and the e-folding length of the
# dispersions in kpc (Mackereth et al. 2019, h_sigma_R).
THICK_DISK_SIGMA = (64.68, 50.82, 43.60)
THICK_DISK_R_SIGMA = 15.0
# Radii in kpc of the Mackereth et al. (2019) sample. Beyond them the
# thick-disk dispersions keep their values at the edges.
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


@jit(nopython=True, cache=True)
def _circular_velocity(R, R_solar, vc_solar, dvc_dR, R_taper):
    """Return v_c(R) and d ln v_c / d ln R.

    The linear curve fits 5-25 kpc (Eilers et al. 2019) and is flat beyond
    VC_LINEAR_R_MAX; the taper tanh(R / R_taper) takes it to 0 at the
    centre. The public functions check that v_lin is positive.
    """
    if R < VC_LINEAR_R_MAX:
        v_lin = vc_solar + dvc_dR * (R - R_solar)
        dln_lin = R * dvc_dR / v_lin
    else:
        v_lin = vc_solar + dvc_dR * (VC_LINEAR_R_MAX - R_solar)
        dln_lin = 0.0
    x = R / R_taper
    t = tanh(x)
    # d ln tanh(x) / d ln x = x (1 - t^2) / t, which tends to 1 as x -> 0.
    dln_taper = x * (1.0 - t * t) / t if x > 0.0 else 1.0
    return v_lin * t, dln_lin + dln_taper


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
def _thin_disk_moments(
    R, abs_z, sig_R0, sig_z0, vc, dlnvc, L_solar, L_range, dlnnu_dlnR
):
    """Return (v_phi, var_R, var_phi, var_z) of the thin disk.

    ``sig_R0`` and ``sig_z0`` come from ``_thin_disk_sigma0``. The angular
    momentum is that of a circular orbit, L_z = R v_c(R), and ``L_solar``
    is R_solar v_c(R_solar), so f_L = 1 at the Sun. ``L_range`` holds L_z
    at the ends of THIN_DISK_R_RANGE; beyond them L_z keeps its edge value.
    sigma_phi follows the epicyclic ratio
    sigma_phi^2 / sigma_R^2 = (1 + d ln v_c / d ln R) / 2.
    """
    L = R * vc
    inside = L_range[0] < L < L_range[1]
    L = min(max(L, L_range[0]), L_range[1])
    u2 = (L / L_solar) ** 2
    a_R, a_z = THIN_DISK_ALPHA_L
    lam_R, lam_z = THIN_DISK_LAMBDA_L
    e_R = exp(-(L - L_solar) / lam_R)
    e_z = exp(-(L - L_solar) / lam_z)
    f_R = (a_R * u2 + e_R) / (1.0 + a_R)
    f_z = (a_z * u2 + e_z) / (1.0 + a_z)
    dlnf_R = (2.0 * a_R * u2 - (L / lam_R) * e_R) / ((1.0 + a_R) * f_R)
    sig_R = sig_R0 * f_R * (1.0 + THIN_DISK_GAMMA_Z[0] * abs_z)
    sig_z = sig_z0 * f_z * (1.0 + THIN_DISK_GAMMA_Z[1] * abs_z)
    var_R = sig_R * sig_R
    var_phi = 0.5 * var_R * (1.0 + dlnvc)
    dlnvar_dlnR = 2.0 * dlnf_R * (1.0 + dlnvc) if inside else 0.0
    vphi = _jeans_vphi(var_R, var_phi, dlnvar_dlnR, dlnnu_dlnR, vc)
    return vphi, var_R, var_phi, sig_z * sig_z


@jit(nopython=True, cache=True)
def _thick_disk_moments(R, vc, var_R0, var_phi0, var_z0, R_sigma, R_solar, dlnnu_dlnR):
    """Return (v_phi, var_R, var_phi, var_z) of the thick disk.

    Beyond THICK_DISK_R_RANGE the dispersions keep their edge values.
    """
    lo, hi = THICK_DISK_R_RANGE
    f = exp(-2.0 * (min(max(R, lo), hi) - R_solar) / R_sigma)
    var_R = var_R0 * f
    var_phi = var_phi0 * f
    dlnvar_dlnR = -2.0 * R / R_sigma if lo < R < hi else 0.0
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
def _thin_disk_array(
    R, Z, age, feh, R_scale, R_smooth, R_solar, vc_solar, dvc_dR, R_taper
):
    N = len(R)
    mean = np.zeros((N, 3))
    cov = np.zeros((N, 3, 3))
    lo, hi = THIN_DISK_R_RANGE
    L_solar = (
        R_solar * _circular_velocity(R_solar, R_solar, vc_solar, dvc_dR, R_taper)[0]
    )
    L_lo = lo * _circular_velocity(lo, R_solar, vc_solar, dvc_dR, R_taper)[0]
    L_hi = hi * _circular_velocity(hi, R_solar, vc_solar, dvc_dR, R_taper)[0]
    for i in prange(N):
        vc, dlnvc = _circular_velocity(R[i], R_solar, vc_solar, dvc_dR, R_taper)
        sig_R0, sig_z0 = _thin_disk_sigma0(age[i], feh[i])
        vphi, var_R, var_phi, var_z = _thin_disk_moments(
            R[i],
            abs(Z[i]),
            sig_R0,
            sig_z0,
            vc,
            dlnvc,
            L_solar,
            (L_lo, L_hi),
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
    dvc_dR,
    R_taper,
):
    N = len(R)
    mean = np.zeros((N, 3))
    cov = np.zeros((N, 3, 3))
    for i in prange(N):
        vc = _circular_velocity(R[i], R_solar, vc_solar, dvc_dR, R_taper)[0]
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


def _vc_solar(v_circ_solar, R_solar, dvcirc_dR):
    """Return v_c at the Sun, after checking that the curve stays positive."""
    if v_circ_solar is None:
        v_circ_solar = solar_velocity(R_solar)[1] - V_PEC_SUN
    v_circ_solar, dvcirc_dR = float(v_circ_solar), float(dvcirc_dR)
    if not (np.isfinite(v_circ_solar) and np.isfinite(dvcirc_dR)):
        raise ValueError("v_circ_solar and dvcirc_dR must be finite")
    # The linear part runs from R = 0 to VC_LINEAR_R_MAX.
    if (
        min(v_circ_solar + dvcirc_dR * (x - R_solar) for x in (0.0, VC_LINEAR_R_MAX))
        <= 0
    ):
        raise ValueError(
            "v_circ_solar and dvcirc_dR must give a positive circular velocity "
            f"at 0 < R < {VC_LINEAR_R_MAX} kpc"
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
    dvcirc_dR=DVCIRC_DR,
    R_taper=VC_TAPER,
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
    angular momentum is that of a circular orbit, :math:`L_z = R\,v_c(R)`,
    so :math:`f_{L_z} = 1` at the Sun. The relations were fitted at
    3 < R < 20 kpc; beyond that range :math:`L_z` keeps its value at the
    edge, so the dispersions do not extrapolate. The azimuthal dispersion follows the epicyclic ratio
    :math:`\sigma_\phi^2 / \sigma_R^2 = (1 + d\ln v_c / d\ln R) / 2`.

    The mean rotation is the radial Jeans solution (Binney & Tremaine 2008,
    eq. 4.228) for a cylindrically aligned ellipsoid and the density
    gradient of `~brutus.priors.galactic.logn_disk`:

    .. math::
        \bar{v}_\phi^2 = v_c^2 - \sigma_\phi^2
        + \sigma_R^2 \left(1 + \frac{\partial \ln \nu}{\partial \ln R}
        + \frac{\partial \ln \sigma_R^2}{\partial \ln R}\right)

    with :math:`v_c(R) = [v_{c,\odot} + (dv_c/dR)(R - R_\odot)]
    \tanh(R / R_\mathrm{taper})`. The taper takes :math:`v_c` to 0 at the
    Galactic centre and changes it by less than 0.1% at the Sun. The mean
    rotation is clipped to :math:`[0, v_c]`: the Jeans equation has no
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
        Circular velocity at ``R_solar`` before the taper, in km/s. Default
        is the solar V from ``solar_velocity(R_solar)`` minus the peculiar
        12.24 km/s of Schönrich et al. (2010): 236.97 km/s for
        ``R_solar=8.2``.
    dvcirc_dR : float, optional
        Slope of the circular velocity curve in km/s/kpc. Default is -1.7
        (Eilers et al. 2019). The curve is flat beyond 25 kpc, the edge of
        the data that slope fits.
    R_taper : float, optional
        Radius of the taper of the circular velocity at the centre, in kpc.
        Default is ``VC_TAPER``, 2.0.

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
        length scale is not positive, or the circular velocity is not
        positive at 0 < R < 25 kpc.
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
    _check_positive(
        R_scale=R_scale, R_smooth=R_smooth, R_solar=R_solar, R_taper=R_taper
    )
    return _thin_disk_array(
        R,
        Z,
        age,
        feh,
        float(R_scale),
        float(R_smooth),
        float(R_solar),
        _vc_solar(v_circ_solar, R_solar, dvcirc_dR),
        float(dvcirc_dR),
        float(R_taper),
    )


def thick_disk_velocity_moments(
    R,
    sigma_solar=THICK_DISK_SIGMA,
    R_sigma=THICK_DISK_R_SIGMA,
    R_scale=2.0,
    R_smooth=2.0,
    R_solar=8.2,
    v_circ_solar=None,
    dvcirc_dR=DVCIRC_DR,
    R_taper=VC_TAPER,
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
    dispersions keep their values at the edges. The mean rotation is the radial Jeans solution of
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
    R_solar, v_circ_solar, dvcirc_dR, R_taper : optional
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
        negative, a length scale is not positive, or the circular velocity
        is not positive at 0 < R < 25 kpc.
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
        R_taper=R_taper,
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
        _vc_solar(v_circ_solar, R_solar, dvcirc_dR),
        float(dvcirc_dR),
        float(R_taper),
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
