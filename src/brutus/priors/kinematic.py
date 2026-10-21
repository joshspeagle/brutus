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
# Slope of the circular velocity curve in km/s/kpc (Eilers et al. 2019).
DVCIRC_DR = -1.7

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

# Thick-disk (high-[alpha/Fe]) dispersions (sigma_R, sigma_phi, sigma_z)
# in km/s (Anguiano et al. 2020, Table 1) and the e-folding length of the
# dispersions in kpc (Mackereth et al. 2019, h_sigma_R).
THICK_DISK_SIGMA = (64.68, 50.82, 43.60)
THICK_DISK_R_SIGMA = 15.0

# Halo mixture by Galactocentric radius (Lancaster et al. 2019, Table 2):
# centres of the radial bins in kpc; weight of the isotropic component;
# its radial and tangential dispersions; the dispersions of the two equal
# radial lobes, their separation from v_r = 0 and their mean v_phi; km/s.
HALO_R = (9.0, 16.15, 24.6, 48.965)
HALO_F_ISO = (0.52, 0.55, 0.55, 0.96)
HALO_SIGMA_R_ISO = (129.24, 122.03, 113.62, 95.21)
HALO_SIGMA_T_ISO = (114.27, 105.41, 96.72, 79.33)
HALO_SIGMA_R_LOBE = (109.93, 78.21, 62.91, 176.36)
HALO_SIGMA_T_LOBE = (58.18, 34.02, 29.14, 145.69)
HALO_DELTA_LOBE = (104.16, 98.60, 67.75, 82.87)
HALO_V_PHI_LOBE = (-4.49, 11.54, 14.13, -27.39)


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
def _circular_velocity(R, R_solar, vc_solar, dvc_dR, R_smooth):
    """Return v_c(R) and d ln v_c / d ln R.

    The linear curve holds at 5-25 kpc (Eilers et al. 2019); the taper
    tanh(R / R_smooth) takes it to 0 at the centre within the smoothing
    radius of the disk density.
    """
    v_lin = vc_solar + dvc_dR * (R - R_solar)
    if v_lin <= 0.0:
        return 0.0, 0.0
    x = R / R_smooth
    t = tanh(x)
    # d ln tanh(x) / d ln x = x (1 - t^2) / t, which tends to 1 as x -> 0.
    dln_taper = x * (1.0 - t * t) / t if x > 0.0 else 1.0
    return v_lin * t, R * dvc_dR / v_lin + dln_taper


@jit(nopython=True, cache=True)
def _jeans_vphi(R, R_eff, var_R, var_phi, dlnvar_dlnR, R_scale, vc):
    """Mean rotation from the radial Jeans equation, clipped to [0, vc].

    Binney & Tremaine (2008), eq. 4.228, with a cylindrically aligned
    ellipsoid and the density gradient of logn_disk,
    d ln(nu) / d ln(R) = -R^2 / (R_scale * R_eff).
    """
    vphi2 = vc * vc - var_phi + var_R * (1.0 - R * R / (R_scale * R_eff) + dlnvar_dlnR)
    # Where the dispersions exceed what rotation can support, the equation
    # has no solution; near the centre, where the density gradient
    # vanishes, it gives rotation faster than circular.
    return min(sqrt(max(vphi2, 0.0)), vc)


@jit(nopython=True, cache=True)
def _sharma_sigma(L, L_solar, age, feh, abs_z, v):
    """Return sigma and d ln sigma / d ln L_z (Sharma et al. 2021, eqs. 3-7)
    for v = 0 (sigma_R) or v = 1 (sigma_z)."""
    s0, beta = THIN_DISK_SIGMA0[v], THIN_DISK_BETA[v]
    lam, alpha = THIN_DISK_LAMBDA_L[v], THIN_DISK_ALPHA_L[v]
    g_feh, g_z = THIN_DISK_GAMMA_FEH[v], THIN_DISK_GAMMA_Z[v]
    u = L / L_solar
    e = exp(-(L - L_solar) / lam)
    f_L = (alpha * u * u + e) / (1.0 + alpha)
    dlnf_dlnL = (2.0 * alpha * u * u - (L / lam) * e) / (1.0 + alpha) / f_L
    sigma = (
        s0
        * ((age + 0.1) / 10.1) ** beta
        * f_L
        * (1.0 + g_feh * feh)
        * (1.0 + g_z * abs_z)
    )
    return sigma, dlnf_dlnL


@jit(nopython=True, cache=True)
def _thin_disk_moments(
    R,
    R_eff,
    abs_z,
    age,
    feh,
    R_scale,
    R_smooth,
    R_solar,
    vc_solar,
    dvc_dR,
):
    """Return (v_phi, var_R, var_phi, var_z, cov_Rz) of the thin disk.

    The angular momentum is that of a circular orbit, L_z = R v_c(R), so
    that L_z = L_z,sun at R_solar. sigma_phi follows the epicyclic ratio
    sigma_phi^2 / sigma_R^2 = (1 + d ln v_c / d ln R) / 2.
    ``R_eff = sqrt(R^2 + R_smooth^2)`` is the smoothed radius of the
    ``logn_disk`` density, which the Galactic prior kernel already holds.
    """
    vc, dlnvc = _circular_velocity(R, R_solar, vc_solar, dvc_dR, R_smooth)
    vc_sun = _circular_velocity(R_solar, R_solar, vc_solar, dvc_dR, R_smooth)[0]
    L = max(R * vc, 1e-10)
    L_solar = R_solar * vc_sun
    sig_R, dlnf_R = _sharma_sigma(L, L_solar, age, feh, abs_z, 0)
    sig_z = _sharma_sigma(L, L_solar, age, feh, abs_z, 1)[0]
    var_R = sig_R * sig_R
    var_phi = 0.5 * var_R * (1.0 + dlnvc)
    dlnvar_dlnR = 2.0 * dlnf_R * (1.0 + dlnvc)
    vphi = _jeans_vphi(R, R_eff, var_R, var_phi, dlnvar_dlnR, R_scale, vc)
    return vphi, var_R, var_phi, sig_z * sig_z, 0.0


@jit(nopython=True, cache=True)
def _thick_disk_moments(
    R,
    R_eff,
    var_R0,
    var_phi0,
    var_z0,
    R_sigma,
    R_scale,
    R_smooth,
    R_solar,
    vc_solar,
    dvc_dR,
):
    """Return (v_phi, var_R, var_phi, var_z, cov_Rz) of the thick disk."""
    vc = _circular_velocity(R, R_solar, vc_solar, dvc_dR, R_smooth)[0]
    f = exp(-2.0 * (R - R_solar) / R_sigma)
    var_R = var_R0 * f
    vphi = _jeans_vphi(R, R_eff, var_R, var_phi0 * f, -2.0 * R / R_sigma, R_scale, vc)
    return vphi, var_R, var_phi0 * f, var_z0 * f, 0.0


@jit(nopython=True, cache=True)
def _interp(x, xs, ys):
    """Linear interpolation in x, constant beyond the ends of xs."""
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


@jit(nopython=True, cache=True)
def _halo_component(R, Z, k, table):
    """Return (weight, m_R, m_phi, m_z, var_R, var_phi, var_z, cov_Rz) of
    halo component k: 0 is isotropic, 1 and 2 are the lobes at v_r = +delta
    and -delta. ``table`` holds the HALO_* rows, in the order of HALO_R.
    """
    r = sqrt(R * R + Z * Z)
    if r > 0.0:
        s, c = R / r, Z / r
    else:
        s, c = 1.0, 0.0
    f_iso = _interp(r, table[0], table[1])
    if k == 0:
        weight, m_r, m_phi = f_iso, 0.0, 0.0
        sig_r = _interp(r, table[0], table[2])
        sig_t = _interp(r, table[0], table[3])
    else:
        weight = 0.5 * (1.0 - f_iso)
        delta = _interp(r, table[0], table[6])
        m_r = delta if k == 1 else -delta
        m_phi = _interp(r, table[0], table[7])
        sig_r = _interp(r, table[0], table[4])
        sig_t = _interp(r, table[0], table[5])
    var_r, var_t = sig_r * sig_r, sig_t * sig_t
    # v_R = v_r sin(theta) + v_theta cos(theta),
    # v_z = v_r cos(theta) - v_theta sin(theta), theta from the +z axis.
    return (
        weight,
        m_r * s,
        m_phi,
        m_r * c,
        var_r * s * s + var_t * c * c,
        var_t,
        var_r * c * c + var_t * s * s,
        (var_r - var_t) * s * c,
    )


@jit(nopython=True, cache=True)
def _store(mean, cov, i, m_R, m_phi, m_z, var_R, var_phi, var_z, cov_Rz):
    mean[i, 0] = m_R
    mean[i, 1] = m_phi
    mean[i, 2] = m_z
    cov[i, 0, 0] = var_R
    cov[i, 1, 1] = var_phi
    cov[i, 2, 2] = var_z
    cov[i, 0, 2] = cov_Rz
    cov[i, 2, 0] = cov_Rz


@jit(nopython=True, parallel=True, cache=True)
def _thin_disk_array(R, Z, age, feh, R_scale, R_smooth, R_solar, vc_solar, dvc_dR):
    N = len(R)
    mean = np.zeros((N, 3))
    cov = np.zeros((N, 3, 3))
    for i in prange(N):
        R_eff = sqrt(R[i] * R[i] + R_smooth * R_smooth)
        vphi, var_R, var_phi, var_z, cov_Rz = _thin_disk_moments(
            R[i],
            R_eff,
            abs(Z[i]),
            age[i],
            feh[i],
            R_scale,
            R_smooth,
            R_solar,
            vc_solar,
            dvc_dR,
        )
        _store(mean, cov, i, 0.0, vphi, 0.0, var_R, var_phi, var_z, cov_Rz)
    return mean, cov


@jit(nopython=True, parallel=True, cache=True)
def _thick_disk_array(
    R, var_R0, var_phi0, var_z0, R_sigma, R_scale, R_smooth, R_solar, vc_solar, dvc_dR
):
    N = len(R)
    mean = np.zeros((N, 3))
    cov = np.zeros((N, 3, 3))
    for i in prange(N):
        R_eff = sqrt(R[i] * R[i] + R_smooth * R_smooth)
        vphi, var_R, var_phi, var_z, cov_Rz = _thick_disk_moments(
            R[i],
            R_eff,
            var_R0,
            var_phi0,
            var_z0,
            R_sigma,
            R_scale,
            R_smooth,
            R_solar,
            vc_solar,
            dvc_dR,
        )
        _store(mean, cov, i, 0.0, vphi, 0.0, var_R, var_phi, var_z, cov_Rz)
    return mean, cov


@jit(nopython=True, parallel=True, cache=True)
def _halo_array(R, Z, table):
    N = len(R)
    weight = np.empty((N, 3))
    mean = np.zeros((N, 3, 3))
    cov = np.zeros((N, 3, 3, 3))
    for i in prange(N):
        for k in range(3):
            w, m_R, m_phi, m_z, var_R, var_phi, var_z, cov_Rz = _halo_component(
                R[i], Z[i], k, table
            )
            weight[i, k] = w
            _store(
                mean[:, k], cov[:, k], i, m_R, m_phi, m_z, var_R, var_phi, var_z, cov_Rz
            )
    return weight, mean, cov


def _check_positions(R, Z):
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if R.ndim != 1 or R.shape != Z.shape:
        raise ValueError(
            f"R and Z must be 1-D with the same shape, got {R.shape} and {Z.shape}"
        )
    if not np.all(np.isfinite(R) & (R >= 0)):
        raise ValueError("R must be finite and non-negative")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z must be finite")
    return R, Z


def _check_rotation(R_scale, R_smooth, R_solar):
    for name, value in (
        ("R_scale", R_scale),
        ("R_smooth", R_smooth),
        ("R_solar", R_solar),
    ):
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}")


def _vc_solar(v_circ_solar, R_solar):
    if v_circ_solar is None:
        return solar_velocity(R_solar)[1] - V_PEC_SUN
    return float(v_circ_solar)


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
    3 < R < 20 kpc. The azimuthal dispersion follows the epicyclic ratio
    :math:`\sigma_\phi^2 / \sigma_R^2 = (1 + d\ln v_c / d\ln R) / 2`.

    The mean rotation is the radial Jeans solution (Binney & Tremaine 2008,
    eq. 4.228) for a cylindrically aligned ellipsoid and the density
    gradient of `~brutus.priors.galactic.logn_disk`:

    .. math::
        \bar{v}_\phi^2 = v_c^2 - \sigma_\phi^2
        + \sigma_R^2 \left(1 + \frac{\partial \ln \nu}{\partial \ln R}
        + \frac{\partial \ln \sigma_R^2}{\partial \ln R}\right)

    with :math:`v_c(R) = [v_{c,\odot} + (dv_c/dR)(R - R_\odot)]
    \tanh(R / R_\mathrm{smooth})`. The taper takes :math:`v_c` to 0 at the
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
    age : array_like of shape `(N,)`
        Stellar age in Gyr.
    feh : array_like of shape `(N,)`
        Metallicity [Fe/H].
    R_scale : float, optional
        Radial scale length of the number density in kpc, as in
        `~brutus.priors.galactic.logn_disk`. Default is 2.6.
    R_smooth : float, optional
        Smoothing radius of the number density and of the taper of the
        circular velocity, in kpc. Default is 2.0.
    R_solar : float, optional
        Distance of the Sun from the Galactic centre in kpc. Default is 8.2.
    v_circ_solar : float, optional
        Circular velocity at ``R_solar`` before the taper, in km/s. Default
        is the solar V from ``solar_velocity(R_solar)`` minus the peculiar
        12.24 km/s of Schönrich et al. (2010): 236.97 km/s for
        ``R_solar=8.2``.
    dvcirc_dR : float, optional
        Slope of the circular velocity curve in km/s/kpc. Default is -1.7
        (Eilers et al. 2019).

    Returns
    -------
    mean : `~numpy.ndarray` of shape `(N, 3)`
        Mean velocity ``(v_R, v_phi, v_z)`` in km/s.
    cov : `~numpy.ndarray` of shape `(N, 3, 3)`
        Velocity covariance in (km/s)^2.

    Raises
    ------
    ValueError
        If the inputs are not finite 1-D arrays of one length, ``R`` or
        ``age`` is negative, ``feh`` makes a dispersion non-positive, or a
        length scale is not positive.
    """
    R, Z = _check_positions(R, Z)
    age = np.asarray(age, dtype=np.float64)
    feh = np.asarray(feh, dtype=np.float64)
    if age.shape != R.shape or feh.shape != R.shape:
        raise ValueError("age and feh must have the shape of R")
    if not np.all(np.isfinite(age) & (age >= 0)):
        raise ValueError("age must be finite and non-negative")
    # 1 + gamma_[Fe/H] [Fe/H] > 0 for both dispersions.
    if not np.all(np.isfinite(feh) & (feh < 1.0 / 0.52)):
        raise ValueError("feh must be finite and below 1.92")
    _check_rotation(R_scale, R_smooth, R_solar)
    return _thin_disk_array(
        R,
        Z,
        age,
        feh,
        float(R_scale),
        float(R_smooth),
        float(R_solar),
        _vc_solar(v_circ_solar, R_solar),
        float(dvcirc_dR),
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
    The mean rotation is the radial Jeans solution of
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
    R_scale : float, optional
        Radial scale length of the number density in kpc, as in
        `~brutus.priors.galactic.logn_disk`. Default is 2.0.
    R_smooth : float, optional
        Smoothing radius of the number density and of the taper of the
        circular velocity, in kpc. Default is 2.0.
    R_solar : float, optional
        Distance of the Sun from the Galactic centre in kpc. Default is 8.2.
    v_circ_solar : float, optional
        Circular velocity at ``R_solar`` before the taper, in km/s. Default
        is the solar V from ``solar_velocity(R_solar)`` minus the peculiar
        12.24 km/s of Schönrich et al. (2010): 236.97 km/s for
        ``R_solar=8.2``.
    dvcirc_dR : float, optional
        Slope of the circular velocity curve in km/s/kpc. Default is -1.7
        (Eilers et al. 2019).

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
        negative, or a length scale is not positive.
    """
    R, _ = _check_positions(R, np.zeros(np.shape(R)))
    if len(sigma_solar) != 3 or min(sigma_solar) < 0:
        raise ValueError(
            f"sigma_solar must be 3 non-negative dispersions, got {sigma_solar}"
        )
    if not R_sigma > 0:
        raise ValueError(f"R_sigma must be positive, got {R_sigma}")
    _check_rotation(R_scale, R_smooth, R_solar)
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
        float(dvcirc_dR),
    )


_HALO_TABLE = np.array(
    [
        HALO_R,
        HALO_F_ISO,
        HALO_SIGMA_R_ISO,
        HALO_SIGMA_T_ISO,
        HALO_SIGMA_R_LOBE,
        HALO_SIGMA_T_LOBE,
        HALO_DELTA_LOBE,
        HALO_V_PHI_LOBE,
    ]
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
        Weights of the isotropic component and the two lobes; each row
        sums to 1.
    mean : `~numpy.ndarray` of shape `(N, 3, 3)`
        Mean velocity ``(v_R, v_phi, v_z)`` of each component in km/s.
    cov : `~numpy.ndarray` of shape `(N, 3, 3, 3)`
        Velocity covariance of each component in (km/s)^2.

    Raises
    ------
    ValueError
        If ``R`` and ``Z`` are not finite 1-D arrays of the same length,
        or ``R`` is negative.
    """
    R, Z = _check_positions(R, Z)
    return _halo_array(R, Z, _HALO_TABLE)


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
        `disk_velocity_moments` and `halo_velocity_moments`.
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
        match, or ``cov`` is not symmetric or has ``(v_R, v_phi)`` or
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
    if not R_solar > 0:
        raise ValueError(f"R_solar must be positive, got {R_solar}")
    if v_sun is None:
        v_sun = solar_velocity(R_solar)
    v_sun = np.asarray(v_sun, dtype=np.float64)
    if v_sun.shape != (3,):
        raise ValueError(f"v_sun must have shape (3,), got {v_sun.shape}")
    E, sun = _sightline(coord, v_sun)
    return _project_array(dists, E, sun, float(R_solar), mean, cov)
