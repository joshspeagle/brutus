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

See Also
--------
brutus.priors.galactic : Number densities of the same components

Examples
--------
>>> import numpy as np
>>> from brutus.priors.kinematic import (
...     disk_velocity_moments, velocity_to_observables)
>>> from brutus.utils.math import galactic_to_galactocentric_cyl
>>>
>>> dists = np.array([0.5, 1.0, 2.0])  # kpc
>>> coord = (90.0, 30.0)  # (l, b) in degrees
>>> R, Z = galactic_to_galactocentric_cyl(dists, *coord)
>>> mean, cov = disk_velocity_moments(R)
>>> obs_mean, obs_cov = velocity_to_observables(dists, coord, mean, cov)
"""

from math import exp, radians, sqrt, tanh

import numpy as np
from numba import jit, prange

__all__ = [
    "solar_velocity",
    "disk_velocity_moments",
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

# Local dispersions (sigma_R, sigma_phi, sigma_z) in km/s (Bensby et al.
# 2014). The thin-disk variances fall off with the density scale length
# (2.6 kpc), as for an isothermal disk of constant scale height. The
# thick-disk scale length of 7.6 kpc gives its measured solar lag of 46 km/s.
THIN_DISK_SIGMA = (35.0, 20.0, 16.0)
THIN_DISK_R_SIGMA = 2.6
THICK_DISK_SIGMA = (67.0, 38.0, 35.0)
THICK_DISK_R_SIGMA = 7.6
# Halo dispersions (sigma_r, sigma_theta, sigma_phi) in km/s and mean
# rotation (Bond et al. 2010).
HALO_SIGMA = (141.0, 75.0, 85.0)
HALO_V_PHI = 0.0


def solar_velocity(R_solar=8.2, pm_sgra=PM_SGRA, U_sun=U_SUN):
    r"""
    Return the velocity of the Sun relative to the Galactic centre.

    Sgr A* is taken to be at rest at the Galactic centre, so its proper
    motion is the reflex of the solar motion:

    .. math::
        V_\odot = -k R_\odot \mu_{l,\mathrm{Sgr\,A^*}}, \qquad
        W_\odot = -k R_\odot \mu_{b,\mathrm{Sgr\,A^*}}

    with :math:`k = 4.74047` km/s per (kpc mas/yr). This ties the solar
    velocity to ``R_solar``: use the same ``R_solar`` in
    `disk_velocity_moments` and `velocity_to_observables`.

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
def _disk_moments(
    R,
    R_eff,
    var_R0,
    var_phi0,
    var_z0,
    R_scale,
    R_smooth,
    R_sigma,
    R_solar,
    vc_solar,
    dvc_dR,
):
    """Return (v_phi, var_R, var_phi, var_z, cov_Rz) of a disk at radius R.

    ``R_eff = sqrt(R^2 + R_smooth^2)`` is the smoothed radius of the
    ``logn_disk`` density, which the Galactic prior kernel already holds.
    """
    f = exp(-(R - R_solar) / R_sigma)
    var_R = var_R0 * f
    # The linear curve holds at 5-25 kpc (Eilers et al. 2019); the taper
    # takes it to 0 at the centre within the density's smoothing radius.
    vc = max(vc_solar + dvc_dR * (R - R_solar), 0.0) * tanh(R / R_smooth)
    # Radial Jeans equation with a cylindrically aligned ellipsoid
    # (Binney & Tremaine 2008, eq. 4.228), with the density gradient of
    # logn_disk, d ln(nu) / d ln(R) = -R^2 / (R_scale * R_eff), and
    # d ln(var_R) / d ln(R) = -R / R_sigma.
    vphi2 = (
        vc * vc - var_phi0 * f + var_R * (1.0 - R * R / (R_scale * R_eff) - R / R_sigma)
    )
    # Where the dispersions exceed what rotation can support, the Jeans
    # equation gives vphi2 < 0: the disk does not rotate there. Near the
    # centre, where the density gradient vanishes, it can exceed vc^2.
    vphi = min(sqrt(max(vphi2, 0.0)), vc)
    return vphi, var_R, var_phi0 * f, var_z0 * f, 0.0


@jit(nopython=True, cache=True)
def _halo_moments(R, Z, var_r, var_theta, var_phi, v_phi):
    """Return (v_phi, var_R, var_phi, var_z, cov_Rz) of a spherical ellipsoid."""
    r = sqrt(R * R + Z * Z)
    if r > 0.0:
        inv_r = 1.0 / r
        s = R * inv_r
        c = Z * inv_r
    else:
        s = 1.0
        c = 0.0
    # v_R = v_r sin(theta) + v_theta cos(theta),
    # v_z = v_r cos(theta) - v_theta sin(theta), theta from the +z axis.
    var_R = var_r * s * s + var_theta * c * c
    var_z = var_r * c * c + var_theta * s * s
    return v_phi, var_R, var_phi, var_z, (var_r - var_theta) * s * c


@jit(nopython=True, cache=True)
def _store(mean, cov, i, v_phi, var_R, var_phi, var_z, cov_Rz):
    mean[i, 1] = v_phi
    cov[i, 0, 0] = var_R
    cov[i, 1, 1] = var_phi
    cov[i, 2, 2] = var_z
    cov[i, 0, 2] = cov_Rz
    cov[i, 2, 0] = cov_Rz


@jit(nopython=True, parallel=True, cache=True)
def _disk_moments_array(
    R, var_R0, var_phi0, var_z0, R_scale, R_smooth, R_sigma, R_solar, vc_solar, dvc_dR
):
    N = len(R)
    mean = np.zeros((N, 3))
    cov = np.zeros((N, 3, 3))
    for i in prange(N):
        R_eff = sqrt(R[i] * R[i] + R_smooth * R_smooth)
        moments = _disk_moments(
            R[i],
            R_eff,
            var_R0,
            var_phi0,
            var_z0,
            R_scale,
            R_smooth,
            R_sigma,
            R_solar,
            vc_solar,
            dvc_dR,
        )
        _store(mean, cov, i, *moments)
    return mean, cov


@jit(nopython=True, parallel=True, cache=True)
def _halo_moments_array(R, Z, var_r, var_theta, var_phi, v_phi):
    N = len(R)
    mean = np.zeros((N, 3))
    cov = np.zeros((N, 3, 3))
    for i in prange(N):
        moments = _halo_moments(R[i], Z[i], var_r, var_theta, var_phi, v_phi)
        _store(mean, cov, i, *moments)
    return mean, cov


def _check_sigma(name, sigma):
    if len(sigma) != 3 or min(sigma) < 0:
        raise ValueError(f"{name} must be 3 non-negative dispersions, got {sigma}")


def _check_radii(R):
    if not np.all(np.isfinite(R) & (R >= 0)):
        raise ValueError("R must be finite and non-negative")


def disk_velocity_moments(
    R,
    sigma_solar=THIN_DISK_SIGMA,
    R_scale=2.6,
    R_smooth=2.0,
    R_sigma=THIN_DISK_R_SIGMA,
    R_solar=8.2,
    v_circ_solar=None,
    dvcirc_dR=DVCIRC_DR,
):
    r"""
    Return the velocity ellipsoid of a disk component at radius ``R``.

    The ellipsoid is aligned with cylindrical coordinates, and each
    dispersion scales as

    .. math::
        \sigma_i^2(R) = \sigma_{i,\odot}^2 \exp\left(-\frac{R - R_\odot}{R_\sigma}\right)

    The mean rotation follows the radial Jeans equation (Binney & Tremaine
    2008, eq. 4.228) with the density gradient of
    `~brutus.priors.galactic.logn_disk`:

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

    The defaults describe the thin disk, whose solar lag
    :math:`v_c - \bar{v}_\phi` is then 14.8 km/s (Bensby et al. 2014:
    15 km/s). For the thick disk, pass ``sigma_solar=THICK_DISK_SIGMA``,
    ``R_scale=2.0`` and ``R_sigma=THICK_DISK_R_SIGMA``, which give the
    measured lag of 46 km/s; ``R_sigma=R_scale`` would give 86 km/s.

    Parameters
    ----------
    R : array_like of shape `(N,)`
        Galactocentric cylindrical radius in kpc.
    sigma_solar : tuple of float, optional
        Dispersions ``(sigma_R, sigma_phi, sigma_z)`` at the Sun in km/s.
        Default is ``THIN_DISK_SIGMA``, ``(35.0, 20.0, 16.0)`` (Bensby et
        al. 2014).
    R_scale : float, optional
        Radial scale length of the number density in kpc, as in
        `~brutus.priors.galactic.logn_disk`. Default is 2.6.
    R_smooth : float, optional
        Smoothing radius of the number density in kpc. Default is 2.0.
    R_sigma : float, optional
        Radial scale length of the velocity variances in kpc. Default is
        ``THIN_DISK_R_SIGMA``, 2.6.
    R_solar : float, optional
        Distance of the Sun from the Galactic centre in kpc. Default is 8.2.
    v_circ_solar : float, optional
        Circular velocity at ``R_solar`` in km/s. Default is the solar V
        from ``solar_velocity(R_solar)`` minus the peculiar 12.24 km/s of
        Schönrich et al. (2010): 236.97 km/s for ``R_solar=8.2``.
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
        If ``R`` is not 1-D, finite and non-negative, a dispersion is
        negative, or a length scale or ``R_solar`` is not positive.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 1:
        raise ValueError(f"R must be 1-D, got shape {R.shape}")
    _check_radii(R)
    _check_sigma("sigma_solar", sigma_solar)
    for name, value in (
        ("R_scale", R_scale),
        ("R_smooth", R_smooth),
        ("R_sigma", R_sigma),
        ("R_solar", R_solar),
    ):
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if v_circ_solar is None:
        v_circ_solar = solar_velocity(R_solar)[1] - V_PEC_SUN
    var_R, var_phi, var_z = (float(s) ** 2 for s in sigma_solar)
    return _disk_moments_array(
        R,
        var_R,
        var_phi,
        var_z,
        float(R_scale),
        float(R_smooth),
        float(R_sigma),
        float(R_solar),
        float(v_circ_solar),
        float(dvcirc_dR),
    )


def halo_velocity_moments(R, Z, sigma_sph=HALO_SIGMA, v_phi=HALO_V_PHI):
    r"""
    Return the velocity ellipsoid of the stellar halo at ``(R, Z)``.

    The ellipsoid is aligned with spherical coordinates, with dispersions
    :math:`(\sigma_r, \sigma_\theta, \sigma_\phi)` that do not vary with
    position. Rotated to cylindrical coordinates, with
    :math:`\sin\theta = R/r` and :math:`\cos\theta = Z/r`:

    .. math::
        \sigma_R^2 = \sigma_r^2 \sin^2\theta + \sigma_\theta^2 \cos^2\theta, \quad
        \sigma_z^2 = \sigma_r^2 \cos^2\theta + \sigma_\theta^2 \sin^2\theta, \quad
        \sigma_{Rz} = (\sigma_r^2 - \sigma_\theta^2) \sin\theta \cos\theta

    Parameters
    ----------
    R : array_like of shape `(N,)`
        Galactocentric cylindrical radius in kpc.
    Z : array_like of shape `(N,)`
        Height above the Galactic midplane in kpc.
    sigma_sph : tuple of float, optional
        Dispersions ``(sigma_r, sigma_theta, sigma_phi)`` in km/s. Default
        is ``HALO_SIGMA``, ``(141.0, 75.0, 85.0)`` (Bond et al. 2010).
    v_phi : float, optional
        Mean rotation in km/s. Default is ``HALO_V_PHI``, 0 (Bond et al.
        2010).

    Returns
    -------
    mean : `~numpy.ndarray` of shape `(N, 3)`
        Mean velocity ``(v_R, v_phi, v_z)`` in km/s.
    cov : `~numpy.ndarray` of shape `(N, 3, 3)`
        Velocity covariance in (km/s)^2.

    Raises
    ------
    ValueError
        If ``R`` and ``Z`` are not finite 1-D arrays of the same length,
        ``R`` is negative, or a dispersion is negative.
    """
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if R.ndim != 1 or R.shape != Z.shape:
        raise ValueError(
            f"R and Z must be 1-D with the same shape, got {R.shape} and {Z.shape}"
        )
    _check_radii(R)
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z must be finite")
    _check_sigma("sigma_sph", sigma_sph)
    var_r, var_theta, var_phi = (float(s) ** 2 for s in sigma_sph)
    return _halo_moments_array(R, Z, var_r, var_theta, var_phi, float(v_phi))


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
