#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Galactic structure priors for Bayesian stellar parameter estimation.

This module gives the number density of stars in a four-component model
of the Milky Way (thin disk, thick disk, halo and bar), as a function of
position and, optionally, of [Fe/H] and age. `logp_galactic_structure`
turns it into a prior on distance along a sightline.

See Also
--------
brutus.analysis.individual.BruteForce : Uses Galactic priors for distance fitting
brutus.priors.kinematic : The velocity ellipsoids of the same components

Examples
--------
Evaluate disk number density:

>>> import numpy as np
>>> from brutus.priors.galactic import logn_disk
>>>
>>> # Position in Galactic disk
>>> R = np.array([8.0])  # kpc from Galactic center
>>> Z = np.array([0.1])  # kpc above midplane
>>> log_density = logn_disk(R, Z)
>>> print(f"Log-density: {log_density[0]:.3f}")

Combined Galactic structure prior:

>>> from brutus.priors.galactic import logp_galactic_structure
>>>
>>> # Galactic coordinates (l, b) in degrees and distances in kpc
>>> coord = (90.0, 30.0)
>>> distances = np.array([1.0, 2.0, 5.0])
>>>
>>> # Evaluate the combined thin disk + thick disk + halo prior
>>> log_prior = logp_galactic_structure(distances, coord)
"""

import inspect
from functools import lru_cache
from math import asinh, cos, exp, isfinite, log, pi, sin, sqrt

import numpy as np
from numba import get_num_threads, jit, prange
from scipy.special import log_ndtr, logsumexp
from scipy.stats import skewnorm

from brutus.priors.kinematic import (
    _V_DPHI,
    _V_DU,
    _V_DX,
    _V_INV_FZ2,
    _V_JEANS,
    _V_VC0,
    _V_Z,
    _V_ZS,
    DISK_EDGE,
    THICK_DISK_SIGMA,
    THIN_DISK_Z_EDGE,
    THIN_DISK_Z_MAX,
    _catmull_rom_table,
    _check_positive,
    _circular_velocity,
    _logn_disk_dlnR,
    _soft_clamp,
    _thick_disk_ln_var,
    _thin_disk_fade,
    _thin_disk_fL,
    _thin_disk_fz,
    _thin_disk_L_refs,
    _thin_disk_mean_L,
    _thin_disk_sigma0,
    _vc_solar,
    _vertical_row,
    _vertical_terms,
)
from brutus.utils.math import (
    LOG_ZERO,
    R_SOLAR,
    Z_SOLAR,
    _log_ndtr_tail,
    _truncnorm_lognorm_scalar,
    galactic_to_galactocentric,
)

_LOG_2PI = log(2.0 * pi)
_LN2 = log(2.0)

# Stellar ages are capped at the age of the Universe, in Gyr.
MAX_AGE = 13.8

# Thin-disk star formation, enrichment and radial migration of Frankel et
# al. (2018), their posterior values, with radii in kpc and ages in Gyr.
# The star formation rate grows with age as exp(-(tau_m - tau) / tau_SFR).
# Stars are born at radius R0 with p(R0 | tau) = exp(-R0 / R_exp) / R_exp,
# R_exp = 3 (1 - alpha_R tau / 8), and [Fe/H] =
# F_m + grad R0 - (F_m + grad R_now) (1 - tau / tau_m)^gamma, then migrate
# by a Gaussian of width sigma_RM sqrt(tau / 8), normalized over R >= 0.
THIN_DISK_SFH = (12.0, 5.9)  # tau_m, tau_SFR
THIN_DISK_INSIDE_OUT = 0.42  # alpha_R
THIN_DISK_ENRICHMENT = (-1.0, -0.075, 8.8, 0.36)  # F_m, grad, R_now, gamma
THIN_DISK_MIGRATION = 3.6  # sigma_RM at 8 Gyr
# Frankel et al. fit stars younger than 8 Gyr and fix tau_m; older stars
# enter their fit only through a separate nuisance term, which brutus
# leaves out. Instead the star formation rate fades beyond 8 Gyr, the start
# of the low-alpha disk (Xiang & Rix 2022), as a half-Gaussian of width
# 1 Gyr: the age at which it starts to fade and the width, in Gyr.
THIN_DISK_ONSET = (8.0, 1.0)
# Scatter of [Fe/H] at birth, in dex, the open-cluster value that Frankel et
# al. cite. Migration supplies the spread of all but the youngest stars;
# this floor keeps the youngest from a single [Fe/H].
THIN_DISK_FEH_FLOOR = 0.03
# Radii of the Frankel et al. fit, in kpc. Outside them the label
# distribution keeps its edge value, through blends of half-width DISK_EDGE.
THIN_DISK_FIT_RADII = (5.0, 14.0)


def _gauss_legendre_panels(edges, n):
    """Return the nodes and weights of n-point Gauss-Legendre quadrature on
    each interval between ``edges``."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    edges = np.asarray(edges, dtype=float)
    lo, hi = edges[:-1, None], edges[1:, None]
    return (0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes).ravel(), (
        0.5 * (hi - lo) * weights
    ).ravel()


# ln Phi of the standard normal on steps of 1/128, from one step below -37
# to past 8: Catmull-Rom cubics in it are within 3e-9 of ln Phi, at less
# than half the cost of erfc. Outside -37 to 8, ln Phi is 0 above (to
# within 7e-16) and the asymptotic series of `_log_ndtr_tail` below,
# where erfc underflows.
_NDTR_DT = 1.0 / 128.0
_LN_NDTR = log_ndtr(-37.0 + _NDTR_DT * np.arange(-1, 5764))


@jit(nopython=True, cache=True)
def _ln_ndtr(t):
    """Return ln Phi(t) of the standard normal, from _LN_NDTR."""
    if t > 8.0:
        return 0.0
    if t > -37.0:
        return _catmull_rom_table((t + 37.0) / _NDTR_DT + 1.0, _LN_NDTR)
    return _log_ndtr_tail(t)


@jit(nopython=True, cache=True)
def _catmull_rom_axis(f, n):
    """Return the entries and weights of the Catmull-Rom cubic of
    `_catmull_rom` at the fractional index f >= 0 of an axis of n entries,
    mirrored about index 0 and held beyond index n - 2: entries
    (abs(i - 1), i, i + 1, i + 2)."""
    f = min(f, n - 2.0)
    i = min(int(f), n - 3)
    t = f - i
    t2 = t * t
    t3 = t2 * t
    return (
        (abs(i - 1), i, i + 1, i + 2),
        (
            0.5 * (2.0 * t2 - t - t3),
            1.0 + 0.5 * (3.0 * t3 - 5.0 * t2),
            0.5 * (t + 4.0 * t2 - 3.0 * t3),
            0.5 * (t3 - t2),
        ),
    )


@jit(nopython=True, cache=True)
def _catmull_rom_2d(fi, fk, table):
    """Return the bicubic Catmull-Rom interpolation of ``table`` at the
    fractional indices (fi, fk) >= 0, as `_catmull_rom_axis` in each axis."""
    rows, wi = _catmull_rom_axis(fi, table.shape[0])
    cols, wk = _catmull_rom_axis(fk, table.shape[1])
    out = 0.0
    for a in range(4):
        row = table[rows[a]]
        out += wi[a] * (
            wk[0] * row[cols[0]]
            + wk[1] * row[cols[1]]
            + wk[2] * row[cols[2]]
            + wk[3] * row[cols[3]]
        )
    return out


@jit(nopython=True, cache=True)
def _catmull_rom_3d(fl, fi, fk, table):
    """Return the tricubic Catmull-Rom interpolation of ``table`` at the
    fractional indices (fl, fi, fk) >= 0, as `_catmull_rom_axis` in each
    axis."""
    slabs, wl = _catmull_rom_axis(fl, table.shape[0])
    rows, wi = _catmull_rom_axis(fi, table.shape[1])
    cols, wk = _catmull_rom_axis(fk, table.shape[2])
    out = 0.0
    for c in range(4):
        for a in range(4):
            row = table[slabs[c], rows[a]]
            out += (
                wl[c]
                * wi[a]
                * (
                    wk[0] * row[cols[0]]
                    + wk[1] * row[cols[1]]
                    + wk[2] * row[cols[2]]
                    + wk[3] * row[cols[3]]
                )
            )
    return out


# Columns of the disk populations. A population with vertical dispersion
# sigma_z at the midplane has ln nu(z) = -2 ln f_z(|z|) - J(z) / s for the
# thin disk and -Phi_z(z) / s for the isothermal thick disk, with
# s = sigma_z^2 / phi, phi the scaling of the tabulated potential. The
# tables hold ln of the column int_0^100 kpc nu(z) dz, in kpc, at each row
# x = i dx of the potential tables and each ln s on _COLUMN_LNS. The heights
# are 8-point Gauss-Legendre panels, fine near the midplane (sigma_z of
# 1 km/s has a scale height near 15 pc), with edges at the ends of the
# thin-disk sigma_z blend; they stop at 100 kpc because the potential binds
# a hot population only logarithmically.
_COLUMN_EDGES = (
    [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.3, 0.6, 1.0]
    + [THIN_DISK_Z_MAX - THIN_DISK_Z_EDGE, THIN_DISK_Z_MAX + THIN_DISK_Z_EDGE]
    + [4.0, 7.0, 12.0, 20.0, 35.0, 60.0, 100.0]
)
_COLUMN_Z, _COLUMN_W = _gauss_legendre_panels(_COLUMN_EDGES, 8)
# First ln s, its step and the number of steps: sigma_z of 0.37 to 1100 km/s.
_COLUMN_LNS0, _COLUMN_DLNS, _COLUMN_NS = -2.0, 0.1, 161


def _column_tables():
    """Return the thin- and thick-disk column tables (see above)."""
    rows = np.arange(_V_DPHI.shape[0])[:, None]
    u = np.arcsinh(_COLUMN_Z / _V_ZS) / _V_DU
    j = np.minimum(u.astype(int), _V_DPHI.shape[1] - 2)
    dphi, jeans = getattr(_vertical_row, "py_func", _vertical_row)(
        rows, j[None, :], (u - j)[None, :]
    )
    ln_fz2 = 2.0 * np.log(np.vectorize(_thin_disk_fz)(_COLUMN_Z))
    inv_s = np.exp(-(_COLUMN_LNS0 + _COLUMN_DLNS * np.arange(_COLUMN_NS)))
    thin = np.empty((len(dphi), _COLUMN_NS))
    thick = np.empty_like(thin)
    for k, v in enumerate(inv_s):
        thin[:, k] = np.exp(-ln_fz2 - jeans * v) @ _COLUMN_W
        thick[:, k] = np.exp(-dphi * v) @ _COLUMN_W
    return np.log(thin), np.log(thick)


_THIN_COLUMN, _THICK_COLUMN = _column_tables()


@jit(nopython=True, cache=True)
def _ln_column(x, ln_s, table):
    """Return ln of the column of a population at x = R / R_solar with
    ln s = ln(sigma_z^2 / phi), from ``_THIN_COLUMN`` or ``_THICK_COLUMN``."""
    fk = max((ln_s - _COLUMN_LNS0) / _COLUMN_DLNS, 0.0)
    return _catmull_rom_2d(x / _V_DX, fk, table)


@jit(nopython=True, cache=True)
def _thin_disk_age_terms(age):
    """Return the terms of `_thin_disk_ln_joint` that depend only on age:
    ln of the star formation rate (LOG_ZERO outside 0 to tau_m), R_exp,
    the migration width s and the [Fe/H] at R0 = 0."""
    tau_m, tau_sfr = THIN_DISK_SFH
    f_m, grad, r_now, gamma = THIN_DISK_ENRICHMENT
    if not (0.0 <= age < tau_m):
        return LOG_ZERO, 1.0, 0.0, 0.0
    ln_sfr = -(tau_m - age) / tau_sfr
    onset, width = THIN_DISK_ONSET
    if age > onset:
        t = (age - onset) / width
        ln_sfr -= 0.5 * t * t
    rexp = 3.0 * (1.0 - THIN_DISK_INSIDE_OUT * age / 8.0)
    s = THIN_DISK_MIGRATION * sqrt(age / 8.0)
    feh0 = f_m - (f_m + grad * r_now) * (1.0 - age / tau_m) ** gamma
    return ln_sfr, rexp, s, feh0


@jit(nopython=True, cache=True)
def _thin_disk_ln_joint(feh, R, ln_sfr, rexp, s, feh0, sig_birth):
    """Return ln p([Fe/H], age, R) of the thin disk, per dex, Gyr and kpc^2
    up to a constant, from the age terms of `_thin_disk_age_terms` and the
    spread of the birth radius ``sig_birth`` (kpc) that the [Fe/H] scatter
    at birth implies.

    The integral over the birth radius R0 is analytic, except for the
    normalization of the migration kernel over R >= 0, 1 / Phi(R0 / s):
    it is taken at the mean R0 of the truncated Gaussian of R0. The error
    is largest for stars born near R0 = 0: 2.5% at most where the density
    exceeds 1e-3 of its peak at that age and R.
    """
    return _thin_disk_ln_at(
        R, _thin_disk_joint_terms(feh, ln_sfr, rexp, s, feh0, sig_birth)
    )


@jit(nopython=True, cache=True)
def _thin_disk_joint_terms(feh, ln_sfr, rexp, s, feh0, sig_birth):
    """Return the terms of `_thin_disk_ln_joint` that do not depend on R,
    for `_thin_disk_ln_at`."""
    if ln_sfr == LOG_ZERO:
        return LOG_ZERO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    grad = THIN_DISK_ENRICHMENT[1]
    # exp(-R0 / R_exp) N(R; R0, s) = exp(-R / R_exp + s^2 / 2 R_exp^2)
    # N(R0; mu, s) with mu = R - s^2 / R_exp, and the [Fe/H] of birth is
    # N(R0; r_birth, sig_birth).
    r_birth = (feh - feh0) / grad
    var, var_birth = s * s, sig_birth * sig_birth
    var2 = var + var_birth
    ln = ln_sfr - log(rexp * abs(grad)) + 0.5 * var / (rexp * rexp)
    ln -= 0.5 * (_LOG_2PI + log(var2))
    # r_birth - mu = d0 - R, and the mean of the product of the Gaussians in
    # R0 is m = w R + m0.
    d0 = r_birth + var / rexp
    m0 = var * (r_birth - var_birth / rexp) / var2
    sv = s * sig_birth / sqrt(var2)
    return ln, 1.0 / rexp, d0, 1.0 / var2, var_birth / var2, m0, sv, s


@jit(nopython=True, cache=True)
def _thin_disk_ln_at(R, terms):
    """Return `_thin_disk_ln_joint` at R from the terms of
    `_thin_disk_joint_terms` (a tuple or an array)."""
    ln = terms[0]
    if ln == LOG_ZERO:
        return LOG_ZERO
    d = terms[2] - R
    ln -= R * terms[1] + 0.5 * d * d * terms[3]
    s = terms[7]
    if s > 0.0:
        # The product of the two Gaussians in R0 is N(R0; m, sv), truncated
        # to R0 >= 0, weighted by the kernel normalization 1 / Phi(R0 / s).
        m = terms[4] * R + terms[5]
        sv = terms[6]
        a = m / sv
        ln_mass = _ln_ndtr(a)
        m_trunc = m + sv * exp(-0.5 * (a * a + _LOG_2PI) - ln_mass)
        ln += ln_mass - _ln_ndtr(m_trunc / s)
    return ln


@jit(nopython=True, cache=True)
def _two_panel_node(k, lo, cut, hi, nodes, weights):
    """Return node k and its ln weight of Gauss-Legendre ``nodes`` and
    ``weights`` on [-1, 1], mapped to the panels [lo, cut] and [cut, hi]."""
    n = len(nodes)
    a, b = (lo, cut) if k < n else (cut, hi)
    return 0.5 * (a + b) + 0.5 * (b - a) * nodes[k % n], log(
        0.5 * (b - a) * weights[k % n]
    )


# Ages of the sums over the thin-disk age distribution: 6-point
# Gauss-Legendre panels with edges where it changes form. Nodes of the sums
# over [Fe/H]: 12-point Gauss-Legendre for each of two panels.
_AGE_EDGES = [0.0, 0.5, 2.0, 5.0, THIN_DISK_ONSET[0], 10.0, THIN_DISK_SFH[0]]
_AGE_NODES, _AGE_WEIGHTS = _gauss_legendre_panels(_AGE_EDGES, 6)
_AGE_LN_W = np.log(_AGE_WEIGHTS)
_P_NODES, _P_WEIGHTS = np.polynomial.legendre.leggauss(12)
# Size of the arrays that `_thin_disk_feh_nodes` fills.
_FEH_NODES = 2 * len(_P_NODES)


@jit(nopython=True, cache=True)
def _thin_disk_feh_nodes(R, terms, sig_birth, feh, ln_w):
    """Fill ``feh`` and ``ln_w`` with the _FEH_NODES [Fe/H] nodes and ln
    weights of a sum over p([Fe/H], age, R) d[Fe/H], unnormalized in R,
    with ``terms`` from `_thin_disk_age_terms`."""
    ln_sfr, rexp, s, feh0 = terms
    grad = THIN_DISK_ENRICHMENT[1]
    # Birth radii implied by [Fe/H] over 6 standard deviations either side,
    # cut where births at R0 < 0 empty them and split at that edge.
    spread = sqrt(s * s + sig_birth * sig_birth)
    mu = R - s * s / rexp
    lo, hi = mu - 6.0 * spread, mu + 6.0 * spread
    cut = 0.5 * (lo + hi)
    if s > 0.0:
        edge = -mu * sig_birth * sig_birth / (s * s)
        width = 5.0 * sig_birth * spread / s
        lo = max(lo, edge - width)
        if lo < edge + width < hi:
            cut = edge + width
    for k in range(_FEH_NODES):
        r_birth, ln_dr = _two_panel_node(k, lo, cut, hi, _P_NODES, _P_WEIGHTS)
        feh[k] = feh0 + grad * r_birth
        ln_w[k] = _thin_disk_ln_joint(feh[k], R, *terms, sig_birth) + ln_dr
        ln_w[k] += log(abs(grad))


# Radii of the table of ln int int p([Fe/H], age, R) d[Fe/H] d(age), in
# kpc: the fitted radii with one node beyond each end.
_NORM_R0, _NORM_DR = THIN_DISK_FIT_RADII[0] - 0.25, 0.25


@lru_cache(maxsize=16)
def _thin_disk_norm_table(sig_birth):
    """Return ln of the thin-disk label normalization at _NORM_R0 + k
    _NORM_DR: the sum over the [Fe/H] nodes at 16-point Gauss-Legendre
    panels in age, so that the sums over the nodes of one label match it."""
    ages, w = _gauss_legendre_panels(_AGE_EDGES, 16)
    n = round((THIN_DISK_FIT_RADII[1] - THIN_DISK_FIT_RADII[0]) / _NORM_DR) + 3
    feh, ln_w = np.empty(_FEH_NODES), np.empty(_FEH_NODES)
    out = np.empty(n)
    for i in range(n):
        R = _NORM_R0 + _NORM_DR * i
        total = 0.0
        for a, wa in zip(ages, w):
            _thin_disk_feh_nodes(R, _thin_disk_age_terms(a), sig_birth, feh, ln_w)
            total += wa * np.exp(ln_w).sum()
        out[i] = log(total)
    out.flags.writeable = False
    return out


@jit(nopython=True, cache=True)
def _thin_disk_fit_radius(R, ln_norm):
    """Return R held to THIN_DISK_FIT_RADII, and ln of the label
    normalization there from a table of `_thin_disk_norm_table`."""
    R = _soft_clamp(R, THIN_DISK_FIT_RADII[0], THIN_DISK_FIT_RADII[1], DISK_EDGE)[0]
    return R, _catmull_rom_table((R - _NORM_R0) / _NORM_DR, ln_norm)


# Radial range of the halo mass of Lane et al. (2023), in kpc, and the
# Gauss-Legendre nodes per axis of the halo volume integral (relative error
# 3.4e-6 for the default shape, against an adaptive integral split at the
# break).
_HALO_MASS_RADII = (2.0, 70.0)
_HALO_NODES = 200


@jit(nopython=True, cache=True)
def _halo_shape(R, Z, Rs, eta_inner, eta_outer, r_break, q):
    """Return ln n_halo up to a constant, for scalars or arrays."""
    ln_m = 0.5 * np.log(R**2 + (Z / q) ** 2 + Rs**2)
    return -eta_inner * ln_m - (eta_outer - eta_inner) * np.maximum(
        ln_m - log(r_break), 0.0
    )


def _halo_args(R_smooth, eta_inner, eta_outer, r_break, q):
    """Return the halo-shape arguments of `_halo_shape` as floats, in order."""
    return tuple(map(float, (R_smooth, eta_inner, eta_outer, r_break, q)))


@lru_cache(maxsize=16)
def _halo_volume(shape):
    """Return the integral of exp(_halo_shape) over _HALO_MASS_RADII in kpc^3,
    by Gauss-Legendre quadrature in ln r and cos(theta)."""
    a, b = _HALO_MASS_RADII
    x, w = np.polynomial.legendre.leggauss(_HALO_NODES)
    mu, w_mu = 0.5 * (x + 1.0), 0.5 * w  # cos(theta) on [0, 1]
    half = 0.5 * log(b / a)
    r = np.exp(half * x + 0.5 * log(a * b))[:, None]
    f = np.exp(_halo_shape(r * np.sqrt(1.0 - mu**2), r * mu, *shape)) * r**3
    return float(4.0 * pi * half * (w @ (f @ w_mu)))


# Bar of Sormani et al. (2022), Table 1: their analytic fit to the stellar
# density of the made-to-measure model of Portail et al. (2017), with the
# full-precision values of AGAMA (py/example_mw_bar_potential.py). Lengths
# in kpc, densities in Msun/kpc^3; x is the long axis, y the intermediate.
# The X-shaped bar (their eqs. 2-5): rho, x, y, z scales, c_par, c_perp, m,
# outer cutoff, alpha, c, n, x_c, y_c.
_BAR_X = (
    3.16273226e09,
    0.490209137,
    0.392017253,
    0.229482096,
    1.99110223,
    2.23179266,
    0.873227940,
    4.36983774,
    0.625670015,
    1.34152138,
    1.94025114,
    0.750504078,
    0.468875471,
)
# The two long-bar terms (their eq. 6): rho, x, y scales, scale height,
# c_perp, n, outer and inner cutoff radii and their exponents.
_BAR_LONG = (
    (
        4.95381575e08,
        5.36363324,
        0.958522229,
        0.610542494,
        0.969645220,
        3.05125124,
        3.19043585,
        0.558255674,
        16.7310332,
        3.19575493,
    ),
    (
        1.74304936e13,
        0.477961423,
        0.266853061,
        0.251516920,
        1.87882599,
        0.980136710,
        2.20415408,
        7.60708626,
        -27.2907665,
        1.62966434,
    ),
)
# Integral of the density in Msun: 1.27656e10, 0.33319e10 and 0.21542e10
# for the three terms (Table 1: 1.28, 0.33, 0.22), by the midpoint rule on
# grids of 0.04 and 0.02 kpc with Richardson extrapolation.
BAR_MASS = 1.8252e10
# Angle in degrees between the long axis and the Sun-centre line, near end
# at l > 0 (AGAMA, after Portail et al. 2017).
BAR_ANGLE = 25.0
# Beyond this distance in kpc from the Galactic centre the bar is left out
# to save its evaluation; on that sphere it is at most 1.4e-4 of the disks.
_BAR_R_MAX = 12.0
# With one sightline, ln of the bar density is interpolated in a table of
# _BAR_NODES distances, uniform in ln d from 1 pc to 25 kpc; beyond 25 kpc
# a star is more than _BAR_R_MAX from the centre while R_solar < 13 kpc. The density has kinks
# (|y|^0.97 in a long-bar term), so the error falls slowly with the step:
# at most 1.2e-3 in ln n_bar where the bar exceeds 1e3 Msun/kpc^3, on 10
# sightlines.
_BAR_NODES = 4096
_BAR_LN_D0 = log(1e-3)
_BAR_DLN_D = (log(25.0) - _BAR_LN_D0) / (_BAR_NODES - 3)


@jit(nopython=True, cache=True)
def _bar_density(x, y, z):
    """Return the bar density in Msun/kpc^3 at (x, y, z) in the bar frame,
    for scalars."""
    ax, ay, az = abs(x), abs(y), abs(z)
    rho, x0, y0, z0, c_par, c_perp, m, r_cut, alpha, c, n, xc, yc = _BAR_X
    a = (
        ((ax / x0) ** c_perp + (ay / y0) ** c_perp) ** (c_par / c_perp)
        + (az / z0) ** c_par
    ) ** (1.0 / c_par)
    a_plus = sqrt(((x + c * z) / xc) ** 2 + (y / yc) ** 2)
    a_minus = sqrt(((x - c * z) / xc) ** 2 + (y / yc) ** 2)
    out = (
        rho
        / np.cosh(a**m)
        * exp(-(x * x + y * y + z * z) / (r_cut * r_cut))
        * (1.0 + alpha * (exp(-(a_plus**n)) + exp(-(a_minus**n))))
    )
    R = sqrt(x * x + y * y)
    if R == 0.0:
        return out  # both long-bar terms have an inner cutoff
    for rho, x0, y0, h, c_perp, n, r_out, r_in, n_out, n_in in _BAR_LONG:
        # The radial cutoffs in log space: either beyond 700 leaves nothing,
        # and its power would overflow without JIT.
        cut_out = n_out * log(R / r_out)
        cut_in = n_in * log(r_in / R)
        if cut_out > 700.0 or cut_in > 700.0:
            continue
        a = ((ax / x0) ** c_perp + (ay / y0) ** c_perp) ** (1.0 / c_perp)
        sech = 1.0 / np.cosh(z / h)
        out += rho * sech * sech * exp(-(a**n) - exp(cut_out) - exp(cut_in))
    return out


@jit(nopython=True, cache=True)
def _thin_disk_ln_s(ln_sig_R0, ln_sig_z0, x, ln_phi, f_L):
    """Return ln s = ln(sigma_z^2 / phi) at the midplane of a thin-disk
    population with ln sigma_R0 and ln sigma_z0 of ``_thin_disk_sigma0``,
    with f_L from `_thin_disk_fL_rows`."""
    fk = max((ln_sig_R0 - _FL_LNSIG0) / _FL_DLNSIG, 0.0)
    ln_fL = _catmull_rom_2d(x / _V_DX, fk, f_L)
    return 2.0 * (ln_sig_z0 + ln_fL) - ln_phi


@jit(nopython=True, cache=True)
def _thin_disk_ln_nu(ln_s, ln_col, ln_fz2, jeans):
    """Return ln nu(|z|) / (2 H) of a thin-disk population in 1/kpc: its
    density at height |z| over its surface density, for nu = 1 at the
    midplane and ln of its column H from ``_THIN_COLUMN``. ``ln_fz2`` and
    ``jeans`` come from ``_vertical_terms`` at (x, |z|); `_ln_height_sums`
    is the same for arrays."""
    return -ln_fz2 - jeans * exp(-ln_s) - _LN2 - ln_col


@jit(nopython=True, cache=True)
def _ln_skew_normal(v, xi, omega, alpha):
    """Return the ln pdf of a skew-normal with location ``xi``, scale
    ``omega`` and shape ``alpha`` at ``v``."""
    z = (v - xi) / omega
    return _LN2 - log(omega) - 0.5 * (z * z + _LOG_2PI) + _ln_ndtr(alpha * z)


@jit(nopython=True, cache=True)
def _ln_age_norm(mean, sd):
    """Return ln of the normalization of a normal in age of ``mean`` and
    ``sd``, truncated to 0 to MAX_AGE: its pdf is
    exp(-0.5 ((age - mean) / sd)^2 - _ln_age_norm(mean, sd))."""
    return (
        0.5 * _LOG_2PI
        + log(sd)
        + _truncnorm_lognorm_scalar(-mean / sd, (MAX_AGE - mean) / sd)
    )


@jit(nopython=True, cache=True)
def _thick_disk_mean_age(feh, age_at_floor, slope):
    """Return the mean age of thick-disk stars of ``feh``: linear in
    [Fe/H] above -1 and held below."""
    return age_at_floor + slope * (max(feh, -1.0) + 1.0)


@jit(nopython=True, cache=True)
def _age_spread(age, frac, floor, sd=0.0):
    """Return the systematic spread in Gyr of a labelled age about the true
    ``age``: ``frac`` of it and ``floor``, in quadrature, and with an
    intrinsic spread ``sd`` also in quadrature."""
    return sqrt(frac * frac * age * age + floor * floor + sd * sd)


@jit(nopython=True, cache=True)
def _ln_labels(feh, age, has_feh, has_age, labels, thick_ages, halo_feh, bar_pops):
    """Return ln p of the given labels for the thick disk, the halo and the
    bar, with ``labels``, ``thick_ages``, ``halo_feh`` and ``bar_pops`` from
    `_label_args`."""
    floor = labels[2]
    thick_xi, thick_omega, thick_alpha = labels[3:6]
    age_at_floor, slope, thick_sd, thick_frac = labels[6:10]
    halo_mean, halo_sd, halo_norm = labels[10:]
    if has_age and age > MAX_AGE:
        return LOG_ZERO, LOG_ZERO, LOG_ZERO
    # The bar: a mixture of populations, each normal in [Fe/H] and in age.
    bar = 0.0
    if has_feh or has_age:
        mx, acc = LOG_ZERO, 0.0
        for ln_w, feh_mean, feh_sd, age_mean, age_sd, age_norm in bar_pops:
            v = ln_w
            if has_feh:
                u = (feh - feh_mean) / feh_sd
                v -= 0.5 * (u * u + _LOG_2PI) + log(feh_sd)
            if has_age:
                u = (age - age_mean) / age_sd
                v -= 0.5 * u * u + age_norm
            mx, acc = _accumulate(mx, acc, v)
        bar = mx + log(acc)
    thick = 0.0
    halo = 0.0
    if has_feh:
        thick = _ln_skew_normal(feh, thick_xi, thick_omega, thick_alpha)
        fk = max((feh - _HALO_FEH0) / _HALO_DFEH, 0.0)
        halo = _catmull_rom_table(fk, halo_feh)
    if has_age:
        u = (age - halo_mean) / halo_sd
        halo -= 0.5 * u * u + halo_norm
        if has_feh:
            mean = _thick_disk_mean_age(feh, age_at_floor, slope)
            sd = _age_spread(mean, thick_frac, floor, thick_sd)
            u = (age - mean) / sd
            thick -= 0.5 * u * u + _ln_age_norm(mean, sd)
        else:
            mx, acc = LOG_ZERO, 0.0
            for k in range(len(thick_ages)):
                u = (age - thick_ages[k, 1]) / thick_ages[k, 2]
                mx, acc = _accumulate(mx, acc, thick_ages[k, 0] - 0.5 * u * u)
            thick = mx + log(acc)
    return thick, halo, bar


@jit(nopython=True, cache=True)
def _accumulate(mx, acc, v):
    """Add exp(v) to the running sum acc exp(mx), keeping mx the largest
    term so far; ln of the sum is mx + ln(acc)."""
    if v > mx:
        return v, acc * exp(mx - v) + 1.0
    return mx, acc + exp(v - mx)


@jit(nopython=True, cache=True)
def _ln_marginal(x, abs_z, table):
    """Return ln nu / (2 H) of the thin disk summed over its labels, from a
    table of `_thin_disk_tables`, at x = R / R_solar and height |z|."""
    return _catmull_rom_2d(
        x / (_V_DX * _MARGINAL_STEP), asinh(abs_z / _V_ZS) / _V_DU, table
    )


@jit(nopython=True, cache=True)
def _age_u(age, frac, floor):
    """Return u = asinh(frac age / floor) / frac, the age in units of its
    spread: d u / d age = 1 / `_age_spread`."""
    return asinh(frac * age / floor) / frac if frac > 0.0 else age / floor


@jit(nopython=True, cache=True)
def _label_index(feh, age, has_feh, labels, by_label):
    """Return the fractional index of the one given label in ``by_label``,
    `_thin_disk_feh_table` or `_thin_disk_age_table`, or -1 where the thin
    disk is summed directly (no table, or [Fe/H] outside it)."""
    if not len(by_label):
        return -1.0
    if has_feh:
        f = (feh - _FEH_TABLE[0]) / _FEH_TABLE[1]
        return f if 1.0 <= f <= len(by_label) - 2.0 else -1.0
    return _age_u(age, labels[1], labels[2]) / _AGE_TABLE_STEP + 1.0


# Gauss-Legendre nodes for the sums over the true age of a labelled age,
# in each of two panels, below and above the peak of the kernel.
_BLUR_NODES, _BLUR_WEIGHTS = np.polynomial.legendre.leggauss(8)


@jit(nopython=True, cache=True)
def _ln_age_kernel(age, true_age, frac, floor):
    """Return ln p(labelled ``age`` | ``true_age``): a normal of spread
    `_age_spread` truncated to 0 to MAX_AGE, so that every true age keeps
    its mass."""
    sd = _age_spread(true_age, frac, floor)
    u = (age - true_age) / sd
    return -0.5 * u * u - _ln_age_norm(true_age, sd)


@jit(nopython=True, cache=True)
def _true_age_nodes(age, frac, floor):
    """Return the true ages of the sum over the true age of a labelled
    ``age`` and their ln weights, with p(labelled age | true age): two
    Gauss-Legendre panels about the peak, over true ages within 6 spreads
    of their own and below the end of star formation; the labelled age
    itself without a spread; or none where no true age can reach it."""
    sd = _age_spread(age, frac, floor)
    if sd == 0.0:
        return np.full(1, age), np.zeros(1)
    # The spread grows with the true age t: the upper end solves
    # t - age = 6 sd(t), and none exists for frac >= 1/6. Below, 6 spreads
    # of the labelled age are more than 6 of the true age's.
    lo, peak, hi = age - 6.0 * sd, age, THIN_DISK_SFH[0]
    c = 36.0 * frac * frac
    if c < 1.0:
        hi = (age + sqrt(c * age * age + 36.0 * (1.0 - c) * floor * floor)) / (1.0 - c)
    onset, width = THIN_DISK_ONSET
    if age > onset:
        # Beyond the onset the fade of the star formation rate bounds the
        # true age: the product of the two Gaussians.
        prec = 1.0 / (sd * sd) + 1.0 / (width * width)
        peak = (age / (sd * sd) + onset / (width * width)) / prec
        hi = min(hi, peak + 6.0 / sqrt(prec))
    lo, hi = max(lo, 0.0), min(hi, THIN_DISK_SFH[0])
    if hi <= lo:
        return np.empty(0), np.empty(0)
    if not lo < peak < hi:
        peak = 0.5 * (lo + hi)
    n = 2 * len(_BLUR_NODES)
    ages, ln_w = np.empty(n), np.empty(n)
    for k in range(n):
        ages[k], ln_w[k] = _two_panel_node(k, lo, peak, hi, _BLUR_NODES, _BLUR_WEIGHTS)
        ln_w[k] += _ln_age_kernel(age, ages[k], frac, floor)
    return ages, ln_w


@jit(nopython=True, cache=True)
def _thin_disk_feh_ages(feh, ages, ln_w, sig_birth):
    """Return a row per true age of a sum over ``ages`` with ln weights
    ``ln_w`` at a given ``feh``: its ln weight, the terms of
    `_thin_disk_joint_terms`, and ln sigma_R0 and ln sigma_z0. Nothing in
    it depends on position."""
    nodes = np.empty((len(ages), 11))
    for k in range(len(ages)):
        terms = _thin_disk_joint_terms(feh, *_thin_disk_age_terms(ages[k]), sig_birth)
        sig_R0, sig_z0 = _thin_disk_sigma0(ages[k], feh)
        nodes[k, 0] = ln_w[k]
        for j in range(8):
            nodes[k, 1 + j] = terms[j]
        nodes[k, 9], nodes[k, 10] = log(sig_R0), log(sig_z0)
    return nodes


@jit(nopython=True, cache=True)
def _thin_disk_ln_feh_sum(nodes, R_fit, x, ln_phi, f_L, ln_fz2, jeans):
    """Return ln of the sum over the true ages of `_thin_disk_feh_ages` of
    p([Fe/H], true age | R) nu / (2 H), each with the vertical profile of
    its true age."""
    mx, acc = LOG_ZERO, 0.0
    for k in range(len(nodes)):
        ln_s = _thin_disk_ln_s(nodes[k, 9], nodes[k, 10], x, ln_phi, f_L)
        v = nodes[k, 0] + _thin_disk_ln_at(R_fit, nodes[k, 1:9])
        v += _thin_disk_ln_nu(ln_s, _ln_column(x, ln_s, _THIN_COLUMN), ln_fz2, jeans)
        mx, acc = _accumulate(mx, acc, v)
    return mx + log(acc) if acc > 0.0 else LOG_ZERO


@jit(nopython=True, cache=True)
def _thin_disk_feh_populations(nodes, x, R_fit, ln_phi, f_L):
    """Return ln of the weights, ln s and ln of the columns of the
    populations of the ``nodes`` of `_thin_disk_feh_ages` at R_fit, for
    `_ln_height_sums`."""
    n = len(nodes)
    ln_w, ln_s, ln_col = np.empty(n), np.empty(n), np.empty(n)
    for k in range(n):
        ln_w[k] = nodes[k, 0] + _thin_disk_ln_at(R_fit, nodes[k, 1:9])
        ln_s[k] = _thin_disk_ln_s(nodes[k, 9], nodes[k, 10], x, ln_phi, f_L)
        ln_col[k] = _ln_column(x, ln_s[k], _THIN_COLUMN)
    return ln_w, ln_s, ln_col


@jit(nopython=True, cache=True)
def _thin_disk_age_populations(age, x, R_fit, ln_phi, f_L, sig_birth):
    """Return ln of the label weights, unnormalized in R, ln s and ln of
    the columns of the thin-disk populations of true ``age`` at R_fit: the
    [Fe/H] nodes of `_thin_disk_feh_nodes`."""
    fehs, ln_w = np.empty(_FEH_NODES), np.empty(_FEH_NODES)
    ln_s, ln_col = np.empty(_FEH_NODES), np.empty(_FEH_NODES)
    _thin_disk_feh_nodes(R_fit, _thin_disk_age_terms(age), sig_birth, fehs, ln_w)
    for j in range(_FEH_NODES):
        sig_R0, sig_z0 = _thin_disk_sigma0(age, fehs[j])
        ln_s[j] = _thin_disk_ln_s(log(sig_R0), log(sig_z0), x, ln_phi, f_L)
        ln_col[j] = _ln_column(x, ln_s[j], _THIN_COLUMN)
    return ln_w, ln_s, ln_col


@jit(nopython=True, cache=True)
def _ln_sum_nu(ln_w, ln_s, ln_col, ln_fz2, jeans):
    """Return ln sum_p w_p nu_p(|z|) / (2 H_p) of thin-disk populations,
    as `_thin_disk_ln_nu`."""
    mx, acc = LOG_ZERO, 0.0
    for p in range(len(ln_w)):
        mx, acc = _accumulate(
            mx, acc, ln_w[p] + _thin_disk_ln_nu(ln_s[p], ln_col[p], ln_fz2, jeans)
        )
    return mx + log(acc) if acc > 0.0 else LOG_ZERO


@jit(nopython=True, cache=True)
def _ln_surface(R, R_solar, R_smooth, R_scale):
    """Return ln of an exponential surface density in the smoothed radius,
    relative to the Sun, as in `logn_disk`."""
    R_eff = sqrt(R * R + R_smooth * R_smooth)
    return -(R_eff - sqrt(R_solar * R_solar + R_smooth * R_smooth)) / R_scale


@jit(nopython=True, cache=True)
def _components(
    R,
    Z,
    feh,
    age,
    has_feh,
    has_age,
    disk,
    halo,
    labels,
    label_terms,
    f_L,
    marginal,
    ln_norm,
    by_label,
):
    """Return ln n of the thin disk, thick disk and halo at (R, Z), relative
    to the thin disk at the Sun, with the given labels, or summed over
    those not given. ``disk``, ``halo`` and the tables ``f_L``,
    ``marginal`` and ``ln_norm`` come from `_normalizations`, ``by_label``
    from `_label_table`, ``labels`` from `_label_args` and ``label_terms``
    from `_label_terms`."""
    if not (isfinite(R) and isfinite(Z)):
        return np.nan, np.nan, np.nan
    R_solar, ln_phi, R_thin, Rs_thin, R_thick, Rs_thick = disk[:6]
    ln_f_thick, var_thick, ln_d0 = disk[6:]
    x = R / R_solar
    abs_z = abs(Z)
    ln_fz2, jeans, dphi = _vertical_terms(x, abs_z)

    # Thin disk: surface density, times the sum over its populations of
    # p(labels | R) nu(z) / (2 H).
    thin = _ln_surface(R, R_solar, Rs_thin, R_thin) - ln_d0
    ln_thick, ln_halo, fl, nodes, ages, ln_w = label_terms
    if has_age and age > MAX_AGE:
        # The labelled ages of every component lie in 0 to MAX_AGE.
        thin = LOG_ZERO
    elif fl >= 0.0:
        fi = x / (_V_DX * _LABEL_STEPS[0])
        fk = asinh(abs_z / _V_ZS) / (_V_DU * _LABEL_STEPS[1])
        thin += _catmull_rom_3d(fl, fi, fk, by_label)
    elif has_feh or has_age:
        R_fit, ln_norm_R = _thin_disk_fit_radius(R, ln_norm)
        if has_feh:
            thin += _thin_disk_ln_feh_sum(nodes, R_fit, x, ln_phi, f_L, ln_fz2, jeans)
        else:
            populations = _thin_disk_age_sum(
                ages, ln_w, x, R_fit, ln_phi, f_L, labels[0]
            )
            thin += _ln_sum_nu(*populations, ln_fz2, jeans)
        thin -= ln_norm_R
    else:
        thin += _ln_marginal(x, abs_z, marginal)

    # Thick disk: one isothermal population.
    s = var_thick * exp(_thick_disk_ln_var(x, 2)[0] - ln_phi)
    thick = (
        ln_f_thick
        + _ln_surface(R, R_solar, Rs_thick, R_thick)
        - dphi / s
        - _LN2
        - _ln_column(x, log(s), _THICK_COLUMN)
        - ln_d0
    )
    hal = _halo_shape(R, Z, halo[0], halo[1], halo[2], halo[3], halo[4]) + halo[5]
    return thin, thick + ln_thick, hal + ln_halo


@jit(nopython=True, cache=True)
def _ln_bar_density(X, Y, Z, cos_a, sin_a):
    """Return ln of the bar density in Msun/kpc^3 at the Galactocentric
    (X, Y, Z), with the bar at the angle of (cos_a, sin_a); ln 1e-300
    where it underflows."""
    rho = _bar_density(X * cos_a - Y * sin_a, X * sin_a + Y * cos_a, Z)
    return log(max(rho, 1e-300))


@jit(nopython=True, parallel=True, cache=True)
def _ln_bar_densities(X, Y, Z, cos_a, sin_a):
    """Return `_ln_bar_density` at each of the positions (X, Y, Z)."""
    out = np.empty(len(X))
    for i in prange(len(X)):
        out[i] = _ln_bar_density(X[i], Y[i], Z[i], cos_a, sin_a)
    return out


@jit(nopython=True, cache=True)
def _label_terms(
    feh,
    age,
    has_feh,
    has_age,
    labels,
    thick_ages,
    halo_feh,
    bar_pops,
    by_label,
    no_nodes,
    no_ages,
):
    """Return the terms of `_components` that depend on the labels alone:
    the thick-disk and halo ln p(labels), the index of `_label_index` in
    ``by_label``, and where the thin-disk sum is direct, its true ages and
    ln weights and, with [Fe/H], their nodes of `_thin_disk_feh_ages`; and
    beside them the bar ln p(labels). ``no_nodes`` and ``no_ages`` stand in
    for those not needed."""
    ln_thick, ln_halo, ln_bar = _ln_labels(
        feh, age, has_feh, has_age, labels, thick_ages, halo_feh, bar_pops
    )
    fl = -1.0
    if has_feh != has_age:
        fl = _label_index(feh, age, has_feh, labels, by_label)
    ages, ln_w, nodes = no_ages, no_ages, no_nodes
    if fl < 0.0 and (has_feh or has_age) and not (has_age and age > MAX_AGE):
        if has_age:
            ages, ln_w = _true_age_nodes(age, labels[1], labels[2])
        else:
            # A systematic in age leaves the sum over ages unchanged.
            ages, ln_w = _AGE_NODES.copy(), _AGE_LN_W.copy()
        if has_feh:
            nodes = _thin_disk_feh_ages(feh, ages, ln_w, labels[0])
    return (ln_thick, ln_halo, fl, nodes, ages, ln_w), ln_bar


@jit(nopython=True, parallel=True, cache=True)
def _galactic_prior_array(
    dists,
    X,
    Y,
    Z,
    feh,
    loga,
    n_labels,
    has_feh,
    has_age,
    disk,
    halo,
    bar,
    bar_line,
    labels,
    thick_ages,
    halo_feh,
    bar_pops,
    f_L,
    marginal,
    ln_norm,
    by_label,
    comps,
):
    """Return ln of the summed number density, with the volume factor d^2,
    at each point of Galactocentric (X, Y, Z); fill ``comps`` with the
    components unless it is empty. Point i has the labels of entry
    i % n_labels of ``feh`` and ``loga``, whose terms are computed once.
    ``bar`` comes from `_bar_args`; unless ``bar_line`` is empty, the bar
    density is interpolated in it, `_ln_bar_density` at ln d =
    _BAR_LN_D0 + k _BAR_DLN_D along the sightline."""
    cos_a, sin_a, ln_bar_amp = bar
    out = np.empty(len(Z))
    no_nodes, no_ages = np.empty((0, 11)), np.empty(0)
    for c in prange(n_labels):
        feh_c = feh[c] if has_feh else 0.0
        age_c = 10.0 ** loga[c] / 1e9 if has_age else 0.0
        label_terms, ln_bar = _label_terms(
            feh_c,
            age_c,
            has_feh,
            has_age,
            labels,
            thick_ages,
            halo_feh,
            bar_pops,
            by_label,
            no_nodes,
            no_ages,
        )
        for i in range(c, len(Z), n_labels):
            R = sqrt(X[i] * X[i] + Y[i] * Y[i])
            thin, thick, hal = _components(
                R,
                Z[i],
                feh_c,
                age_c,
                has_feh,
                has_age,
                disk,
                halo,
                labels,
                label_terms,
                f_L,
                marginal,
                ln_norm,
                by_label,
            )
            ln_d = log(dists[i] + 1e-300)
            # Bar: by mass like the halo, left out far from the centre.
            bar_n = LOG_ZERO
            if ln_bar_amp > LOG_ZERO and R * R + Z[i] * Z[i] < _BAR_R_MAX**2:
                if len(bar_line):
                    f = (ln_d - _BAR_LN_D0) / _BAR_DLN_D
                    if not f > 0.0:  # below 1 pc, or NaN for a bad distance
                        f = 0.0
                    ln_rho = _catmull_rom_table(f, bar_line)
                else:
                    ln_rho = _ln_bar_density(X[i], Y[i], Z[i], cos_a, sin_a)
                bar_n = ln_rho + ln_bar_amp + ln_bar
            mx = max(max(thin, thick), max(hal, bar_n))
            acc = exp(thin - mx) + exp(thick - mx) + exp(hal - mx)
            if bar_n > LOG_ZERO:
                acc += exp(bar_n - mx)
            vol = 2.0 * ln_d
            out[i] = mx + log(acc) + vol
            if len(comps):
                comps[0, i] = thin + vol
                comps[1, i] = thick + vol
                comps[2, i] = hal + vol
                comps[3, i] = bar_n + vol
    return out


# Rows of the potential tables per row of the summed thin-disk table, a
# step of 0.05 in x.
_MARGINAL_STEP = 4


# ln sigma_R0 (km/s) of the f_L table: 9 to 68 km/s, which covers the
# thin-disk ages and [Fe/H].
_FL_LNSIG0, _FL_DLNSIG, _FL_NSIG = 2.2, 0.025, 81


@jit(nopython=True, cache=True)
def _thin_disk_fL_rows(R_solar, vc_solar, R_thin, Rs_thin):
    """Return ln f_L of the thin-disk sigma_z at the mean L_z of a
    population, at the rows x = i dx of the potential tables and ln
    sigma_R0 on the _FL grid. At the midplane, where brutus takes sigma_z,
    f_L depends on the population only through sigma_R0."""
    L_solar, L_clamp = _thin_disk_L_refs(R_solar, vc_solar)
    rows = _V_DPHI.shape[0]
    table = np.empty((rows, _FL_NSIG))
    for r in range(rows):
        R = r * _V_DX * R_solar
        vc, dlnvc = _circular_velocity(R, R_solar, vc_solar)
        dlnnu = _logn_disk_dlnR(R, sqrt(R * R + Rs_thin * Rs_thin), R_thin)
        fade = _thin_disk_fade(R)
        for k in range(_FL_NSIG):
            sig_R0 = exp(_FL_LNSIG0 + k * _FL_DLNSIG)
            L = _thin_disk_mean_L(
                R, 0.0, sig_R0, vc, dlnvc, L_solar, L_clamp, dlnnu, fade
            )
            table[r, k] = log(_thin_disk_fL(L, L_solar, L_clamp)[1])
    return table


@jit(nopython=True, cache=True)
def _thin_disk_age_sum(ages, ln_w_ages, x, R_fit, ln_phi, f_L, sig_birth):
    """Return ln of the weights, unnormalized in R, ln s and ln of the
    columns of the thin-disk populations of a sum over ``ages`` with ln
    weights ``ln_w_ages`` and over the [Fe/H] nodes of each, at R_fit."""
    n = len(ages) * _FEH_NODES
    ln_w, ln_s, ln_col = np.empty(n), np.empty(n), np.empty(n)
    for k in range(len(ages)):
        m = slice(k * _FEH_NODES, (k + 1) * _FEH_NODES)
        w, ln_s[m], ln_col[m] = _thin_disk_age_populations(
            ages[k], x, R_fit, ln_phi, f_L, sig_birth
        )
        ln_w[m] = w + ln_w_ages[k]
    return ln_w, ln_s, ln_col


def _ln_height_sums(row, ln_w, ln_s, ln_col, step=1):
    """Return ln sum_p w_p nu_p(|z|) / (2 H_p) of thin-disk populations
    along the last axis, at row ``row`` of the potential tables and every
    ``step``-th of their heights, as `_thin_disk_ln_nu`."""
    jeans = _V_JEANS[row, ::step]
    terms = (ln_w - ln_col - _LN2)[..., None, :] - jeans[:, None] * np.exp(-ln_s)[
        ..., None, :
    ]
    # Written out, a third of the time of scipy's logsumexp here.
    top = terms.max(axis=-1)
    ln_sum = np.log(np.exp(terms - top[..., None]).sum(axis=-1)) + top
    return ln_sum + np.log(_V_INV_FZ2[::step])


# Rows of the potential tables at the rows of the thin-disk tables.
_TABLE_ROWS = range(0, _V_DPHI.shape[0], _MARGINAL_STEP)


@lru_cache(maxsize=16)
def _thin_disk_tables(R_solar, vc_solar, R_thin, Rs_thin, sig_birth):
    """Return the f_L table of `_thin_disk_fL_rows`; ln nu(|z|) / (2 H) of
    the thin disk summed over its age and [Fe/H] distribution, in 1/kpc,
    at x = i _MARGINAL_STEP dx and the heights of the potential tables;
    and the table of `_thin_disk_norm_table`. Costs 0.1 s, or 2 s with
    NUMBA_DISABLE_JIT."""
    ln_phi = 2.0 * log(vc_solar / _V_VC0)
    f_L = _thin_disk_fL_rows(R_solar, vc_solar, R_thin, Rs_thin)
    ln_norm = _thin_disk_norm_table(sig_birth)
    marginal = np.empty((len(_TABLE_ROWS), len(_V_Z)))
    for r, i in enumerate(_TABLE_ROWS):
        x = i * _V_DX
        R_fit, ln_norm_R = _thin_disk_fit_radius(x * R_solar, ln_norm)
        populations = _thin_disk_age_sum(
            _AGE_NODES, _AGE_LN_W, x, R_fit, ln_phi, f_L, sig_birth
        )
        marginal[r] = _ln_height_sums(i, *populations) - ln_norm_R
    for table in (f_L, marginal):
        table.flags.writeable = False
    return f_L, marginal, ln_norm


# [Fe/H] of the thin-disk table by [Fe/H]: first value, step and number.
# Outside it the sum over ages is direct. The tables by one label take
# every _LABEL_STEPS[0]-th row and _LABEL_STEPS[1]-th height of the
# potential tables.
_FEH_TABLE = (-3.5, 0.025, 181)
_LABEL_STEPS = (2, 4)
_LABEL_ROWS = range(0, _V_DPHI.shape[0], _LABEL_STEPS[0])


@lru_cache(maxsize=4)
def _thin_disk_feh_table(R_solar, vc_solar, R_thin, Rs_thin, sig_birth):
    """Return ln of the thin-disk sum over ages of p([Fe/H], age | R)
    nu(|z|) / (2 H), against the [Fe/H] of _FEH_TABLE and the rows and
    heights of the tables of `_thin_disk_tables`."""
    f_L, _, ln_norm = _thin_disk_tables(R_solar, vc_solar, R_thin, Rs_thin, sig_birth)
    ln_phi = 2.0 * log(vc_solar / _V_VC0)
    fehs = _FEH_TABLE[0] + _FEH_TABLE[1] * np.arange(_FEH_TABLE[2])
    nodes = [_thin_disk_feh_ages(f, _AGE_NODES, _AGE_LN_W, sig_birth) for f in fehs]
    table = np.empty((len(fehs), len(_LABEL_ROWS), len(_V_Z[:: _LABEL_STEPS[1]])))
    for r, i in enumerate(_LABEL_ROWS):
        x = i * _V_DX
        R_fit, ln_norm_R = _thin_disk_fit_radius(x * R_solar, ln_norm)
        populations = np.array(
            [_thin_disk_feh_populations(n, x, R_fit, ln_phi, f_L) for n in nodes]
        )
        table[:, r] = (
            _ln_height_sums(i, *populations.transpose(1, 0, 2), _LABEL_STEPS[1])
            - ln_norm_R
        )
    table.flags.writeable = False
    return table


# Labelled ages of the thin-disk table by age: steps of _AGE_TABLE_STEP in
# `_age_u` from one step below 0 to past MAX_AGE, where the kernel
# continues smoothly. With more than _AGE_TABLE_MAX steps (a
# spread below about 0.03 Gyr) the sum over true ages is direct. The true
# ages of the table are 6-point Gauss-Legendre panels 2 wide in u.
_AGE_TABLE_STEP, _AGE_TABLE_MAX = 0.25, 400
_U_POINTS = 6


@lru_cache(maxsize=4)
def _thin_disk_age_table(R_solar, vc_solar, R_thin, Rs_thin, sig_birth, frac, floor):
    """Return ln of the thin-disk sum over true ages and [Fe/H] of
    p(true age, [Fe/H] | R) p(labelled age | true age) nu(|z|) / (2 H),
    against the labelled ages of _AGE_TABLE_STEP and the rows and heights
    of the tables of `_thin_disk_tables`; or an empty table where the
    spread is too small for it."""
    if floor <= 0.0:
        return np.empty((0, 0, 0))
    u_top = _age_u(MAX_AGE, frac, floor)
    if u_top > _AGE_TABLE_MAX * _AGE_TABLE_STEP:
        return np.empty((0, 0, 0))
    f_L, _, ln_norm = _thin_disk_tables(R_solar, vc_solar, R_thin, Rs_thin, sig_birth)
    ln_phi = 2.0 * log(vc_solar / _V_VC0)

    def age_of(u):
        return floor * np.sinh(frac * u) / frac if frac > 0.0 else floor * u

    n = int(np.ceil(u_top / _AGE_TABLE_STEP)) + 4
    labelled = age_of(_AGE_TABLE_STEP * np.arange(-1, n - 1))
    u_max = _age_u(THIN_DISK_SFH[0], frac, floor)
    edges = np.linspace(0.0, u_max, int(np.ceil(u_max / 2.0)) + 1)
    u, w = _gauss_legendre_panels(edges, _U_POINTS)
    true = age_of(u)
    ln_dt = np.log(w * [_age_spread(t, frac, floor) for t in true])
    ln_k = ln_dt + np.array(
        [[_ln_age_kernel(a, t, frac, floor) for t in true] for a in labelled]
    )
    k_top = ln_k.max(axis=1, keepdims=True)
    kernel = np.exp(ln_k - k_top)
    table = np.empty((n, len(_LABEL_ROWS), len(_V_Z[:: _LABEL_STEPS[1]])))
    for r, i in enumerate(_LABEL_ROWS):
        x = i * _V_DX
        R_fit, ln_norm_R = _thin_disk_fit_radius(x * R_solar, ln_norm)
        populations = np.array(
            [
                _thin_disk_age_populations(t, x, R_fit, ln_phi, f_L, sig_birth)
                for t in true
            ]
        )
        by_true = _ln_height_sums(i, *populations.transpose(1, 0, 2), _LABEL_STEPS[1])
        # The sum over true ages as a product of matrices scaled to their
        # largest entries. Sums below 1e-200 of that scale, where a term
        # underflows, are redone in logs.
        z_top = by_true.max(axis=0)
        sums = kernel @ np.exp(by_true - z_top)
        small = sums < 1e-200
        sums[small] = 1.0
        ln_sum = np.log(sums) + k_top + z_top
        a, z = np.nonzero(small)
        ln_sum[a, z] = logsumexp(ln_k[a] + by_true[:, z].T, axis=1)
        table[:, r] = ln_sum - ln_norm_R
    table.flags.writeable = False
    return table


def _skew_normal_args(name, value, blur):
    """Return (xi, omega, alpha, mean, sd) of the skew-normal with
    ``value`` = (mean, standard deviation, skewness), convolved with a
    normal of standard deviation ``blur``, which keeps it skew-normal."""
    mean, sd, skew = _floats(name, value[:3], 3)
    if not (sd > 0 and abs(skew) < 0.99):
        raise ValueError(
            f"{name} must start with (mean, standard deviation, skewness), "
            f"with a positive standard deviation and |skewness| < 0.99; "
            f"got {value}"
        )
    # Moments of the skew-normal: delta = alpha / sqrt(1 + alpha^2) from
    # the skewness, then the scale and location.
    k = abs(skew) ** (2.0 / 3.0)
    delta2 = 0.5 * pi * k / (k + (0.5 * (4.0 - pi)) ** (2.0 / 3.0))
    delta = np.copysign(sqrt(delta2), skew)
    omega = sd / sqrt(1.0 - 2.0 * delta2 / pi)
    xi = mean - omega * delta * sqrt(2.0 / pi)
    alpha = delta / sqrt(1.0 - delta2)
    # Adding an independent normal keeps the location and the mean.
    blurred = sqrt(omega**2 + blur**2)
    alpha *= omega / sqrt(omega**2 + blur**2 * (1.0 + alpha**2))
    return xi, blurred, alpha, mean, sqrt(sd**2 + blur**2)


def _floats(name, value, n):
    """Return ``value`` as a tuple of ``n`` finite floats."""
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        out = ()
    if len(out) != n or not all(np.isfinite(out)):
        raise ValueError(f"{name} must be {n} finite numbers, got {value!r}")
    return out


# Grid of the halo [Fe/H] table: first value and step, and the grid.
_HALO_FEH0, _HALO_DFEH = -6.0, 0.005
_HALO_FEHS = _HALO_FEH0 + _HALO_DFEH * np.arange(1601)


@lru_cache(maxsize=16)
def _halo_feh_table(xi, omega, alpha, tail_break, tail_slope, blur):
    """Return ln p([Fe/H]) of the halo on the _HALO_FEH grid: the
    skew-normal core above ``tail_break``, below it 10^(tail_slope
    ([Fe/H] - tail_break)) times the core's value there, normalized in
    closed form, then convolved with a normal of standard deviation
    ``blur`` by quadrature on the grid."""
    x = _HALO_FEHS
    k = tail_slope * log(10.0)
    core = skewnorm(alpha, xi, omega)
    core_break = core.logpdf(tail_break)
    ln_p = np.where(x >= tail_break, core.logpdf(x), core_break + k * (x - tail_break))
    # The core's mass above the break plus the tail's.
    ln_p -= log(core.sf(tail_break) + exp(core_break) / k)
    if blur > 0.0:
        half = int(np.ceil(8.0 * blur / _HALO_DFEH))
        u = np.arange(-half, half + 1) * _HALO_DFEH / blur
        kernel = np.exp(-0.5 * u * u)
        kernel /= kernel.sum()
        top = ln_p.max()
        padded = np.concatenate(
            (
                np.exp(ln_p[0] - top + k * _HALO_DFEH * np.arange(-half, 0)),
                np.exp(ln_p - top),
                np.zeros(half),
            )
        )
        ln_p = np.log(np.maximum(np.convolve(padded, kernel, "valid"), 1e-300)) + top
    ln_p.flags.writeable = False
    return ln_p


def _feh_systematics(feh_systematic):
    """Return ``feh_systematic`` as 3 floats, raising ValueError unless they
    are finite and not negative."""
    values = _floats("feh_systematic", feh_systematic, 3)
    if min(values) < 0:
        raise ValueError(f"feh_systematic must not be negative, got {values}")
    return values


def _thin_disk_sig_birth(feh_systematic):
    """Return the spread in kpc of the thin-disk birth radius that the
    [Fe/H] scatter at birth and the thin-disk [Fe/H] systematic imply."""
    thin = _feh_systematics(feh_systematic)[0]
    return sqrt(THIN_DISK_FEH_FLOOR**2 + thin**2) / abs(THIN_DISK_ENRICHMENT[1])


def _label_args(
    thick_feh,
    thick_age,
    halo_feh,
    halo_age,
    bar_feh,
    bar_age,
    feh_systematic,
    age_systematic,
    age_floor,
):
    """Return the label arguments of `_components`: a flat tuple; the
    thick-disk age density summed over [Fe/H], as rows of (ln weight less
    the normalizations, mean age, standard deviation); the halo [Fe/H]
    table of `_halo_feh_table`; and the bar populations of
    `_bar_populations`."""
    _, thick_feh_sys, halo_feh_sys = _feh_systematics(feh_systematic)
    thick_feh = _floats("thick_feh", thick_feh, 3)
    halo_feh = _floats("halo_feh", halo_feh, 5)
    thin_frac, thick_frac, halo_frac = _floats("age_systematic", age_systematic, 3)
    (floor,) = _floats("age_floor", (age_floor,), 1)
    if min(thin_frac, thick_frac, halo_frac, floor) < 0:
        raise ValueError(
            f"age_systematic and age_floor must not be negative, got "
            f"{age_systematic} and {age_floor}"
        )
    age_at_floor, slope, thick_sd = _floats("thick_age", thick_age, 3)
    halo_mean, halo_sd = _floats("halo_age", halo_age, 2)
    _check_positive(**{"thick_age[2]": thick_sd, "halo_age[1]": halo_sd})
    xi, omega, alpha, mean, sd = _skew_normal_args(
        "thick_feh", thick_feh, thick_feh_sys
    )
    # The thick-disk age density summed over [Fe/H]: 32-point
    # Gauss-Legendre in each of two panels over 6 standard deviations either
    # side, split at [Fe/H] = -1, where the mean age is held.
    lo, hi = mean - 6.0 * sd, mean + 6.0 * sd
    cut = -1.0 if lo < -1.0 < hi else 0.5 * (lo + hi)
    fehs, w = _gauss_legendre_panels((lo, cut, hi), 32)
    ln_w = np.log(w) + [_ln_skew_normal(f, xi, omega, alpha) for f in fehs]
    ages = np.array([_thick_disk_mean_age(f, age_at_floor, slope) for f in fehs])
    sds = np.array([_age_spread(a, thick_frac, floor, thick_sd) for a in ages])
    ln_w -= logsumexp(ln_w)
    ln_w -= [_ln_age_norm(a, d) for a, d in zip(ages, sds)]
    halo_xi, halo_omega, halo_alpha, *_ = _skew_normal_args("halo_feh", halo_feh, 0.0)
    tail_break, tail_slope = _floats("halo_feh[3:]", halo_feh[3:], 2)
    _check_positive(**{"halo_feh[4]": tail_slope})
    halo_table = _halo_feh_table(
        halo_xi, halo_omega, halo_alpha, tail_break, tail_slope, halo_feh_sys
    )
    halo_sd = _age_spread(halo_mean, halo_frac, floor, halo_sd)
    labels = (
        _thin_disk_sig_birth(feh_systematic),
        thin_frac,
        floor,
        xi,
        omega,
        alpha,
        age_at_floor,
        slope,
        thick_sd,
        thick_frac,
        halo_mean,
        halo_sd,
        _ln_age_norm(halo_mean, halo_sd),
    )
    bar_pops = _bar_populations(bar_feh, bar_age, thick_feh_sys, thick_frac, floor)
    return labels, np.column_stack((ln_w, ages, sds)), halo_table, bar_pops


def _bar_populations(bar_feh, bar_age, feh_sys, age_frac, age_floor):
    """Return the bar populations as rows of (ln weight, [Fe/H] mean and
    standard deviation, age mean and standard deviation, ln normalization
    of the age density), each width widened by the thick-disk systematics."""
    bar_feh = _floats("bar_feh", bar_feh, 5)
    bar_age = _floats("bar_age", bar_age, 4)
    if not 0.0 <= bar_feh[0] <= 1.0:
        raise ValueError(
            f"bar_feh[0], the metal-poor fraction, must be in [0, 1]; got {bar_feh}"
        )
    _check_positive(
        **{"bar_feh[2]": bar_feh[2], "bar_feh[4]": bar_feh[4]},
        **{"bar_age[1]": bar_age[1], "bar_age[3]": bar_age[3]},
    )
    rows = []
    for weight, k in ((bar_feh[0], 0), (1.0 - bar_feh[0], 2)):
        age_mean = bar_age[k]
        age_sd = _age_spread(age_mean, age_frac, age_floor, bar_age[k + 1])
        rows.append(
            (
                log(weight) if weight > 0.0 else LOG_ZERO,
                bar_feh[k + 1],
                sqrt(bar_feh[k + 2] ** 2 + feh_sys**2),
                age_mean,
                age_sd,
                _ln_age_norm(age_mean, age_sd),
            )
        )
    return np.array(rows)


def _bar_args(bar_mass, bar_angle, rho_thin_solar):
    """Return the bar arguments of `_galactic_prior_array`: the cosine and
    sine of ``bar_angle`` and ln of the bar amplitude relative to the thin
    disk at the Sun (LOG_ZERO for no bar)."""
    (bar_mass,) = _floats("bar_mass", (bar_mass,), 1)
    (bar_angle,) = _floats("bar_angle", (bar_angle,), 1)
    if bar_mass < 0:
        raise ValueError(f"bar_mass must not be negative, got {bar_mass}")
    angle = np.deg2rad(bar_angle)
    # Msun/kpc^3 to Msun/pc^3.
    amp = bar_mass / (BAR_MASS * 1e9 * float(rho_thin_solar))
    return cos(angle), sin(angle), log(amp) if amp > 0.0 else LOG_ZERO


def _bar_table(ell, b, R_solar, Z_solar, bar):
    """Return ln of the bar density at the _BAR_NODES distances of its table
    toward (l, b), or an empty table where the call has more than one
    sightline, R_solar is 13 kpc or more, or there is no bar."""
    ell, b = np.ravel(ell), np.ravel(b)
    if (
        np.ptp(ell) > 0
        or np.ptp(b) > 0
        or R_solar + _BAR_R_MAX >= 25.0
        or bar[2] <= LOG_ZERO
    ):
        return np.empty(0)
    d = np.exp(_BAR_LN_D0 + _BAR_DLN_D * np.arange(_BAR_NODES))
    X, Y, Z = galactic_to_galactocentric(d, ell[0], b[0], R_solar, Z_solar)
    return _ln_bar_densities(X, Y, Z, bar[0], bar[1])


def _normalizations(
    R_solar,
    Z_solar,
    R_thin,
    Rs_thin,
    R_thick,
    Rs_thick,
    f_thick_surface,
    v_circ_solar,
    thick_sigma_z,
    Rs_halo,
    eta_halo_inner,
    eta_halo_outer,
    r_break_halo,
    q_halo,
    halo_mass,
    bar_mass,
    bar_angle,
    rho_thin_solar,
    feh_systematic,
):
    """Return the disk and halo arguments of `_components`, the bar
    arguments of `_bar_args`, the tables of `_thin_disk_tables` and their
    key, and the normalizations that `galactic_prior_summary` reports."""
    _check_positive(
        R_thin=R_thin,
        R_thick=R_thick,
        f_thick_surface=f_thick_surface,
        thick_sigma_z=thick_sigma_z,
        Rs_halo=Rs_halo,
        eta_halo_inner=eta_halo_inner,
        eta_halo_outer=eta_halo_outer,
        r_break_halo=r_break_halo,
        q_halo=q_halo,
        halo_mass=halo_mass,
        rho_thin_solar=rho_thin_solar,
    )
    R_solar, Z_solar = float(R_solar), float(Z_solar)
    vc_solar = _vc_solar(v_circ_solar, R_solar)
    key = (
        R_solar,
        vc_solar,
        float(R_thin),
        float(Rs_thin),
        _thin_disk_sig_birth(feh_systematic),
    )
    tables = _thin_disk_tables(*key)
    ln_d0 = _ln_marginal(1.0, abs(Z_solar), tables[1])
    ln_f_thick = log(f_thick_surface)
    disk = (R_solar, 2.0 * log(vc_solar / _V_VC0))
    disk += tuple(map(float, (R_thin, Rs_thin, R_thick, Rs_thick)))
    disk += (ln_f_thick, float(thick_sigma_z) ** 2, ln_d0)
    shape = _halo_args(
        R_smooth=Rs_halo,
        eta_inner=eta_halo_inner,
        eta_outer=eta_halo_outer,
        r_break=r_break_halo,
        q=q_halo,
    )
    # Msun/kpc^3 to Msun/pc^3.
    ln_halo_amplitude = log(
        float(halo_mass) / (_halo_volume(shape) * 1e9 * float(rho_thin_solar))
    )
    halo = shape + (ln_halo_amplitude,)
    # At the Sun the thin disk is 1, so the others are their ratios to it;
    # without labels, the label arguments are not read.
    no_nodes = np.empty((0, 11))
    no_labels = (
        (0.0,) * 13,
        (0.0, 0.0, -1.0, no_nodes, np.empty(0), np.empty(0)),
    )
    _, thick, ln_f_halo = _components(
        R_solar,
        Z_solar,
        0.0,
        0.0,
        False,
        False,
        disk,
        halo,
        *no_labels,
        *tables,
        np.empty((0, 0, 0)),
    )
    bar = _bar_args(bar_mass, bar_angle, rho_thin_solar)
    summary = dict(
        thin_column_height=0.5 * exp(-ln_d0),
        thick_column_height=0.5 * exp(ln_f_thick - ln_d0 - thick),
        f_thick_solar=exp(thick),
        halo_density_solar=exp(ln_f_halo) * float(rho_thin_solar),
        f_halo_solar=exp(ln_f_halo),
        f_bar_solar=exp(_ln_bar_density(-R_solar, 0.0, Z_solar, *bar[:2]) + bar[2]),
    )
    return disk, halo, bar, tables, key, summary


__all__ = [
    "logn_disk",
    "logn_halo",
    "logp_galactic_structure",
    "galactic_prior_summary",
]


def logn_disk(
    R, Z, R_solar=R_SOLAR, Z_solar=Z_SOLAR, R_scale=2.74, Z_scale=0.3, R_smooth=2.0
):
    r"""
    Log-number density for the Galactic disk stellar population.

    Implements an exponential disk model with separate radial and vertical
    scale lengths, smoothed near the Galactic center to avoid singularities.

    Parameters
    ----------
    R : array_like
        Galactocentric cylindrical radius in kpc.
    Z : array_like
        Height above the Galactic midplane in kpc.
    R_solar : float, optional
        Solar Galactocentric radius in kpc.
        Default is ``R_SOLAR``, 8.275 (GRAVITY Collaboration 2021).
    Z_solar : float, optional
        Solar height above midplane in kpc.
        Default is ``Z_SOLAR``, 0.0208 (Bennett & Bovy 2019).
    R_scale : float, optional
        Disk radial scale length in kpc. Default is 2.74, that of the thin
        disk.
    Z_scale : float, optional
        Disk vertical scale height in kpc. Default is 0.3.
    R_smooth : float, optional
        Smoothing radius to avoid central singularity in kpc. Default is 2.0.

    Returns
    -------
    logn : array_like
        Normalized log-number density relative to Solar neighborhood.

    See Also
    --------
    logn_halo : Halo number density
    logp_galactic_structure : Combined disk+halo model

    Notes
    -----
    The disk number density follows:

    .. math::
        n_{\text{disk}}(R, Z) \propto \exp\left(-\frac{R_{\text{eff}} - R_{\text{eff},\odot}}{R_{\text{scale}}} - \frac{|Z| - |Z_\odot|}{Z_{\text{scale}}}\right)

    where :math:`R_{\text{eff}} = \sqrt{R^2 + R_{\text{smooth}}^2}` provides
    smoothing near the Galactic center and
    :math:`R_{\text{eff},\odot} = \sqrt{R_\odot^2 + R_{\text{smooth}}^2}`
    is the smoothed solar radius, so the normalized density is exactly 1 at
    the solar position (consistent with :func:`logn_halo`). This keeps the
    thick-disk and halo fractions of :func:`logp_galactic_structure` exact
    at the solar position.

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    """
    R = np.asarray(R)
    Z = np.asarray(Z)

    # Smoothed effective radius
    R_eff = np.sqrt(R**2 + R_smooth**2)

    # Exponential disk components, normalized at the SMOOTHED solar radius
    # so that logn_disk(R_solar, Z_solar) == 0 (as for logn_halo).
    R_eff_solar = np.sqrt(R_solar**2 + R_smooth**2)
    radial_term = (R_eff - R_eff_solar) / R_scale
    vertical_term = (np.abs(Z) - np.abs(Z_solar)) / Z_scale

    return -(radial_term + vertical_term)


def logn_halo(
    R,
    Z,
    R_solar=R_SOLAR,
    Z_solar=Z_SOLAR,
    R_smooth=2.0,
    eta_inner=2.85,
    eta_outer=4.55,
    r_break=19.1,
    q=0.77,
):
    r"""
    Return the log-number density of the stellar halo relative to the Sun.

    The halo is the broken power law of Amarante et al. (2024) in a
    flattened radius, with a core:

    .. math::
        \ln n_{\rm halo} = -\eta_{\rm in} \ln m
        - (\eta_{\rm out} - \eta_{\rm in}) \max(\ln (m / r_b), 0)
        - (\ldots)_\odot,
        \qquad
        m = \sqrt{R^2 + (Z/q)^2 + R_{\rm smooth}^2},

    Amarante et al. raise q to 0.99 beyond 30 kpc; here q is constant,
    which avoids a jump in density (see the priors page of the docs).

    Parameters
    ----------
    R : array_like
        Galactocentric cylindrical radius in kpc.
    Z : array_like
        Height above the Galactic midplane in kpc.
    R_solar : float, optional
        Solar Galactocentric radius in kpc.
        Default is ``R_SOLAR``, 8.275 (GRAVITY Collaboration 2021).
    Z_solar : float, optional
        Solar height above midplane in kpc.
        Default is ``Z_SOLAR``, 0.0208 (Bennett & Bovy 2019).
    R_smooth : float, optional
        Core radius in kpc, which keeps the density finite at the Galactic
        centre. Default is 2.0.
    eta_inner, eta_outer : float, optional
        Power-law slopes inside and outside ``r_break``. Defaults are 2.85
        and 4.55.
    r_break : float, optional
        Break radius of the flattened radius ``m`` in kpc. Default is 19.1.
    q : float, optional
        Flattening, the ratio of the vertical to the radial axis. Default
        is 0.77.

    Returns
    -------
    logn : numpy.ndarray
        Log-number density, 0 at the Sun.

    See Also
    --------
    logn_disk : Disk number density
    logp_galactic_structure : Combined disk+halo model

    References
    ----------
    Amarante, Koposov & Laporte (2024), A&A (arXiv:2404.09825)
    Lane, Bovy & Mackereth (2023), MNRAS (arXiv:2306.03084)
    """
    shape = _halo_args(R_smooth, eta_inner, eta_outer, r_break, q)
    R, Z = np.asarray(R, dtype=float), np.asarray(Z, dtype=float)
    return np.asarray(_halo_shape(R, Z, *shape) - _halo_shape(R_solar, Z_solar, *shape))


def logp_galactic_structure(
    dists,
    coord,
    labels=None,
    feh=None,
    loga=None,
    R_solar=R_SOLAR,
    Z_solar=Z_SOLAR,
    R_thin=2.74,
    Rs_thin=2.0,
    R_thick=1.9,
    f_thick_surface=0.18,
    Rs_thick=2.0,
    Rs_halo=2.0,
    eta_halo_inner=2.85,
    eta_halo_outer=4.55,
    r_break_halo=19.1,
    q_halo=0.77,
    halo_mass=7.5e8,
    bar_mass=BAR_MASS,
    bar_angle=BAR_ANGLE,
    rho_thin_solar=0.030,
    thick_feh=(-0.45, 0.22, -0.5),
    thick_age=(13.0, -4.0, 0.8),
    halo_feh=(-1.18, 0.28, -0.67, -1.95, 1.0),
    halo_age=(12.0, 1.5),
    bar_feh=(0.5, -0.31, 0.31, 0.26, 0.20),
    bar_age=(11.0, 1.5, 8.0, 3.0),
    feh_systematic=(0.1, 0.15, 0.15),
    age_systematic=(0.1, 0.15, 0.15),
    age_floor=0.2,
    v_circ_solar=None,
    thick_sigma_z=THICK_DISK_SIGMA[2],
    return_components=False,
):
    r"""
    Return the log-prior of distance, [Fe/H] and age from a model of the Galaxy.

    The prior is the number density of stars with the given labels at each
    distance along the sightline, times the volume factor :math:`d^2`,
    summed over four components: the thin disk, the thick disk, the halo
    and the bar. It is relative to the thin disk at the Sun.

    Parameters
    ----------
    dists : array_like
        Distance from observer in kpc, of any shape.
    coord : tuple of floats or `~astropy.coordinates.SkyCoord`
        Galactic coordinates (l, b) in degrees.
    labels : structured array, optional
        Stellar labels with the fields 'feh' and/or 'loga', in the shape of
        ``dists`` or of its trailing axes, or 1D with one entry per
        distance. Labels on the trailing axes broadcast across the leading
        ones: with ``dists`` of shape (Ndraws, Nmodels) and labels of shape
        (Nmodels,), the terms of each model's labels are computed once.
    feh : array_like, optional
        [Fe/H] as a float array, shaped as ``labels``, an alternative to the
        'feh' field of ``labels`` that takes precedence over it.
    loga : array_like, optional
        log10(age / yr) as a float array, shaped as ``labels``, an
        alternative to the 'loga' field of ``labels`` that takes precedence
        over it.
    R_solar : float, optional
        Solar Galactocentric radius in kpc.
        Default is ``R_SOLAR``, 8.275 (GRAVITY Collaboration 2021).
    Z_solar : float, optional
        Solar height above midplane in kpc.
        Default is ``Z_SOLAR``, 0.0208 (Bennett & Bovy 2019).
    R_thin : float, optional
        Radial scale length of the thin-disk surface density in kpc.
        Default is 2.74 (Imig et al. 2025).
    Rs_thin : float, optional
        Thin disk smoothing radius in kpc. Default is 2.0.
    R_thick : float, optional
        Radial scale length of the thick-disk surface density in kpc.
        Default is 1.9 (Mackereth et al. 2017).
    f_thick_surface : float, optional
        Ratio of the thick- to the thin-disk surface density at the Sun.
        Default is 0.18, the high-[alpha/Fe] surface mass fraction of
        Mackereth et al. (2017). The prior converts it to the ratio of
        densities at the Sun's position with the column heights of the
        vertical profiles; `galactic_prior_summary` returns both.
    Rs_thick : float, optional
        Thick disk smoothing radius in kpc. Default is 2.0.
    Rs_halo : float, optional
        Halo core radius in kpc. Default is 2.0.
    eta_halo_inner, eta_halo_outer, r_break_halo : float, optional
        Halo slopes inside and outside the break radius in kpc. Defaults
        are 2.85, 4.55 and 19.1 (Amarante et al. 2024).
    q_halo : float, optional
        Halo flattening. Default is 0.77 (Amarante et al. 2024).
    halo_mass : float, optional
        Halo stellar mass in Msun between Galactocentric radii of 2 and
        70 kpc. Default is 7.5e8, the middle of the 6.7e8 to 8.4e8 of
        Lane et al. (2023).
    bar_mass : float, optional
        Stellar mass of the bar in Msun. Default is ``BAR_MASS``, 1.8252e10,
        the integral of the density of Sormani et al. (2022). 0 removes the
        bar.
    bar_angle : float, optional
        Angle in degrees between the long axis of the bar and the line from
        the Sun to the Galactic centre, with the near end at l > 0. Default
        is ``BAR_ANGLE``, 25.0.
    rho_thin_solar : float, optional
        Thin-disk stellar mass density at the Sun in Msun/pc^3, in living
        stars. Default is 0.030. With ``halo_mass`` it sets the halo to
        thin-disk density ratio at the Sun that `galactic_prior_summary`
        returns, and with ``bar_mass`` the bar's amplitude.
    thick_feh : tuple of 3 floats, optional
        Mean, standard deviation and skewness of the skew-normal [Fe/H]
        distribution of the thick disk. Default is (-0.45, 0.22, -0.5):
        the high-[alpha/Fe] disk of Hayden et al. (2015), with its
        metal-poor tail.
    thick_age : tuple of 3 floats, optional
        Mean age in Gyr of thick-disk stars at [Fe/H] = -1, its change in
        Gyr per dex of [Fe/H] above -1, and the standard deviation in Gyr
        of a normal truncated to 0 to 13.8 Gyr. The mean is held below
        [Fe/H] = -1. Default is (13.0, -4.0, 0.8), the old sequence of
        Xiang & Rix (2022).
    halo_feh : tuple of 5 floats, optional
        The halo [Fe/H] distribution: the mean, standard deviation and
        skewness of a skew-normal core, the [Fe/H] below which a metal-poor
        tail replaces it, and the tail's slope k, with density proportional
        to 10^(k [Fe/H]). Default is (-1.18, 0.28, -0.67, -1.95, 1.0): the
        16th, 50th and 84th percentiles of Naidu et al. (2020), -1.48,
        -1.15 and -0.91, and the tail of Youakim et al. (2020).
    halo_age : tuple of 2 floats, optional
        Mean and standard deviation in Gyr of the halo age, a normal
        truncated to 0 to 13.8 Gyr. Default is (12.0, 1.5); the width
        covers the spread between age methods.
    bar_feh : tuple of 5 floats, optional
        The bar [Fe/H] distribution, two normal populations: the fraction
        of the metal-poor one, then the mean and standard deviation of the
        metal-poor and of the metal-rich population. Default is
        (0.5, -0.31, 0.31, 0.26, 0.20): the components of Gonzalez et al.
        (2015), with a fraction between the 0.33 and 0.73 that Zoccali et
        al. (2017) find at latitudes -3.5 and -8.5 deg.
    bar_age : tuple of 4 floats, optional
        Mean and standard deviation in Gyr of the age of the metal-poor and
        of the metal-rich population, each a normal truncated to 0 to 13.8
        Gyr. Default is (11.0, 1.5, 8.0, 3.0), after Bensby et al. (2017).
    feh_systematic : tuple of 3 floats, optional
        Standard deviation in dex of the difference between the [Fe/H]
        scale of the grid and that of the measured distributions, for the
        thin disk, thick disk and halo; the bar takes the thick-disk value.
        Each [Fe/H] distribution is
        convolved with it. Default is (0.1, 0.15, 0.15): survey [Fe/H]
        scales differ by about 0.1 dex, and a solar-scaled grid reads
        alpha-rich stars 0.14 to 0.21 dex high.
    age_systematic : tuple of 3 floats, optional
        Fractional standard deviation of the difference between the age
        scale of the grid and that of the measured distributions, for the
        thin disk, thick disk and halo (the bar takes the thick-disk value),
        added in quadrature to
        ``age_floor``. Default is (0.1, 0.15, 0.15): grids differ by about
        10% in age, and a solar-scaled grid makes alpha-rich stars 1 to
        2 Gyr older.
    age_floor : float, optional
        Absolute part of the age systematic in Gyr. Default is 0.2.
    v_circ_solar : float or None, optional
        Circular velocity at the Sun in km/s, which scales the measured
        rotation curve of `~brutus.priors.thin_disk_velocity_moments`.
        Default (None) is the Sgr A* proper motion less the solar peculiar
        velocity, as in `~brutus.priors.solar_velocity`. The rotation curve
        sets the angular momentum on which the thin-disk vertical
        dispersion depends, and the vertical potential scales as its
        square.
    thick_sigma_z : float, optional
        Vertical dispersion of the thick disk at the Sun in km/s. Default
        is ``brutus.priors.kinematic.THICK_DISK_SIGMA[2]``.
    return_components : bool, optional
        Whether to return the components. Default is False.

    Returns
    -------
    logp : numpy.ndarray
        Log-prior, ln of the number density per kpc^3, per dex of [Fe/H]
        and per Gyr of age (for the labels given) times :math:`d^2`.
    components : dict, optional
        If ``return_components``: ``'number_density'``, the list of ln
        number density times :math:`d^2` of the thin disk, thick disk, halo
        and bar; and, with labels, ``'labels'``, the list of ln p(labels) of
        each component at that position, 0 where the component has no
        stars (the bar beyond 12 kpc from the centre, or with
        ``bar_mass=0``). Their sums are the
        terms of ``logp``.

    Raises
    ------
    ValueError
        If ``labels``, ``feh`` or ``loga`` is not shaped as above or not
        finite, if a label-distribution keyword is not a tuple of
        finite numbers with positive widths, or if a keyword that sets a
        normalization is not positive and finite (see
        `galactic_prior_summary`).

    Notes
    -----
    Each disk has an exponential surface density with scale length
    ``R_thin`` or ``R_thick``, as in `logn_disk`. A disk population's
    density is its surface density times :math:`\nu(z) / 2H`, with
    :math:`\nu = 1` at the midplane and :math:`H` its column. The
    vertical profiles solve the vertical Jeans equation in the potential
    of the Milky Way model of Binney & Vasiliev (2024), tabulated in
    ``brutus.priors.kinematic``:

    .. math::
        \ln \nu(z) = -2 \ln f_z(|z|) - \frac{1}{\sigma_{z,0}^2}
        \int_0^{|z|} \frac{K_z(t)}{f_z(t)^2}\,dt

    A thin-disk population of a given age and [Fe/H] takes the vertical
    dispersion of Sharma et al. (2021), which grows with height as
    :math:`f_z = 1 + \gamma_z |z|` up to 2 kpc. Its ages and [Fe/H] at
    radius R follow the star formation, enrichment and radial migration
    model of Frankel et al. (2018), read as the distribution per unit
    area and held outside the radii of their fit, 5 to 14 kpc
    (``THIN_DISK_*`` constants). Labels not given are summed over. A
    thin-disk star's labelled age is a normal about its true age, of the
    age systematic and truncated to 0 to 13.8 Gyr, and its vertical
    profile is that of the true age; labelled ages above 13.8 Gyr have
    zero prior. With one label, the thin disk is interpolated in a table
    against that label, R and z, built once per parameter set in about
    1 s; with both, it sums over 16 true ages at each point. The
    thick disk is isothermal (:math:`\gamma_z = 0`), with
    ``thick_sigma_z`` at the Sun and the radial profile of
    `~brutus.priors.thick_disk_velocity_moments`; its labels and those of
    the halo do not depend on position. The halo, `logn_halo`, is
    normalized by ``halo_mass``, as in `galactic_prior_summary`.

    The bar is the analytic density of Sormani et al. (2022), a fit to the
    dynamical model of Portail et al. (2017): an X-shaped bar and a long
    bar, which together reach 5 kpc from the centre. It is turned by
    ``bar_angle`` and scaled to ``bar_mass``, and is left out beyond
    12 kpc from the Galactic centre, where it is at most 1.4e-4 of the
    disks.
    Its labels do not depend on position, and its [Fe/H] and ages are
    widened by the thick-disk systematics.

    References
    ----------
    Frankel et al. (2018), ApJ, 865, 96
    Xiang & Rix (2022), Nature, 603, 599
    Naidu et al. (2020), ApJ, 901, 48
    Youakim et al. (2020), MNRAS, 492, 4986
    Sormani et al. (2022), MNRAS, 514, L1
    Portail et al. (2017), MNRAS, 465, 1621
    Gonzalez et al. (2015), A&A, 584, A46
    Zoccali et al. (2017), A&A, 599, A12
    Bensby et al. (2017), A&A, 605, A89
    """
    dists = np.asarray(dists, dtype=float)

    if hasattr(coord, "galactic"):
        ell, b = coord.galactic.l.deg, coord.galactic.b.deg
    else:
        ell, b = coord[0], coord[1]

    # A plain array overrides only its own field of `labels`.
    if labels is not None:
        if getattr(labels, "dtype", None) is None or labels.dtype.names is None:
            raise ValueError(
                "`labels` must be a structured array with the fields 'feh' "
                "and/or 'loga'."
            )
        _label_shape("labels", np.shape(labels), dists.shape)
        if feh is None and "feh" in labels.dtype.names:
            feh = labels["feh"]
        if loga is None and "loga" in labels.dtype.names:
            loga = labels["loga"]
    feh, loga = (
        _label_array(name, v, dists.shape) for name, v in (("feh", feh), ("loga", loga))
    )

    label_args = _label_args(
        thick_feh,
        thick_age,
        halo_feh,
        halo_age,
        bar_feh,
        bar_age,
        feh_systematic,
        age_systematic,
        age_floor,
    )
    disk, halo, bar, tables, key, _ = _normalizations(
        R_solar=R_solar,
        Z_solar=Z_solar,
        R_thin=R_thin,
        Rs_thin=Rs_thin,
        R_thick=R_thick,
        Rs_thick=Rs_thick,
        f_thick_surface=f_thick_surface,
        v_circ_solar=v_circ_solar,
        thick_sigma_z=thick_sigma_z,
        Rs_halo=Rs_halo,
        eta_halo_inner=eta_halo_inner,
        eta_halo_outer=eta_halo_outer,
        r_break_halo=r_break_halo,
        q_halo=q_halo,
        halo_mass=halo_mass,
        bar_mass=bar_mass,
        bar_angle=bar_angle,
        rho_thin_solar=rho_thin_solar,
        feh_systematic=feh_systematic,
    )
    by_label = _label_table(key, feh is not None, loga is not None, label_args[0])

    X, Y, Z = np.broadcast_arrays(
        *galactic_to_galactocentric(dists, ell, b, float(R_solar), float(Z_solar))
    )
    bar_line = _bar_table(ell, b, float(R_solar), float(Z_solar), bar)

    # Array coordinates broadcast the positions past the shape of `dists`.
    # Labels that span only their trailing axes are kept unbroadcast, so
    # that their terms are computed once per label, unless they are too few
    # to share among the threads.
    given = [v.shape for v in (feh, loga) if v is not None]
    label_shape = X.shape
    if given and given.count(given[0]) == len(given):
        trailing = X.shape[X.ndim - len(given[0]) :] == given[0]
        if trailing and np.prod(given[0]) >= 4 * get_num_threads():
            label_shape = given[0]

    def per_label(v):
        return np.empty(0) if v is None else np.broadcast_to(v, label_shape).ravel()

    args = (
        np.broadcast_to(dists, X.shape).ravel(),
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
        per_label(feh),
        per_label(loga),
        int(np.prod(label_shape)),
    )
    has = (feh is not None, loga is not None)
    rest = (disk, halo, bar, bar_line) + label_args + tables + (by_label,)
    if not return_components:
        return _galactic_prior_array(*args, *has, *rest, np.empty((0, 0))).reshape(
            X.shape
        )
    joint = np.empty((4, X.size))
    logp = _galactic_prior_array(*args, *has, *rest, joint).reshape(X.shape)
    if not any(has):
        return logp, {"number_density": list(joint.reshape((4,) + X.shape))}
    density = np.empty_like(joint)
    _galactic_prior_array(*args, False, False, *rest, density)
    return logp, {
        "number_density": list(density.reshape((4,) + X.shape)),
        "labels": list((joint - density).reshape((4,) + X.shape)),
    }


def _label_table(key, has_feh, has_age, labels):
    """Return the thin-disk table by the one label given, from the key of
    `_normalizations` and the labels of `_label_args`, or an empty table."""
    if has_feh and not has_age:
        return _thin_disk_feh_table(*key)
    if has_age and not has_feh:
        return _thin_disk_age_table(*key, labels[1], labels[2])
    return np.empty((0, 0, 0))


def _label_shape(name, shape, dists_shape):
    """Raise ValueError unless ``shape`` is that of `dists`, of its
    trailing axes, or 1D with one entry per distance."""
    trailing = dists_shape[len(dists_shape) - len(shape) :] == shape
    if not (trailing or shape == (int(np.prod(dists_shape)),)):
        raise ValueError(
            f"`{name}` must have the shape of `dists` {dists_shape}, of its "
            f"trailing axes, or one entry per distance; got {shape}."
        )


def _label_array(name, value, dists_shape):
    """Return ``value`` as a finite float64 array in the shape of `dists`
    or of its trailing axes, or None."""
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64)
    _label_shape(name, value.shape, dists_shape)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"`{name}` must be finite.")
    if dists_shape[len(dists_shape) - value.ndim :] != value.shape:
        value = value.reshape(dists_shape)
    return value


def galactic_prior_summary(**kwargs):
    r"""
    Return the normalizations that `logp_galactic_structure` derives.

    The column height of a disk at the Sun is
    :math:`h = \Sigma / (2 \nu(Z_\odot))`, with the thin disk summed over
    its ages and [Fe/H], and the thick disk's density ratio at the Sun is

    .. math::
        f_{\rm thick} = f_{\Sigma} \, \frac{h_{\rm thin}}{h_{\rm thick}}.

    The halo density at the Sun is its mass over the volume integral of
    its shape between Galactocentric radii of 2 and 70 kpc, and its ratio
    to the thin disk is

    .. math::
        f_{\rm halo} = \frac{M_{\rm halo}}{\rho_{\rm thin,\odot}
        \int_{r_1}^{r_2} n_{\rm halo}\, dV}.

    Parameters
    ----------
    **kwargs
        Keywords of `logp_galactic_structure` that set the normalizations:
        ``R_solar``, ``Z_solar``, the disk scale lengths and smoothing
        radii, ``f_thick_surface``, ``v_circ_solar``, ``thick_sigma_z``,
        the halo-shape keywords, ``halo_mass``, ``bar_mass``,
        ``bar_angle``, ``rho_thin_solar`` and ``feh_systematic``, whose
        thin-disk entry changes the thin-disk column. Omitted keywords take
        its defaults.

    Returns
    -------
    summary : dict
        ``thin_column_height`` and ``thick_column_height`` in kpc,
        ``f_thick_solar`` (the thick- to thin-disk density ratio at the
        Sun's position), ``halo_density_solar`` in Msun/pc^3,
        ``f_halo_solar`` (the halo to thin-disk density ratio at the Sun)
        and ``f_bar_solar`` (the same for the bar).

    Raises
    ------
    TypeError
        If a keyword does not set a normalization.
    ValueError
        If ``R_thin``, ``R_thick``, ``f_thick_surface``, ``thick_sigma_z``,
        ``v_circ_solar``, ``Rs_halo``, ``eta_halo_inner``,
        ``eta_halo_outer``, ``r_break_halo``, ``q_halo``, ``halo_mass`` or
        ``rho_thin_solar`` is not positive and finite, if ``bar_mass`` is
        negative or not finite, or if ``bar_angle`` is not finite.

    Examples
    --------
    >>> from brutus.priors import galactic_prior_summary
    >>> round(galactic_prior_summary()["f_halo_solar"], 5)
    0.00177
    """
    unknown = set(kwargs) - set(_NORMALIZATION_KEYS)
    if unknown:
        raise TypeError(
            f"galactic_prior_summary() got keywords that set no "
            f"normalization: {sorted(unknown)}."
        )
    return _normalizations(**{**_NORMALIZATION_DEFAULTS, **kwargs})[-1]


_NORMALIZATION_DEFAULTS = {
    k: v.default
    for k, v in inspect.signature(logp_galactic_structure).parameters.items()
    if k in inspect.signature(_normalizations).parameters
}
_NORMALIZATION_KEYS = tuple(_NORMALIZATION_DEFAULTS)
