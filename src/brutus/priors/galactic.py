#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Galactic structure priors for Bayesian stellar parameter estimation.

This module provides log-prior functions for Galactic structure modeling
including disk and halo number densities, metallicity distributions,
and age-metallicity relations. These priors encode our knowledge of the
Milky Way's stellar populations and are essential for distance estimation.

See Also
--------
brutus.analysis.individual.BruteForce : Uses Galactic priors for distance fitting
brutus.priors.stellar : Stellar population priors
brutus.priors.astrometric : Parallax priors

Notes
-----
These priors provide critical constraints on stellar distances by
incorporating knowledge of Galactic structure:

- **Number density priors** (disk and halo) weight distance based on
  expected stellar distributions in the Galaxy

- **Metallicity priors** provide realistic [Fe/H] distributions for
  disk and halo populations

- **Age-metallicity relations** link stellar age and composition

The combined prior `logp_galactic_structure` integrates disk and halo
models with appropriate mixing fractions.

The priors use Galactocentric coordinates, which requires coordinate
transformations from equatorial coordinates and distances.

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
from math import exp, log, pi, sqrt

import numpy as np
from numba import jit, prange

from brutus.priors.kinematic import (
    _V_DPHI,
    _V_DX,
    _V_VC0,
    THICK_DISK_SIGMA,
    THIN_DISK_Z_EDGE,
    THIN_DISK_Z_MAX,
    _catmull_rom_table,
    _check_positive,
    _circular_velocity,
    _logn_disk_dlnR,
    _thick_disk_ln_var,
    _thick_disk_vertical,
    _thin_disk_fade,
    _thin_disk_feh_factor,
    _thin_disk_fL,
    _thin_disk_L_refs,
    _thin_disk_mean_L,
    _thin_disk_sigma0,
    _thin_disk_vertical,
    _vc_solar,
    _vertical_terms,
)
from brutus.utils import truncnorm_logpdf
from brutus.utils.math import (
    LOG_ZERO,
    _truncnorm_lognorm_scalar,
    galactic_to_galactocentric_cyl,
)

_LOG_2PI = log(2.0 * pi)


def _logsumexp3(a, b, c):
    """Fast logsumexp of 3 arrays using numpy's logaddexp ufunc."""
    return np.logaddexp(np.logaddexp(a, b), c)


# Gauss-Legendre nodes for the average over the thin-disk age prior, used
# when no age is given. They span the ages where the prior is within e^-40
# of its peak, so a narrow prior is resolved as well as a wide one.
_AGE_NODES = 32


@lru_cache(maxsize=16)
def _thin_age_nodes(feh_thin, age_args):
    """Return the thin-disk (sigma_R, sigma_z) model at [Fe/H] = 0 for each
    node of the age average, and the log weights of the nodes, whose
    exponentials sum to 1 under the age prior
    ``logp_age_from_feh(age, feh_thin, *age_args)``. The arrays are
    read-only, because the cache returns them to every caller."""
    max_age, min_age = age_args[:2]
    grid = np.linspace(min_age, max_age, 2001)
    lnp = logp_age_from_feh(grid, feh_thin, *age_args)
    keep = np.flatnonzero(lnp > lnp.max() - 40.0)
    lo, hi = grid[max(keep[0] - 1, 0)], grid[min(keep[-1] + 1, grid.size - 1)]
    x, w = np.polynomial.legendre.leggauss(_AGE_NODES)
    ages = 0.5 * (hi - lo) * (x + 1.0) + lo
    ln_w = np.log(w) + logp_age_from_feh(ages, feh_thin, *age_args)
    sig = np.array([_thin_disk_sigma0(a, 0.0) for a in ages])
    nodes = (sig[:, 0].copy(), sig[:, 1].copy(), ln_w - np.logaddexp.reduce(ln_w))
    for a in nodes:
        a.flags.writeable = False
    return nodes


# Gauss-Hermite nodes for the average of the thin-disk column height over
# the [Fe/H] prior, and the heights (kpc) and weights of the column
# integrals: 8-point Gauss-Legendre panels, with edges at the ends of the
# thin-disk sigma_z blend (1.5 and 2.5 kpc), out to 100 kpc because the
# potential binds a hot disk only logarithmically. Relative error 1.4e-6.
_FEH_NODES = 8
_COLUMN_EDGES = np.array(
    [0.0, 0.25, 0.5, 1.0]
    + [THIN_DISK_Z_MAX - THIN_DISK_Z_EDGE, THIN_DISK_Z_MAX + THIN_DISK_Z_EDGE]
    + [4.0, 7.0, 12.0, 20.0, 35.0, 60.0, 100.0]
)


def _gauss_legendre_panels(edges, n):
    """Return the nodes and weights of n-point Gauss-Legendre quadrature on
    each interval between ``edges``."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    lo, hi = edges[:-1, None], edges[1:, None]
    return (0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes).ravel(), (
        0.5 * (hi - lo) * weights
    ).ravel()


_COLUMN_Z, _COLUMN_W = _gauss_legendre_panels(_COLUMN_EDGES, 8)


# Step in x = R / R_solar of the column-height tables, which run from 0 to
# the end of the potential table; node 1 / _COLUMN_DX is the Sun.
_COLUMN_DX = 0.1
_COLUMN_X = np.arange(round(_V_DX * (len(_V_DPHI) - 1) / _COLUMN_DX) + 1) * _COLUMN_DX


@lru_cache(maxsize=16)
def _disk_columns(rot, abs_Z_solar, thin_R, thick, feh_thin, feh_thin_sigma, age_args):
    """Return the column heights Sigma / (2 nu(Z_sun)) in kpc of the thin and
    thick disks at R_solar, and ln(h(R) / h(R_solar)) of each at
    R = _COLUMN_X R_solar. The thin disk averages its vertical profile
    over the age prior and, by Gauss-Hermite quadrature, over its [Fe/H]
    prior, both of which are densities at the Sun's height. Costs 0.07 s,
    or 10 s with NUMBA_DISABLE_JIT."""
    x, w = np.polynomial.hermite_e.hermegauss(_FEH_NODES)
    shape = (_COLUMN_X.size, x.size, _COLUMN_Z.size)
    R = np.repeat(_COLUMN_X * rot[0], x.size * _COLUMN_Z.size)
    feh = np.tile(np.repeat(feh_thin + feh_thin_sigma * x, _COLUMN_Z.size), shape[0])
    args = (R, np.tile(_COLUMN_Z, shape[0] * shape[1]), abs_Z_solar)
    args += (np.empty(0), feh, False, rot, thin_R, *thick)
    thin_z, thick_z = _disk_vertical_array(*args, *_thin_age_nodes(feh_thin, age_args))
    thin = np.exp(thin_z).reshape(shape) @ _COLUMN_W @ w / w.sum()
    # The thick disk does not depend on [Fe/H]: the first node's profile serves.
    thick = np.exp(thick_z).reshape(shape)[:, 0] @ _COLUMN_W
    sun = round(1.0 / _COLUMN_DX)
    ln_thin, ln_thick = np.log(thin / thin[sun]), np.log(thick / thick[sun])
    for a in (ln_thin, ln_thick):
        a.flags.writeable = False
    return float(thin[sun]), float(thick[sun]), ln_thin, ln_thick


@jit(nopython=True, cache=True)
def _ln_column(x, table):
    """Return ln(h(R) / h(R_solar)) at x = R / R_solar from a table of
    ``_disk_columns``, by Catmull-Rom interpolation, mirrored about x = 0
    and held beyond its second-to-last node (x = 3.9)."""
    if np.isnan(x):
        return np.nan
    return _catmull_rom_table(x / _COLUMN_DX, table)


@jit(nopython=True, cache=True)
def _ln_column_array(x, table):
    """`_ln_column` at each element of the 1-D array ``x``."""
    out = np.empty(len(x))
    for k in range(len(x)):
        out[k] = _ln_column(x[k], table)
    return out


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


@jit(nopython=True, cache=True)
def _vertical_solar(abs_Z_solar, rot):
    """Return what ``_disk_vertical_terms`` needs that is fixed per call, as
    one flat tuple (numba's parallel loops take no nested tuples): the L_z
    references of the thin disk, the Sun's height and the factor
    (vc_solar / v_c(R0))^2 that scales the tabulated potential to the
    rotation curve, as x = R / R_solar scales its radii. ``rot`` is
    (R_solar, vc_solar)."""
    L_solar, L_clamp = _thin_disk_L_refs(*rot)
    return (L_solar,) + L_clamp + (abs_Z_solar, (rot[1] / _V_VC0) ** 2)


@jit(nopython=True, cache=True)
def _disk_vertical_terms(
    R,
    abs_Z,
    age,
    feh,
    has_age,
    rot,
    thin_R,
    solar,
    thick_var_z,
    age_sig_R,
    age_sig_z,
    age_ln_w,
):
    """Return ln nu(|Z|) - ln nu(|Z_sun|) of the thin and thick disks.

    With an age, the thin disk uses that age's vertical dispersion. Without
    one, it averages the profile over the age nodes ``age_sig_R``,
    ``age_sig_z`` and ``age_ln_w`` from ``_thin_age_nodes``. [Fe/H] scales
    the dispersions in both, and f_L is taken at the population's mean L_z,
    which uses the radial density gradient of the thin disk with
    ``thin_R`` = (scale length, smoothing radius). ``solar`` comes from
    ``_vertical_solar``.
    """
    L_solar, L_lo, L_hi, L_w, abs_Z_solar, phi_scale = solar
    L_clamp = (L_lo, L_hi, L_w)
    x = R / rot[0]
    terms, terms_solar = _vertical_terms(x, abs_Z), _vertical_terms(x, abs_Z_solar)
    vc, dlnvc = _circular_velocity(R, *rot)
    dlnnu = _logn_disk_dlnR(R, sqrt(R * R + thin_R[1] * thin_R[1]), thin_R[0])
    a, b = _thin_disk_vertical(terms, terms_solar)
    b *= phi_scale
    jeans = (L_solar, L_clamp, dlnnu, _thin_disk_fade(R))
    if has_age:
        sig_R0, sig_z0 = _thin_disk_sigma0(age, feh)
        L = _thin_disk_mean_L(R, 0.0, sig_R0, vc, dlnvc, *jeans)
        sig_z = sig_z0 * _thin_disk_fL(L, L_solar, L_clamp)[1]
        thin = a - b / (sig_z * sig_z)
    else:
        # a + ln sum_k w_k exp(-b / sig_k^2), with a running maximum.
        f_R, f_z = _thin_disk_feh_factor(feh)
        mx = LOG_ZERO
        acc = 0.0
        for k in range(len(age_ln_w)):
            L = _thin_disk_mean_L(R, 0.0, age_sig_R[k] * f_R, vc, dlnvc, *jeans)
            sig_z = age_sig_z[k] * f_z * _thin_disk_fL(L, L_solar, L_clamp)[1]
            v = age_ln_w[k] - b / (sig_z * sig_z)
            if v > mx:
                acc = acc * exp(mx - v) + 1.0
                mx = v
            else:
                acc += exp(v - mx)
        thin = a + mx + log(acc)
    var_z = thick_var_z * exp(_thick_disk_ln_var(x, 2)[0])
    thick = _thick_disk_vertical(terms, terms_solar, var_z / phi_scale)
    return thin, thick


@jit(nopython=True, parallel=True, cache=True)
def _disk_vertical_array(
    R,
    abs_Z,
    abs_Z_solar,
    loga,
    feh,
    has_age,
    rot,
    thin_R,
    thick_var_z,
    age_sig_R,
    age_sig_z,
    age_ln_w,
):
    solar = _vertical_solar(abs_Z_solar, rot)
    thin = np.empty(len(R))
    thick = np.empty(len(R))
    for i in prange(len(R)):
        thin[i], thick[i] = _disk_vertical_terms(
            R[i],
            abs_Z[i],
            10.0 ** loga[i] / 1e9 if has_age else 0.0,
            feh[i],
            has_age,
            rot,
            thin_R,
            solar,
            thick_var_z,
            age_sig_R,
            age_sig_z,
            age_ln_w,
        )
    return thin, thick


@jit(nopython=True, parallel=True, cache=True)
def _galactic_prior_fused(
    dists,
    sin_l,
    cos_l,
    sin_b,
    cos_b,
    feh_arr,
    loga_arr,
    has_feh,
    has_loga,
    R_solar,
    Z_solar,
    # Thin disk
    R_thin,
    Rs_thin,
    # Thick disk
    R_thick,
    Rs_thick,
    f_thick,
    ln_h_thin,
    ln_h_thick,
    # Vertical profiles
    vc_solar,
    thick_var_z,
    age_sig_R,
    age_sig_z,
    age_ln_w,
    # Halo
    halo,
    ln_halo_amplitude,
    # Metallicity
    feh_thin_mean,
    feh_thin_sigma,
    feh_thick_mean,
    feh_thick_sigma,
    feh_halo_mean,
    feh_halo_sigma,
    # Age
    max_age,
    min_age,
    feh_age_ctr,
    feh_age_scale,
    nsigma_from_max_age,
    max_sigma_age,
    min_sigma_age,
):
    """Fused galactic prior: coords + density + vol + met + age in one pass.

    The disk vertical profiles come from ``_disk_vertical_terms``; without
    ``loga``, the thin disk averages over the age nodes ``age_sig_R``,
    ``age_sig_z`` and ``age_ln_w``.

    ``sin_l``/``cos_l``/``sin_b``/``cos_b`` are the precomputed sine/cosine
    of the (scalar) Galactic longitude/latitude; the Galactocentric (R, Z)
    conversion is folded into the parallel loop with the exact operation
    order of ``galactic_to_galactocentric_cyl`` (bitwise-identical results,
    no full-size temporaries).
    """
    N = len(dists)
    logp_out = np.empty(N)

    # --- Loop-invariant constants (hoisted out of the prange loop) ---

    # Solar disk normalizations use the SMOOTHED solar radius so the disk
    # density is exactly 1 at the solar position (consistent with
    # ``logn_disk``); otherwise the f_thick mixture fraction is silently
    # rescaled by ~10%.
    R_solar_eff_thin = sqrt(R_solar**2 + Rs_thin**2)
    R_solar_eff_thick = sqrt(R_solar**2 + Rs_thick**2)
    abs_Z_solar = abs(Z_solar)
    rot = (R_solar, vc_solar)
    thin_R = (R_thin, Rs_thin)
    solar = _vertical_solar(abs_Z_solar, rot)
    ln_f_thick = log(f_thick)

    # Age-prior constants depend only on the component metallicity means,
    # not on the grid point, so they are computed once per call.
    age_mean_c = np.empty(3)
    age_sigma_c = np.empty(3)
    lndenom_c = np.empty(3)
    for comp_idx in range(3):
        if comp_idx == 0:
            fm = feh_thin_mean
        elif comp_idx == 1:
            fm = feh_thick_mean
        else:
            fm = feh_halo_mean
        age_mean = (max_age - min_age) / (
            1.0 + exp((fm - feh_age_ctr) / feh_age_scale)
        ) + min_age
        age_sigma = (max_age - age_mean) / nsigma_from_max_age
        if age_sigma < min_sigma_age:
            age_sigma = min_sigma_age
        if age_sigma > max_sigma_age:
            age_sigma = max_sigma_age
        alpha = (min_age - age_mean) / age_sigma
        beta = (max_age - age_mean) / age_sigma
        age_mean_c[comp_idx] = age_mean
        age_sigma_c[comp_idx] = age_sigma
        lndenom_c[comp_idx] = log(age_sigma) + _truncnorm_lognorm_scalar(alpha, beta)

    for i in prange(N):
        d = dists[i]

        # Galactocentric cylindrical coordinates (same operation order as
        # galactic_to_galactocentric_cyl for bitwise-identical results;
        # explicit multiplication instead of **2 matches numpy's squaring
        # fast path in both JIT and NUMBA_DISABLE_JIT modes)
        x = d * cos_b * cos_l - R_solar
        y = d * cos_b * sin_l
        R = sqrt(x * x + y * y)
        Z = d * sin_b + Z_solar

        # Volume factor
        vol = 2.0 * log(d + 1e-300)

        # Disk vertical profiles from the vertical Jeans equation
        age_val = 10.0 ** loga_arr[i] / 1e9 if has_loga else 0.0  # Gyr
        thin_z, thick_z = _disk_vertical_terms(
            R,
            abs(Z),
            age_val,
            feh_arr[i] if has_feh else feh_thin_mean,
            has_loga,
            rot,
            thin_R,
            solar,
            thick_var_z,
            age_sig_R,
            age_sig_z,
            age_ln_w,
        )

        # Disks: exponential surface densities over the column heights.
        x_col = R / R_solar
        R_eff_thin = sqrt(R**2 + Rs_thin**2)
        lnp_thin = -(R_eff_thin - R_solar_eff_thin) / R_thin + thin_z + vol
        lnp_thin -= _ln_column(x_col, ln_h_thin)
        R_eff_thick = sqrt(R**2 + Rs_thick**2)
        lnp_thick = (
            -(R_eff_thick - R_solar_eff_thick) / R_thick + thick_z + vol + ln_f_thick
        )
        lnp_thick -= _ln_column(x_col, ln_h_thick)

        # Halo
        lnp_halo = _halo_shape(R, Z, *halo) + vol + ln_halo_amplitude

        # logsumexp3
        mx = max(lnp_thin, max(lnp_thick, lnp_halo))
        logp_total = mx + log(
            exp(lnp_thin - mx) + exp(lnp_thick - mx) + exp(lnp_halo - mx)
        )

        # Metallicity prior
        feh_lnp = 0.0
        feh_lnp_thin = 0.0
        feh_lnp_thick = 0.0
        feh_lnp_halo = 0.0
        if has_feh:
            feh_val = feh_arr[i]
            # Component membership
            ln_w_thin = lnp_thin - logp_total
            ln_w_thick = lnp_thick - logp_total
            ln_w_halo = lnp_halo - logp_total

            # Gaussian logpdf for each component
            feh_lnp_thin = (
                -0.5
                * (
                    (feh_val - feh_thin_mean) ** 2 / feh_thin_sigma**2
                    + _LOG_2PI
                    + 2 * log(feh_thin_sigma)
                )
                + ln_w_thin
            )
            feh_lnp_thick = (
                -0.5
                * (
                    (feh_val - feh_thick_mean) ** 2 / feh_thick_sigma**2
                    + _LOG_2PI
                    + 2 * log(feh_thick_sigma)
                )
                + ln_w_thick
            )
            feh_lnp_halo = (
                -0.5
                * (
                    (feh_val - feh_halo_mean) ** 2 / feh_halo_sigma**2
                    + _LOG_2PI
                    + 2 * log(feh_halo_sigma)
                )
                + ln_w_halo
            )

            mx2 = max(feh_lnp_thin, max(feh_lnp_thick, feh_lnp_halo))
            feh_lnp = mx2 + log(
                exp(feh_lnp_thin - mx2)
                + exp(feh_lnp_thick - mx2)
                + exp(feh_lnp_halo - mx2)
            )
            logp_total += feh_lnp

        # Age prior
        if has_loga:
            if has_feh:
                # Metallicity-updated membership weights: the joint mixture
                # prior is sum_c w_c(pos) p(feh|c) p(age|c), so the age term
                # must weight components by w_c(pos) p(feh|c) / Z_feh rather
                # than by the position-only w_c(pos).
                ln_w_thin = feh_lnp_thin - feh_lnp
                ln_w_thick = feh_lnp_thick - feh_lnp
                ln_w_halo = feh_lnp_halo - feh_lnp
            else:
                ln_w_thin = lnp_thin - logp_total
                ln_w_thick = lnp_thick - logp_total
                ln_w_halo = lnp_halo - logp_total

            # Compute truncated normal for each component
            age_lnp_total = LOG_ZERO
            for comp_idx in range(3):
                if comp_idx == 0:
                    ln_w = ln_w_thin
                elif comp_idx == 1:
                    ln_w = ln_w_thick
                else:
                    ln_w = ln_w_halo

                xi = (age_val - age_mean_c[comp_idx]) / age_sigma_c[comp_idx]
                lnphi = -0.5 * (_LOG_2PI + xi * xi)
                age_comp = lnphi - lndenom_c[comp_idx] + ln_w

                # Bounds check
                if age_val < min_age or age_val > max_age:
                    age_comp = LOG_ZERO

                # Accumulate logsumexp
                if age_comp > age_lnp_total:
                    age_lnp_total = (
                        age_comp + log(1.0 + exp(age_lnp_total - age_comp))
                        if age_lnp_total > -1e200
                        else age_comp
                    )
                else:
                    age_lnp_total = (
                        age_lnp_total + log(1.0 + exp(age_comp - age_lnp_total))
                        if age_comp > -1e200
                        else age_lnp_total
                    )

            logp_total += age_lnp_total

        logp_out[i] = logp_total

    return logp_out


__all__ = [
    "logn_disk",
    "logn_halo",
    "logp_feh",
    "logp_age_from_feh",
    "logp_galactic_structure",
    "galactic_prior_summary",
]


def logn_disk(
    R, Z, R_solar=8.2, Z_solar=0.025, R_scale=2.74, Z_scale=0.3, R_smooth=2.0
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
        Solar Galactocentric radius in kpc. Default is 8.2.
    Z_solar : float, optional
        Solar height above midplane in kpc. Default is 0.025.
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
    R_solar=8.2,
    Z_solar=0.025,
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
        Solar Galactocentric radius in kpc. Default is 8.2.
    Z_solar : float, optional
        Solar height above midplane in kpc. Default is 0.025.
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


def logp_feh(feh, feh_mean=-0.2, feh_sigma=0.3):
    r"""
    Log-prior for stellar metallicity in Galactic components.

    Implements a Gaussian metallicity distribution appropriate for
    different Galactic stellar populations (disk, thick disk, halo).

    Parameters
    ----------
    feh : array_like
        Stellar metallicity [Fe/H] in dex.
    feh_mean : float, optional
        Mean metallicity of the population in dex. Default is -0.2 (thin disk).
    feh_sigma : float, optional
        Metallicity dispersion in dex. Default is 0.3.

    Returns
    -------
    logp : array_like
        Normalized log-probability density for the input metallicities.

    Notes
    -----
    The metallicity prior follows a normal distribution:

    .. math::
        \log p([\text{Fe/H}]) = -\frac{1}{2}\left[\frac{([\text{Fe/H}] - \mu_{\text{Fe/H}})^2}{\sigma_{\text{Fe/H}}^2} + \log(2\pi\sigma_{\text{Fe/H}}^2)\right]

    Typical values for different Galactic components:
    - Thin disk: feh_mean = -0.2, feh_sigma = 0.3
    - Thick disk: feh_mean = -0.7, feh_sigma = 0.4
    - Halo: feh_mean = -1.6, feh_sigma = 0.5

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    """
    feh = np.asarray(feh)

    # Gaussian log-prior
    chi2 = (feh - feh_mean) ** 2 / feh_sigma**2
    log_norm = np.log(2.0 * np.pi * feh_sigma**2)
    logp = -0.5 * (chi2 + log_norm)

    return logp


def logp_age_from_feh(
    age,
    feh_mean=-0.2,
    max_age=13.8,
    min_age=0.0,
    feh_age_ctr=-0.5,
    feh_age_scale=0.5,
    nsigma_from_max_age=2.0,
    max_sigma=4.0,
    min_sigma=1.0,
):
    r"""
    Log-prior for stellar age based on metallicity-age relation.

    Implements the age-metallicity relation observed in the Galactic disk,
    where older stars tend to be more metal-poor. Uses truncated normal
    distribution bounded by physically reasonable ages.

    Parameters
    ----------
    age : array_like
        Stellar ages in Gyr.
    feh_mean : float, optional
        Mean metallicity of the population in dex. Default is -0.2.
    max_age : float, optional
        Maximum allowed stellar age in Gyr. Default is 13.8 (age of Universe).
    min_age : float, optional
        Minimum allowed stellar age in Gyr. Default is 0.0.
    feh_age_ctr : float, optional
        Metallicity where mean age is halfway between min/max. Default is -0.5.
    feh_age_scale : float, optional
        Scale length for metallicity-age relation in dex. Default is 0.5.
    nsigma_from_max_age : float, optional
        Number of σ the mean age is below max_age. Default is 2.0.
    max_sigma : float, optional
        Maximum age dispersion in Gyr. Default is 4.0.
    min_sigma : float, optional
        Minimum age dispersion in Gyr. Default is 1.0.

    Returns
    -------
    logp : array_like
        Normalized log-probability density for the input ages.

    Notes
    -----
    The age-metallicity relation follows a logistic function:

    .. math::
        \langle t \rangle = \frac{t_{\max} - t_{\min}}{1 + \exp\left(\frac{[\text{Fe/H}] - c}{s}\right)} + t_{\min}

    where c is the central metallicity and s is the scale length.

    The age dispersion increases for younger (more metal-rich) stars:

    .. math::
        \sigma_t = \min\left(\max\left(\frac{t_{\max} - \langle t \rangle}{n\sigma}, \sigma_{\min}\right), \sigma_{\max}\right)

    The age density is a truncated normal distribution bounded by
    [min_age, max_age].

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    Nordström et al. (2004) - Age-metallicity relation in Solar neighborhood
    """
    age = np.asarray(age)

    # Predicted mean age from metallicity
    age_mean_pred = (max_age - min_age) / (
        1.0 + np.exp((feh_mean - feh_age_ctr) / feh_age_scale)
    ) + min_age

    # Age dispersion (younger stars have larger dispersion)
    age_sigma_pred = (max_age - age_mean_pred) / nsigma_from_max_age
    age_sigma_pred = np.clip(age_sigma_pred, min_sigma, max_sigma)

    # Truncated normal distribution bounds
    a = (min_age - age_mean_pred) / age_sigma_pred  # Lower bound
    b = (max_age - age_mean_pred) / age_sigma_pred  # Upper bound

    # Compute truncated normal log-probability
    logp = truncnorm_logpdf(age, a, b, loc=age_mean_pred, scale=age_sigma_pred)

    return logp


def _normalizations(
    R_solar,
    Z_solar,
    R_thin,
    Rs_thin,
    f_thick_surface,
    feh_thin,
    feh_thin_sigma,
    max_age,
    min_age,
    feh_age_ctr,
    feh_age_scale,
    nsigma_from_max_age,
    max_sigma,
    min_sigma,
    v_circ_solar,
    thick_sigma_z,
    Rs_halo,
    eta_halo_inner,
    eta_halo_outer,
    r_break_halo,
    q_halo,
    halo_mass,
    rho_thin_solar,
):
    """Return the rotation, thick-disk, age and halo-shape arguments of the
    components, the log halo amplitude (ln n_halo minus `_halo_shape`
    relative to the thin disk at the Sun), and the normalizations that
    `galactic_prior_summary` reports."""
    _check_positive(
        f_thick_surface=f_thick_surface,
        feh_thin_sigma=feh_thin_sigma,
        thick_sigma_z=thick_sigma_z,
        Rs_halo=Rs_halo,
        eta_halo_inner=eta_halo_inner,
        eta_halo_outer=eta_halo_outer,
        r_break_halo=r_break_halo,
        q_halo=q_halo,
        halo_mass=halo_mass,
        rho_thin_solar=rho_thin_solar,
    )
    rot = (float(R_solar), _vc_solar(v_circ_solar, R_solar))
    thick = (float(thick_sigma_z) ** 2,)
    age_args = tuple(
        map(
            float,
            (
                max_age,
                min_age,
                feh_age_ctr,
                feh_age_scale,
                nsigma_from_max_age,
                max_sigma,
                min_sigma,
            ),
        )
    )
    h_thin, h_thick, ln_h_thin, ln_h_thick = _disk_columns(
        rot,
        abs(float(Z_solar)),
        (float(R_thin), float(Rs_thin)),
        thick,
        float(feh_thin),
        float(feh_thin_sigma),
        age_args,
    )
    halo = _halo_args(
        R_smooth=Rs_halo,
        eta_inner=eta_halo_inner,
        eta_outer=eta_halo_outer,
        r_break=r_break_halo,
        q=q_halo,
    )
    # Msun/kpc^3 to Msun/pc^3.
    ln_halo_amplitude = log(
        float(halo_mass) / (_halo_volume(halo) * 1e9 * float(rho_thin_solar))
    )
    f_halo = exp(ln_halo_amplitude + _halo_shape(rot[0], float(Z_solar), *halo))
    summary = dict(
        thin_column_height=h_thin,
        thick_column_height=h_thick,
        f_thick_solar=float(f_thick_surface) * h_thin / h_thick,
        halo_density_solar=f_halo * float(rho_thin_solar),
        f_halo_solar=f_halo,
    )
    columns = (ln_h_thin, ln_h_thick)
    return rot, thick, age_args, halo, ln_halo_amplitude, columns, summary


def logp_galactic_structure(
    dists,
    coord,
    labels=None,
    feh=None,
    loga=None,
    R_solar=8.2,
    Z_solar=0.025,
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
    rho_thin_solar=0.030,
    feh_thin=-0.2,
    feh_thin_sigma=0.3,
    feh_thick=-0.7,
    feh_thick_sigma=0.4,
    feh_halo=-1.6,
    feh_halo_sigma=0.5,
    max_age=13.8,
    min_age=0.0,
    feh_age_ctr=-0.5,
    feh_age_scale=0.5,
    nsigma_from_max_age=2.0,
    max_sigma=4.0,
    min_sigma=1.0,
    v_circ_solar=None,
    thick_sigma_z=THICK_DISK_SIGMA[2],
    return_components=False,
):
    r"""
    Complete Galactic structure log-prior with thin disk, thick disk, and halo.

    Implements a sophisticated three-component Galactic model based on
    Bland-Hawthorn & Gerhard (2016). Combines spatial number density priors
    with optional metallicity and age priors for realistic stellar populations.

    Parameters
    ----------
    dists : array_like
        Distance from observer in kpc.
    coord : tuple of floats
        Galactic coordinates (l, b) in degrees.
    labels : structured array, optional
        Stellar labels containing 'feh' and/or 'loga' for metallicity/age
        priors, with one entry per distance.
    feh : array_like, optional
        Per-point [Fe/H] as a plain float array, an alternative to ``labels``
        for the metallicity prior. Lets callers (e.g. ``logpost_grid``) avoid
        tiling a structured ``labels`` array; equivalent to passing the same
        values via ``labels``. Honored on both code paths (fused kernel
        and numpy). If both ``labels`` and
        ``feh``/``loga`` are given, the plain arrays take precedence.
    loga : array_like, optional
        Per-point log10(age/yr) as a plain float array, an alternative to
        ``labels`` for the age prior (see ``feh``).
    R_solar : float, optional
        Solar Galactocentric radius in kpc. Default is 8.2.
    Z_solar : float, optional
        Solar height above midplane in kpc. Default is 0.025.
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
    rho_thin_solar : float, optional
        Thin-disk stellar mass density at the Sun in Msun/pc^3, in living
        stars. Default is 0.030. With ``halo_mass`` it sets the halo to
        thin-disk density ratio at the Sun that `galactic_prior_summary`
        returns.
    feh_thin : float, optional
        Thin disk mean metallicity in dex. Default is -0.2.
    feh_thin_sigma : float, optional
        Thin disk metallicity dispersion in dex. Default is 0.3.
    feh_thick : float, optional
        Thick disk mean metallicity in dex. Default is -0.7.
    feh_thick_sigma : float, optional
        Thick disk metallicity dispersion in dex. Default is 0.4.
    feh_halo : float, optional
        Halo mean metallicity in dex. Default is -1.6.
    feh_halo_sigma : float, optional
        Halo metallicity dispersion in dex. Default is 0.5.
    max_age : float, optional
        Maximum stellar age in Gyr. Default is 13.8.
    min_age : float, optional
        Minimum stellar age in Gyr. Default is 0.0.
    feh_age_ctr : float, optional
        Central metallicity for age-metallicity relation. Default is -0.5.
    feh_age_scale : float, optional
        Scale length for age-metallicity relation. Default is 0.5.
    nsigma_from_max_age : float, optional
        Age dispersion parameter. Default is 2.0.
    max_sigma : float, optional
        Maximum age dispersion in Gyr. Default is 4.0.
    min_sigma : float, optional
        Minimum age dispersion in Gyr. Default is 1.0.
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
        Whether to return individual component contributions. Default is False.

    Returns
    -------
    logp : array_like
        Total log-prior probability density.
    components : dict, optional
        Individual component contributions (if return_components=True).

    Raises
    ------
    ValueError
        If ``labels``, ``feh`` or ``loga`` does not have one finite entry
        per distance, or if a keyword that sets a normalization is not
        positive and finite (see `galactic_prior_summary`).

    Notes
    -----
    The Galactic model combines three stellar populations:

    1. **Thin Disk**: Young, metal-rich stars with small scale height
    2. **Thick Disk**: Intermediate-age, metal-poor stars with larger scale height
    3. **Halo**: Old, very metal-poor stars with a flattened broken power law

    Each disk has an exponential surface density with scale length
    ``R_thin`` or ``R_thick``, as in `logn_disk`; its density at the Sun's
    height is that surface density divided by the column height of its
    vertical profile at the same radius, relative to the Sun. Their vertical
    profiles solve the vertical Jeans equation in one vertical potential,
    that of the Milky Way model of Binney & Vasiliev (2024), tabulated in
    ``brutus.priors.kinematic``:

    .. math::
        \ln \nu(z) = -2 \ln f_z(|z|) - \frac{1}{\sigma_{z,0}^2}
        \int_0^{|z|} \frac{K_z(t)}{f_z(t)^2}\,dt

    The thin disk takes the vertical dispersion of the star's age and
    [Fe/H] (Sharma et al. 2021), which grows with height as
    :math:`f_z = 1 + \gamma_z |z|` up to 2 kpc, the heights of their fit,
    and keeps its value above; younger stars therefore lie closer to the
    midplane. Without an age, the thin-disk profile is the average over
    the thin-disk age prior, by 32-point Gauss-Legendre quadrature on the
    ages where the prior has weight. Without [Fe/H], the dispersion uses
    ``feh_thin``; [Fe/H] outside -1 to 0.5, the range of Sharma et al.
    (2021), takes the edge value. The thick disk is isothermal
    (:math:`\gamma_z = 0`), with ``thick_sigma_z`` at the Sun and the
    radial profile of `~brutus.priors.thick_disk_velocity_moments`.
    Each profile is normalized to 1 at the Sun's height, at every radius,
    so the age and [Fe/H] priors are distributions at the Sun's position.
    The thick disk is normalized by its surface density: at the Sun its
    density is ``f_thick_surface`` times that of the thin disk, times the
    ratio of their column heights (thin over thick). The thin-disk column
    height averages the profile over the thin-disk age and [Fe/H] priors.
    The halo, `logn_halo`, is normalized by ``halo_mass``, as in
    `galactic_prior_summary`.

    Each component has distinct spatial, metallicity, and age distributions
    calibrated from observations. The model accounts for:

    - Coordinate transformations from observer to Galactocentric frame
    - Volume correction factors (dV ∝ distance²)
    - Component membership probabilities
    - Conditional metallicity and age priors

    When stellar labels are provided, applies population-specific priors:
    - Metallicity: Gaussian distributions with different means/dispersions
    - Age: Age-metallicity relation with truncated normal distributions

    When both metallicity and age are supplied, the joint prior is the
    proper mixture ``sum_c w_c(pos) p(feh|c) p(age|c)``: the age term
    weights each component by its metallicity-updated membership
    probability ``w_c(pos) p(feh|c) / sum_c' w_c'(pos) p(feh|c')``, not by
    the position-only weight. This matters for chemically extreme stars
    (e.g. nearby metal-poor stars are correctly weighted toward old,
    halo-like ages).

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    """
    dists = np.asarray(dists, dtype=float)

    # Extract Galactic (l, b) in degrees
    if hasattr(coord, "galactic"):
        # coord is already a SkyCoord object
        ell = coord.galactic.l.deg
        b = coord.galactic.b.deg
    else:
        # coord is a tuple of (l, b) in degrees
        ell, b = coord[0], coord[1]

    n = dists.size
    # Multi-dimensional labels must match `dists` entry by entry, since the
    # fields are flattened below.
    if labels is not None and (
        np.size(labels) != n
        or (max(np.ndim(labels), dists.ndim) > 1 and np.shape(labels) != dists.shape)
    ):
        raise ValueError(
            f"`labels` must have one entry per distance, in the shape of "
            f"`dists` {dists.shape}; got {np.shape(labels)}."
        )

    # Callers may pass the per-point ``feh`` / ``loga`` as plain float arrays
    # instead of a structured ``labels`` array. This is the hot path from
    # logpost_grid, where it avoids tiling a structured array across all MC
    # samples (a structured-dtype ``np.tile`` is ~100x slower than a float
    # tile). A plain array overrides only its own field; any other field comes
    # from ``labels``. Both styles continue as float64 arrays, so
    # the numba kernel types cleanly and the results are bitwise identical.
    if labels is not None:
        if feh is None and "feh" in labels.dtype.names:
            feh = labels["feh"].reshape(-1)
        if loga is None and "loga" in labels.dtype.names:
            loga = labels["loga"].reshape(-1)
    if feh is not None:
        feh = np.asarray(feh, dtype=np.float64)
        if feh.ndim != 1 or feh.shape[0] != n:
            raise ValueError(
                f"`feh` must be a 1D array with one entry per distance "
                f"({n}); got shape {np.shape(feh)}."
            )
        if not np.all(np.isfinite(feh)):
            raise ValueError("`feh` must be finite.")
    if loga is not None:
        loga = np.asarray(loga, dtype=np.float64)
        if loga.ndim != 1 or loga.shape[0] != n:
            raise ValueError(
                f"`loga` must be a 1D array with one entry per distance "
                f"({n}); got shape {np.shape(loga)}."
            )
        if not np.all(np.isfinite(loga)):
            raise ValueError("`loga` must be finite.")

    # Vertical profiles: the rotation curve, the thick disk dispersion and,
    # without an age, the nodes of the thin-disk age average.
    rot, thick, age_args, halo, ln_halo_amplitude, columns, summary = _normalizations(
        R_solar=R_solar,
        Z_solar=Z_solar,
        R_thin=R_thin,
        Rs_thin=Rs_thin,
        f_thick_surface=f_thick_surface,
        feh_thin=feh_thin,
        feh_thin_sigma=feh_thin_sigma,
        max_age=max_age,
        min_age=min_age,
        feh_age_ctr=feh_age_ctr,
        feh_age_scale=feh_age_scale,
        nsigma_from_max_age=nsigma_from_max_age,
        max_sigma=max_sigma,
        min_sigma=min_sigma,
        v_circ_solar=v_circ_solar,
        thick_sigma_z=thick_sigma_z,
        Rs_halo=Rs_halo,
        eta_halo_inner=eta_halo_inner,
        eta_halo_outer=eta_halo_outer,
        r_break_halo=r_break_halo,
        q_halo=q_halo,
        halo_mass=halo_mass,
        rho_thin_solar=rho_thin_solar,
    )
    f_thick = summary["f_thick_solar"]
    if loga is None:
        age_nodes = _thin_age_nodes(float(feh_thin), age_args)
    else:
        age_nodes = (np.empty(0),) * 3

    # Fast path: use fused numba kernel when labels are provided and
    # return_components is not needed. Eliminates ~15 temporary arrays
    # (the coordinate conversion and volume factor are computed per point
    # inside the kernel). Requires a single (scalar) sky position.
    if (
        (labels is not None or feh is not None or loga is not None)
        and not return_components
        and n > 1000
        and dists.ndim == 1
        and np.ndim(ell) == 0
        and np.ndim(b) == 0
    ):
        has_feh = feh is not None
        has_loga = loga is not None
        feh_arr = feh if has_feh else np.empty(0)
        loga_arr = loga if has_loga else np.empty(0)
        ell_rad = np.deg2rad(float(ell))
        b_rad = np.deg2rad(float(b))
        return _galactic_prior_fused(
            dists,
            np.sin(ell_rad),
            np.cos(ell_rad),
            np.sin(b_rad),
            np.cos(b_rad),
            feh_arr,
            loga_arr,
            has_feh,
            has_loga,
            R_solar,
            Z_solar,
            R_thin,
            Rs_thin,
            R_thick,
            Rs_thick,
            f_thick,
            *columns,
            *rot[1:],
            *thick,
            *age_nodes,
            halo,
            ln_halo_amplitude,
            feh_thin,
            feh_thin_sigma,
            feh_thick,
            feh_thick_sigma,
            feh_halo,
            feh_halo_sigma,
            max_age,
            min_age,
            feh_age_ctr,
            feh_age_scale,
            nsigma_from_max_age,
            max_sigma,
            min_sigma,
        )

    # ---- Numpy path (small arrays, return_components, array coordinates) ----

    # Volume correction factor (dV ∝ r² dr); the fused kernel computes this
    # per point, so it is only materialized here.
    vol_factor = 2.0 * np.log(dists + 1e-300)

    # Convert to Galactocentric cylindrical coordinates using fast NumPy math
    R, Z = galactic_to_galactocentric_cyl(
        dists, ell, b, R_solar=R_solar, Z_solar=Z_solar
    )

    # Disk components: the radial part of logn_disk and the vertical
    # Jeans profiles. Array coordinates broadcast R past the shape of
    # `dists`, so the labels broadcast with it.
    def per_point(x):
        return np.broadcast_to(x.reshape(dists.shape), R.shape).ravel()

    thin_z, thick_z = _disk_vertical_array(
        R.ravel(),
        np.abs(Z).ravel(),
        abs(Z_solar),
        np.empty(0) if loga is None else per_point(loga),
        np.full(R.size, feh_thin) if feh is None else per_point(feh),
        loga is not None,
        rot,
        (float(R_thin), float(Rs_thin)),
        *thick,
        *age_nodes,
    )
    # logn_disk at Z = Z_solar is its radial part alone: the surface density,
    # divided by the column height to give the density at Z_solar.
    radial = dict(Z=Z_solar, R_solar=R_solar, Z_solar=Z_solar)
    logp_thin = logn_disk(R, R_scale=R_thin, R_smooth=Rs_thin, **radial)
    logp_thin += thin_z.reshape(R.shape) + vol_factor
    ln_h_thin, ln_h_thick = (
        _ln_column_array(R.ravel() / R_solar, table).reshape(R.shape)
        for table in columns
    )
    logp_thin -= ln_h_thin
    logp_thick = logn_disk(R, R_scale=R_thick, R_smooth=Rs_thick, **radial)
    logp_thick += thick_z.reshape(R.shape) + vol_factor + np.log(f_thick)
    logp_thick -= ln_h_thick

    # Halo component
    logp_halo = _halo_shape(R, Z, *halo) + vol_factor + ln_halo_amplitude

    # Combined number density prior
    logp = _logsumexp3(logp_thin, logp_thick, logp_halo)

    # Component tracking
    components = {"number_density": [logp_thin, logp_thick, logp_halo]}

    # Apply metallicity and age priors if labels provided
    if feh is not None or loga is not None:
        # Component membership probabilities (reuse logp_thin/thick/halo)
        lnprior_thin = logp_thin - logp
        lnprior_thick = logp_thick - logp
        lnprior_halo = logp_halo - logp

        # Membership weights for the age mixture. Position-only by default;
        # updated with the metallicity evidence below when 'feh' is present,
        # so the joint prior is sum_c w_c(pos) p(feh|c) p(age|c) (matching
        # the fused kernel) rather than a product of marginal mixtures.
        age_w_thin = lnprior_thin
        age_w_thick = lnprior_thick
        age_w_halo = lnprior_halo

        if feh is not None:
            feh = feh.reshape(dists.shape)
            feh_lnp_thin = logp_feh(feh, feh_thin, feh_thin_sigma) + lnprior_thin
            feh_lnp_thick = logp_feh(feh, feh_thick, feh_thick_sigma) + lnprior_thick
            feh_lnp_halo = logp_feh(feh, feh_halo, feh_halo_sigma) + lnprior_halo
            feh_lnp = _logsumexp3(feh_lnp_thin, feh_lnp_thick, feh_lnp_halo)
            logp += feh_lnp
            components["feh"] = [feh_lnp_thin, feh_lnp_thick, feh_lnp_halo]

            # Metallicity-updated membership weights for the age mixture
            age_w_thin = feh_lnp_thin - feh_lnp
            age_w_thick = feh_lnp_thick - feh_lnp
            age_w_halo = feh_lnp_halo - feh_lnp

        if loga is not None:
            age = 10.0 ** loga.reshape(dists.shape) / 1e9  # Convert log(age) to Gyr
            age_lnp_thin, age_lnp_thick, age_lnp_halo = (
                logp_age_from_feh(
                    age,
                    feh_mean=fm,
                    max_age=max_age,
                    min_age=min_age,
                    feh_age_ctr=feh_age_ctr,
                    feh_age_scale=feh_age_scale,
                    nsigma_from_max_age=nsigma_from_max_age,
                    max_sigma=max_sigma,
                    min_sigma=min_sigma,
                )
                + lnp_comp
                for fm, lnp_comp in [
                    (feh_thin, age_w_thin),
                    (feh_thick, age_w_thick),
                    (feh_halo, age_w_halo),
                ]
            )
            age_lnp = _logsumexp3(age_lnp_thin, age_lnp_thick, age_lnp_halo)
            logp += age_lnp
            components["age"] = [age_lnp_thin, age_lnp_thick, age_lnp_halo]

    if return_components:
        return logp, components
    else:
        return logp


def galactic_prior_summary(**kwargs):
    r"""
    Return the normalizations that `logp_galactic_structure` derives.

    The column height of a disk at the Sun is
    :math:`h = \Sigma / (2 \nu(Z_\odot))`, and the thick disk's density
    ratio at the Sun is

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
        ``R_solar``, ``Z_solar``, ``R_thin``, ``Rs_thin``,
        ``f_thick_surface``, ``feh_thin``, ``feh_thin_sigma``, the age
        keywords, ``v_circ_solar``, ``thick_sigma_z``, the
        halo-shape keywords, ``halo_mass`` and ``rho_thin_solar``.
        Omitted keywords take its defaults.

    Returns
    -------
    summary : dict
        ``thin_column_height`` and ``thick_column_height`` in kpc (the thin
        disk averaged over its age and [Fe/H] priors) and
        ``f_thick_solar`` (the thick- to thin-disk density ratio at the
        Sun's position), ``halo_density_solar`` in Msun/pc^3 and
        ``f_halo_solar`` (the halo to thin-disk density ratio at the Sun).

    Raises
    ------
    TypeError
        If a keyword does not set a normalization.
    ValueError
        If ``f_thick_surface``, ``feh_thin_sigma``, ``thick_sigma_z``,
        ``v_circ_solar``, ``Rs_halo``,
        ``eta_halo_inner``, ``eta_halo_outer``, ``r_break_halo``,
        ``q_halo``, ``halo_mass`` or ``rho_thin_solar`` is not positive
        and finite.

    Examples
    --------
    >>> from brutus.priors import galactic_prior_summary
    >>> round(galactic_prior_summary()["f_thick_solar"], 3)
    0.054
    >>> round(galactic_prior_summary()["f_halo_solar"], 5)
    0.00181
    """
    unknown = set(kwargs) - set(_NORMALIZATION_KEYS)
    if unknown:
        raise TypeError(
            f"galactic_prior_summary() got keywords that set no "
            f"normalization: {sorted(unknown)}."
        )
    return _normalizations(**{**_NORMALIZATION_DEFAULTS, **kwargs})[-1]


_NORMALIZATION_KEYS = tuple(inspect.signature(_normalizations).parameters)
_NORMALIZATION_DEFAULTS = {
    k: v.default
    for k, v in inspect.signature(logp_galactic_structure).parameters.items()
    if k in _NORMALIZATION_KEYS
}
