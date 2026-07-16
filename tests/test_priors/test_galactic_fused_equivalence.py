#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bitwise-equivalence tests for the optimized `_galactic_prior_fused` kernel.

The production kernel hoists loop-invariant work out of the parallel loop
(age-prior constants: 3 exp + 6 erf + logs per point; solar disk/halo
normalizations) and folds the Galactocentric coordinate conversion into the
loop. These are pure performance rewrites and must be BITWISE identical to
the unhoisted formulation. `reference_kernel` below is a copy of the fused
kernel with the pre-optimization structure (per-point age constants,
precomputed R/Z arrays from `galactic_to_galactocentric_cyl`) but the same
semantics (smoothed solar disk normalization, metallicity-updated age
weights).
"""

from math import erf, exp, log, pi, sqrt

import numpy as np
from numba import jit, prange

from brutus.priors.galactic import logp_galactic_structure
from brutus.utils.math import galactic_to_galactocentric_cyl

_LOG_2PI = log(2.0 * pi)
_SQRT2 = sqrt(2.0)


@jit(nopython=True, parallel=True, cache=False)
def reference_kernel(
    dists,
    R_arr,
    Z_arr,
    feh_arr,
    loga_arr,
    has_feh,
    has_loga,
    R_solar,
    Z_solar,
    R_thin,
    Z_thin,
    Rs_thin,
    R_thick,
    Z_thick,
    Rs_thick,
    f_thick,
    Rs_halo,
    eta_halo,
    q_halo_ctr,
    q_halo_inf,
    r_q_halo,
    f_halo,
    feh_thin_mean,
    feh_thin_sigma,
    feh_thick_mean,
    feh_thick_sigma,
    feh_halo_mean,
    feh_halo_sigma,
    max_age,
    min_age,
    feh_age_ctr,
    feh_age_scale,
    nsigma_from_max_age,
    max_sigma_age,
    min_sigma_age,
):
    """Unhoisted reference: age constants recomputed per point, R/Z arrays
    precomputed outside (the pre-optimization kernel structure)."""
    N = len(dists)
    logp_out = np.empty(N)

    r_solar = sqrt(R_solar**2 + Z_solar**2)
    r_prime_solar = sqrt(r_solar**2 + r_q_halo**2)
    q_solar = q_halo_inf - (q_halo_inf - q_halo_ctr) * exp(
        1.0 - r_prime_solar / r_q_halo
    )
    R_eff_solar_halo = sqrt(R_solar**2 + (Z_solar / q_solar) ** 2 + Rs_halo**2)
    R_solar_eff_thin = sqrt(R_solar**2 + Rs_thin**2)
    R_solar_eff_thick = sqrt(R_solar**2 + Rs_thick**2)

    for i in prange(N):
        d = dists[i]
        R = R_arr[i]
        Z = Z_arr[i]

        vol = 2.0 * log(d + 1e-300)

        R_eff_thin = sqrt(R**2 + Rs_thin**2)
        lnp_thin = (
            -(R_eff_thin - R_solar_eff_thin) / R_thin
            - (abs(Z) - abs(Z_solar)) / Z_thin
            + vol
        )

        R_eff_thick = sqrt(R**2 + Rs_thick**2)
        lnp_thick = (
            -(R_eff_thick - R_solar_eff_thick) / R_thick
            - (abs(Z) - abs(Z_solar)) / Z_thick
            + vol
            + log(f_thick)
        )

        r = sqrt(R**2 + Z**2)
        r_prime = sqrt(r**2 + r_q_halo**2)
        q = q_halo_inf - (q_halo_inf - q_halo_ctr) * exp(1.0 - r_prime / r_q_halo)
        R_eff_halo = sqrt(R**2 + (Z / q) ** 2 + Rs_halo**2)
        lnp_halo = -eta_halo * log(R_eff_halo / R_eff_solar_halo) + vol + log(f_halo)

        mx = max(lnp_thin, max(lnp_thick, lnp_halo))
        logp_total = mx + log(
            exp(lnp_thin - mx) + exp(lnp_thick - mx) + exp(lnp_halo - mx)
        )

        feh_lnp = 0.0
        feh_lnp_thin = 0.0
        feh_lnp_thick = 0.0
        feh_lnp_halo = 0.0
        if has_feh:
            feh_val = feh_arr[i]
            ln_w_thin = lnp_thin - logp_total
            ln_w_thick = lnp_thick - logp_total
            ln_w_halo = lnp_halo - logp_total

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

        if has_loga:
            age_val = 10.0 ** loga_arr[i] / 1e9
            if has_feh:
                ln_w_thin = feh_lnp_thin - feh_lnp
                ln_w_thick = feh_lnp_thick - feh_lnp
                ln_w_halo = feh_lnp_halo - feh_lnp
            else:
                ln_w_thin = lnp_thin - logp_total
                ln_w_thick = lnp_thick - logp_total
                ln_w_halo = lnp_halo - logp_total

            age_lnp_total = -1e300
            for comp_idx in range(3):
                if comp_idx == 0:
                    fm = feh_thin_mean
                    ln_w = ln_w_thin
                elif comp_idx == 1:
                    fm = feh_thick_mean
                    ln_w = ln_w_thick
                else:
                    fm = feh_halo_mean
                    ln_w = ln_w_halo

                # Loop-invariant constants recomputed per point (the
                # pre-optimization formulation)
                age_mean = (max_age - min_age) / (
                    1.0 + exp((fm - feh_age_ctr) / feh_age_scale)
                ) + min_age
                age_sigma = (max_age - age_mean) / nsigma_from_max_age
                if age_sigma < min_sigma_age:
                    age_sigma = min_sigma_age
                if age_sigma > max_sigma_age:
                    age_sigma = max_sigma_age

                xi = (age_val - age_mean) / age_sigma
                alpha = (min_age - age_mean) / age_sigma
                beta = (max_age - age_mean) / age_sigma
                lnphi = -0.5 * (_LOG_2PI + xi * xi)
                denom = max(erf(beta / _SQRT2) - erf(alpha / _SQRT2), 1e-300)
                lndenom = log(age_sigma / 2.0) + log(denom)
                age_comp = lnphi - lndenom + ln_w

                if age_val < min_age or age_val > max_age:
                    age_comp = -1e300

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


DEFAULT_ARGS = dict(
    R_solar=8.2,
    Z_solar=0.025,
    R_thin=2.6,
    Z_thin=0.3,
    Rs_thin=2.0,
    R_thick=2.0,
    Z_thick=0.9,
    Rs_thick=2.0,
    f_thick=0.04,
    Rs_halo=2.0,
    eta_halo=4.2,
    q_halo_ctr=0.2,
    q_halo_inf=0.8,
    r_q_halo=6.0,
    f_halo=0.005,
    feh_thin_mean=-0.2,
    feh_thin_sigma=0.3,
    feh_thick_mean=-0.7,
    feh_thick_sigma=0.4,
    feh_halo_mean=-1.6,
    feh_halo_sigma=0.5,
    max_age=13.8,
    min_age=0.0,
    feh_age_ctr=-0.5,
    feh_age_scale=0.5,
    nsigma_from_max_age=2.0,
    max_sigma_age=4.0,
    min_sigma_age=1.0,
)


def _run_reference(dists, ell, b, feh, loga):
    has_feh = feh is not None
    has_loga = loga is not None
    R, Z = galactic_to_galactocentric_cyl(dists, ell, b, R_solar=8.2, Z_solar=0.025)
    return reference_kernel(
        dists,
        R,
        Z,
        feh if has_feh else np.empty(0),
        loga if has_loga else np.empty(0),
        has_feh,
        has_loga,
        **DEFAULT_ARGS,
    )


class TestFusedKernelBitwiseEquivalence:
    """Production fused path must be bitwise identical to the unhoisted
    reference formulation."""

    ELL, B = 90.0, 30.0

    def _data(self, N=20000, seed=42):
        rng = np.random.default_rng(seed)
        dists = rng.uniform(0.01, 20.0, N)
        feh = rng.normal(-0.5, 0.6, N)
        loga = rng.uniform(7.5, 10.2, N)
        return dists, feh, loga

    def test_bitwise_feh_and_loga(self):
        dists, feh, loga = self._data()
        ref = _run_reference(dists, self.ELL, self.B, feh, loga)
        prod = logp_galactic_structure(dists, (self.ELL, self.B), feh=feh, loga=loga)
        assert np.array_equal(ref, prod)

    def test_bitwise_feh_only(self):
        dists, feh, _ = self._data(seed=7)
        ref = _run_reference(dists, self.ELL, self.B, feh, None)
        prod = logp_galactic_structure(dists, (self.ELL, self.B), feh=feh)
        assert np.array_equal(ref, prod)

    def test_bitwise_loga_only(self):
        dists, _, loga = self._data(seed=8)
        ref = _run_reference(dists, self.ELL, self.B, None, loga)
        prod = logp_galactic_structure(dists, (self.ELL, self.B), loga=loga)
        assert np.array_equal(ref, prod)

    def test_bitwise_other_sightlines(self):
        dists, feh, loga = self._data(N=5000, seed=9)
        for ell, b in [(0.0, 0.0), (180.0, -45.0), (321.5, 12.3)]:
            ref = _run_reference(dists, ell, b, feh, loga)
            prod = logp_galactic_structure(dists, (ell, b), feh=feh, loga=loga)
            assert np.array_equal(ref, prod)

    def test_edge_ages_out_of_bounds(self):
        # loga beyond max_age (13.8 Gyr) and tiny ages exercise the
        # bounds-check branch of the age accumulation
        N = 2000
        dists = np.linspace(0.05, 15.0, N)
        feh = np.full(N, -0.9)
        loga = np.concatenate(
            [np.full(N // 2, 10.5), np.full(N - N // 2, 6.0)]  # 31.6 Gyr, 1 Myr
        )
        ref = _run_reference(dists, self.ELL, self.B, feh, loga)
        prod = logp_galactic_structure(dists, (self.ELL, self.B), feh=feh, loga=loga)
        assert np.array_equal(ref, prod)


class TestFusedVsNumpyPath:
    """The fused kernel and the numpy fallback implement the same prior."""

    COORD = (90.0, 30.0)

    def test_fused_matches_numpy_path(self):
        N = 2000
        rng = np.random.default_rng(5)
        d = rng.uniform(0.05, 10.0, N)
        labels = np.empty(N, dtype=[("feh", float), ("loga", float)])
        labels["feh"] = rng.normal(-0.5, 0.6, N)
        labels["loga"] = rng.uniform(8.0, 10.1, N)

        fused = logp_galactic_structure(d, self.COORD, labels=labels)
        # return_components forces the numpy path
        numpy_logp, _ = logp_galactic_structure(
            d, self.COORD, labels=labels, return_components=True
        )
        assert np.allclose(fused, numpy_logp, rtol=1e-10, atol=1e-12)
