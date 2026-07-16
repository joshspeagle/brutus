#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Photometric offset analysis for systematic calibration corrections.

This module provides robust photometric offset computation for correcting
systematic differences between observed photometry and model predictions.
The implementation uses bootstrap resampling for uncertainty estimation
and supports optional Bayesian prior constraints.

Photometric offsets are multiplicative corrections applied to observed fluxes
to account for systematic calibration differences, photometric system
transformations, or model systematics. The offsets are computed by analyzing
model/data flux ratios across a sample of well-fit objects.

Classes
-------
PhotometricOffsetsConfig : Configuration container
    Encapsulates all configuration parameters with validation

Functions
---------
photometric_offsets : Compute offsets
    Main function for computing multiplicative photometric offsets
_vectorized_bootstrap_median : Bootstrap implementation
    Vectorized bootstrap for performance
_loo_log_weights : Leave-one-out importance weights
    Reweight full-posterior draws to the leave-one-out posterior
_generate_seds : Chunked model SED generation
    Bounds peak memory for large samples
_validate_inputs : Input validation
    Validate input arrays for consistency

See Also
--------
brutus.analysis.individual.BruteForce : Provides fitted parameters for offset computation
brutus.utils.photometry.phot_loglike : Likelihood convention used for reweighting
brutus.core.sed_utils.get_seds : Model SED generation

Notes
-----
The offset computation workflow:

1. **Generate model SEDs** for all fitted objects and posterior samples
2. **Scale by distance**: Apply inverse square law
3. **Compute flux ratios**: model_flux / observed_flux for each band
4. **Reweight samples**: For bands used in fitting, importance-reweight the
   full-posterior draws to the leave-one-out posterior :math:`P(M|D_{-i})`
   using :math:`w \\propto L_{-i}/L_{\\rm full}` to avoid circularity
5. **Bootstrap sampling**: Resample objects and models to estimate median
   offset and uncertainty
6. **Apply priors** (optional): Incorporate Bayesian prior constraints

The key innovation is reweighting step 4: the posterior draws provided as
input target the full-data posterior, so weighting each draw by the
likelihood ratio between the leave-one-out and full-data likelihoods yields
unbiased offset estimates for bands that were used in the original fitting.

Examples
--------
Basic offset computation from BruteForce results:

>>> from brutus.analysis.offsets import photometric_offsets
>>> from brutus.data import load_models
>>> from brutus.core import StarGrid
>>> from brutus.analysis import BruteForce
>>>
>>> # Fit photometry (assuming this has been done)
>>> # fitter = BruteForce(grid)
>>> # results = fitter.fit(phot, err, mask, ...)
>>>
>>> # Extract fitted parameters
>>> # models, idxs, avs, rvs, dists = extract_from_results(results)
>>>
>>> # Compute offsets
>>> offsets, errors, n_used = photometric_offsets(
...     phot, err, mask, models, idxs, avs, rvs, dists
... )
>>>
>>> # Apply corrections
>>> phot_corrected = phot * offsets[None, :]

Advanced usage with configuration:

>>> from brutus.analysis.offsets import PhotometricOffsetsConfig
>>>
>>> # Custom configuration
>>> config = PhotometricOffsetsConfig(
...     min_bands_used=5,
...     n_bootstrap=500,
...     uncertainty_method='bootstrap_std',
...     random_seed=42
... )
>>>
>>> offsets, errors, n_used = photometric_offsets(
...     phot, err, mask, models, idxs, avs, rvs, dists,
...     config=config
... )
"""

from typing import Optional, Tuple

import numpy as np
from scipy.special import xlogy

from ..core.sed_utils import get_seds

__all__ = ["photometric_offsets", "PhotometricOffsetsConfig"]

# Cap on (object, sample) rows per get_seds call. Bounds the transient
# fancy-index copy of `models` and get_seds' internal allocations to
# O(_SED_CHUNK_ROWS * nfilt) instead of O(nobj * nsamps * nfilt).
_SED_CHUNK_ROWS = 500_000


class PhotometricOffsetsConfig:
    """
    Configuration class for photometric offsets computation.

    This class encapsulates all configuration parameters and provides
    sensible defaults with the ability to customize behavior.

    Parameters
    ----------
    min_bands_used : int, optional
        Minimum number of bands, excluding the band being calibrated,
        required for objects where the current band was used in fitting.
        Default is 4 (i.e. at least 5 observed bands in total, matching
        the legacy ``> 3 + 1`` cut).

    min_bands_unused : int, optional
        Minimum number of observed bands required for objects where the
        current band was not used in fitting. Default is 3. (The legacy
        v0.8.3 pipeline effectively required 4; pass ``min_bands_unused=4``
        to reproduce it.)

    n_bootstrap : int, optional
        Number of bootstrap realizations for uncertainty estimation.
        Default is 300.

    uncertainty_method : str, optional
        Method for uncertainty estimation. Options:
        - 'bootstrap_std': Standard deviation of bootstrap medians
        - 'bootstrap_iqr': Scaled interquartile range of bootstrap medians
        Default is 'bootstrap_iqr'.

    progress_interval : int, optional
        Print progress every N iterations. Set to 0 for no progress.
        Default is 10.

    use_vectorized_bootstrap : bool, optional
        Use vectorized bootstrap implementation for better performance.
        Default is True.

    random_seed : int, optional
        Random seed for reproducible results. Default is None.

    validate_inputs : bool, optional
        Perform input validation. Default is True.

    See Also
    --------
    photometric_offsets : Main function using this configuration

    Notes
    -----
    The configuration defaults are chosen to balance statistical robustness
    with computational efficiency:

    - min_bands_used=4: Ensures the leave-one-out reweighting retains at
      least 4 bands (positive degrees of freedom under the default
      dimensionality prior)
    - min_bands_unused=3: Minimum for meaningful photometric constraints
    - n_bootstrap=300: Sufficient for stable uncertainty estimates
    - bootstrap_iqr: More robust to outliers than standard deviation

    Examples
    --------
    >>> config = PhotometricOffsetsConfig(
    ...     min_bands_used=5,
    ...     n_bootstrap=500,
    ...     random_seed=42
    ... )
    >>> offsets, errors, n_used = photometric_offsets(
    ...     phot, err, mask, models, idxs, avs, rvs, dists,
    ...     config=config
    ... )
    """

    def __init__(
        self,
        min_bands_used: int = 4,
        min_bands_unused: int = 3,
        n_bootstrap: int = 300,
        uncertainty_method: str = "bootstrap_iqr",
        progress_interval: int = 10,
        use_vectorized_bootstrap: bool = True,
        random_seed: Optional[int] = None,
        validate_inputs: bool = True,
    ):
        self.min_bands_used = min_bands_used
        self.min_bands_unused = min_bands_unused
        self.n_bootstrap = n_bootstrap
        self.uncertainty_method = uncertainty_method
        self.progress_interval = progress_interval
        self.use_vectorized_bootstrap = use_vectorized_bootstrap
        self.random_seed = random_seed
        self.validate_inputs = validate_inputs

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.min_bands_used < 1:
            raise ValueError("min_bands_used must be >= 1")
        if self.min_bands_unused < 1:
            raise ValueError("min_bands_unused must be >= 1")
        if self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1")
        if self.uncertainty_method not in [
            "bootstrap_std",
            "bootstrap_iqr",
        ]:
            raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method}")
        if self.progress_interval < 0:
            raise ValueError("progress_interval must be >= 0")


def _validate_inputs(
    phot: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    models: np.ndarray,
    idxs: np.ndarray,
    reds: np.ndarray,
    dreds: np.ndarray,
    dists: np.ndarray,
) -> None:
    """Validate input arrays for photometric_offsets.

    Value checks apply only to observed (mask > 0) entries: the canonical
    BruteForce input format uses NaN flux and zero/NaN error as placeholders
    in unobserved bands, and those entries never enter the computation.
    """
    # Check basic types
    arrays = [phot, err, mask, models, idxs, reds, dreds, dists]
    names = ["phot", "err", "mask", "models", "idxs", "reds", "dreds", "dists"]

    for arr, name in zip(arrays, names):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be numpy array, got {type(arr)}")

    # Check shapes
    nobj, nfilt = phot.shape
    nsamps = idxs.shape[1]

    if err.shape != (nobj, nfilt):
        raise ValueError(f"err shape {err.shape} != phot shape {phot.shape}")
    if mask.shape != (nobj, nfilt):
        raise ValueError(f"mask shape {mask.shape} != phot shape {phot.shape}")
    if idxs.shape != (nobj, nsamps):
        raise ValueError(f"idxs shape {idxs.shape} != expected ({nobj}, {nsamps})")
    if reds.shape != (nobj, nsamps):
        raise ValueError(f"reds shape {reds.shape} != expected ({nobj}, {nsamps})")
    if dreds.shape != (nobj, nsamps):
        raise ValueError(f"dreds shape {dreds.shape} != expected ({nobj}, {nsamps})")
    if dists.shape != (nobj, nsamps):
        raise ValueError(f"dists shape {dists.shape} != expected ({nobj}, {nsamps})")

    # Check for valid values in observed entries only (masked entries may
    # legitimately hold NaN/zero placeholders).
    observed = mask > 0
    if not np.all(np.isfinite(phot[observed])):
        raise ValueError("phot contains non-finite values in unmasked bands")
    if not np.all(err[observed] > 0):
        raise ValueError("err must be positive in unmasked bands")
    with np.errstate(invalid="ignore"):
        if not np.all(np.isin(mask[np.isfinite(mask)], [0, 1])) or np.any(
            ~np.isfinite(mask)
        ):
            raise ValueError("mask must contain only 0s and 1s")
    if not np.all(dists > 0):
        raise ValueError("dists must be positive")


def _generate_seds(
    models: np.ndarray,
    idxs: np.ndarray,
    reds: np.ndarray,
    dreds: np.ndarray,
    dists: np.ndarray,
    chunk_rows: int = _SED_CHUNK_ROWS,
) -> np.ndarray:
    """
    Generate distance-scaled model SED fluxes in memory-bounded chunks.

    Chunking only bounds the size of the transient inputs/outputs of each
    get_seds call; the per-element operations are identical to a single
    call, so the result is bitwise identical.

    Parameters
    ----------
    models : np.ndarray of shape (n_models, n_filters, n_coeffs)
        Magnitude polynomial coefficients.
    idxs, reds, dreds, dists : np.ndarray of shape (n_objects, n_samples)
        Model indices, A(V), R(V), and distances (kpc) per posterior draw.
    chunk_rows : int, optional
        Maximum number of (object, sample) rows per get_seds call.

    Returns
    -------
    seds : np.ndarray of shape (n_objects, n_samples, n_filters)
        Model fluxes scaled to the observed distances.
    """
    nobj, nsamps = idxs.shape
    nfilt = models.shape[1]
    seds = np.empty((nobj, nsamps, nfilt), dtype=float)
    block = max(1, chunk_rows // max(nsamps, 1))
    for start in range(0, nobj, block):
        stop = min(start + block, nobj)
        flat = get_seds(
            models[idxs[start:stop].ravel()],
            av=reds[start:stop].ravel(),
            rv=dreds[start:stop].ravel(),
            return_flux=True,
        )
        flat = flat / dists[start:stop].ravel()[:, None] ** 2
        seds[start:stop] = flat.reshape(stop - start, nsamps, nfilt)
    return seds


def _loo_log_weights(
    chi2_full: np.ndarray,
    band_chi2: np.ndarray,
    ndim: np.ndarray,
    dim_prior: bool,
    dof_reduction: int = 1,
) -> np.ndarray:
    r"""
    Importance log-weights reweighting full-posterior draws to the
    leave-one-out (LOO) posterior.

    The posterior draws supplied to `photometric_offsets` target the
    full-data posterior :math:`P(M|D)`, while the offset for a fitted band
    i must be estimated under :math:`P(M|D_{-i})`. The importance weight is
    therefore the likelihood ratio :math:`w = L_{-i}/L_{\rm full}`
    evaluated at each draw — weighting by :math:`L_{-i}` alone would target
    :math:`\propto L_{\rm full} \cdot L_{-i}` and only partially remove the
    circular dependence on band i. Terms constant across draws of the same
    object (Gaussian normalizations, log-determinants, gamma functions)
    cancel under the per-object weight normalization and are omitted.

    Parameters
    ----------
    chi2_full : np.ndarray of shape (n_objects, n_samples)
        Chi-square summed over all observed bands (band i included).
    band_chi2 : np.ndarray of shape (n_objects, n_samples)
        Band i's chi-square contribution (0 where band i is masked).
    ndim : np.ndarray of shape (n_objects,)
        Number of observed bands per object (band i included).
    dim_prior : bool
        If True, likelihoods follow the chi-square dimensional log-PDF
        convention of `phot_loglike(dim_prior=True)`; if False, the
        Gaussian log-likelihood convention.
    dof_reduction : int, optional
        Degrees of freedom subtracted from the effective dimensionality
        (matching the phot_loglike convention used by this module).
        Default is 1.

    Returns
    -------
    lnw : np.ndarray of shape (n_objects, n_samples)
        Unnormalized importance log-weights; -inf where the LOO target is
        undefined (non-positive degrees of freedom).
    """
    chi2_loo = chi2_full - band_chi2
    np.clip(chi2_loo, 0.0, None, out=chi2_loo)  # guard cancellation error
    if dim_prior:
        # ln w = ln chi2pdf(chi2_loo; dof_loo) - ln chi2pdf(chi2_full; dof_full)
        # with per-object gamma/log-2 normalizations dropped.
        dof_full = ndim - dof_reduction
        dof_loo = dof_full - 1  # band i removed from the data
        a_full = 0.5 * dof_full[:, None]
        a_loo = 0.5 * dof_loo[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            lnw = (
                xlogy(a_loo - 1.0, chi2_loo)
                - 0.5 * chi2_loo
                - xlogy(a_full - 1.0, chi2_full)
                + 0.5 * chi2_full
            )
        lnw[dof_loo <= 0] = -np.inf
        lnw[np.isnan(lnw)] = -np.inf
    else:
        # Gaussian likelihoods: ln L_{-i} - ln L_full = +0.5 * chi2_i (+ const).
        lnw = 0.5 * band_chi2
    return lnw


def _vectorized_bootstrap_median(
    ratios: np.ndarray,
    weights: np.ndarray,
    obj_weights: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorized bootstrap implementation for better performance.

    Model draws use per-row inverted-CDF sampling: the weight CDFs are
    computed once and each replicate's draw reduces to a vectorized
    comparison count, equivalent to ``np.searchsorted(cdf_row, u, "left")``
    per row. This samples the same distribution as per-object
    ``rng.choice(nsamps, p=w)`` calls (zero-weight samples are never
    selected) while removing the O(n_bootstrap * n_objects) Python loop;
    the random stream differs from the per-object-choice implementation.

    Parameters
    ----------
    ratios : np.ndarray of shape (n_objects, n_samples)
        Model/data ratios for each object and sample
    weights : np.ndarray of shape (n_objects, n_samples)
        Model weights for each object and sample
    obj_weights : np.ndarray of shape (n_objects,)
        Object selection weights
    n_bootstrap : int
        Number of bootstrap realizations
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    bootstrap_medians : np.ndarray of shape (n_bootstrap,)
        Bootstrap median estimates
    """
    n_objects, n_samples = ratios.shape
    bootstrap_medians = np.zeros(n_bootstrap)

    # Pre-generate object resampling indices
    obj_indices = rng.choice(n_objects, size=(n_bootstrap, n_objects), p=obj_weights)

    # Normalized per-row weight CDFs (computed once). Rows with zero total
    # weight fall back to sample 0, matching the legacy per-object behavior.
    cdf = np.cumsum(weights, axis=1)
    totals = cdf[:, -1].copy()
    valid_rows = totals > 0
    cdf[valid_rows] /= totals[valid_rows, None]  # last entry becomes exactly 1.0

    for i in range(n_bootstrap):
        rows = obj_indices[i]

        # Inverted-CDF draw: index of the first CDF entry >= u.
        u = rng.random(n_objects)
        model_indices = np.sum(cdf[rows] < u[:, None], axis=1)
        model_indices[~valid_rows[rows]] = 0
        np.clip(model_indices, 0, n_samples - 1, out=model_indices)

        bootstrap_medians[i] = np.median(ratios[rows, model_indices])

    return bootstrap_medians


def photometric_offsets(
    phot: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    models: np.ndarray,
    idxs: np.ndarray,
    reds: np.ndarray,
    dreds: np.ndarray,
    dists: np.ndarray,
    sel: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    mask_fit: Optional[np.ndarray] = None,
    old_offsets: Optional[np.ndarray] = None,
    dim_prior: bool = True,
    prior_mean: Optional[np.ndarray] = None,
    prior_std: Optional[np.ndarray] = None,
    verbose: bool = True,
    config: Optional[PhotometricOffsetsConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute multiplicative photometric offsets between data and models.

    This function computes photometric offsets that account for systematic
    differences between observed photometry and model predictions. The offsets
    are computed by comparing model/data flux ratios across a sample of objects,
    with proper uncertainty estimation and optional prior constraints.

    Parameters
    ----------
    phot : np.ndarray of shape (n_objects, n_filters)
        Observed flux densities for all objects. Entries in masked bands
        may be non-finite placeholders; observed entries must be finite,
        and non-positive observed fluxes are excluded per band.

    err : np.ndarray of shape (n_objects, n_filters)
        Associated flux errors for all objects. Must be positive in
        observed bands; masked entries are ignored.

    mask : np.ndarray of shape (n_objects, n_filters)
        Binary mask (0/1, bool or numeric) indicating observed bands for
        each object.

    models : np.ndarray of shape (n_models, n_filters, n_coeffs)
        Magnitude polynomial coefficients for generating reddened photometry.

    idxs : np.ndarray of shape (n_objects, n_samples)
        Model indices fit to each object.

    reds : np.ndarray of shape (n_objects, n_samples)
        A(V) reddening values for each object and sample.

    dreds : np.ndarray of shape (n_objects, n_samples)
        R(V) reddening curve shape values for each object and sample.

    dists : np.ndarray of shape (n_objects, n_samples)
        Distance values (kpc) for each object and sample.

    sel : np.ndarray of shape (n_objects), optional
        Boolean selection of objects to use. Default uses all objects.

    weights : np.ndarray of shape (n_objects, n_samples), optional
        Sample weights for each object. Default uses uniform weights.

    mask_fit : np.ndarray of shape (n_filters), optional
        Boolean mask indicating which filters were used in fitting.
        Default assumes all filters were used.

    old_offsets : np.ndarray of shape (n_filters), optional
        Previous offsets to remove before computing new ones.
        Default is no previous offsets.

    dim_prior : bool, optional
        Whether the likelihoods entering the leave-one-out importance
        ratio follow the chi-square dimensionality-prior convention of
        `phot_loglike(dim_prior=True, dof_reduction=1)` (True) or the
        Gaussian log-likelihood convention (False). Default is True.

    prior_mean : np.ndarray of shape (n_filters), optional
        Gaussian prior means for offsets. Must be provided with prior_std.

    prior_std : np.ndarray of shape (n_filters), optional
        Gaussian prior standard deviations for offsets.

    verbose : bool, optional
        Whether to print progress information. Default is True.

    config : PhotometricOffsetsConfig, optional
        Configuration object with analysis parameters.
        Default uses standard configuration.

    rng : np.random.Generator, optional
        Random number generator for reproducible results.
        Default creates new generator.

    Returns
    -------
    offsets : np.ndarray of shape (n_filters)
        Multiplicative photometric offsets (model/data ratios).

    offset_errors : np.ndarray of shape (n_filters)
        Uncertainties on the photometric offsets.

    n_objects_used : np.ndarray of shape (n_filters)
        Number of objects that actually informed each offset (objects with
        non-positive fluxes or fully degenerate weights are excluded).

    Examples
    --------
    >>> import numpy as np
    >>> from brutus.analysis.offsets import photometric_offsets
    >>>
    >>> # Mock data for demonstration
    >>> n_obj, n_filt, n_samp = 100, 5, 50
    >>> phot = np.random.uniform(0.1, 10, (n_obj, n_filt))
    >>> err = 0.1 * phot
    >>> mask = np.random.choice([0, 1], (n_obj, n_filt), p=[0.1, 0.9])
    >>>
    >>> # Mock fitted parameters
    >>> models = np.random.random((1000, n_filt, 3))
    >>> idxs = np.random.randint(0, 1000, (n_obj, n_samp))
    >>> reds = np.random.uniform(0, 2, (n_obj, n_samp))
    >>> dreds = np.random.uniform(2.5, 4.5, (n_obj, n_samp))
    >>> dists = np.random.uniform(0.1, 10, (n_obj, n_samp))
    >>>
    >>> # Compute offsets
    >>> offsets, errors, n_used = photometric_offsets(
    ...     phot, err, mask, models, idxs, reds, dreds, dists
    ... )
    >>> print(f"Computed offsets: {offsets}")

    See Also
    --------
    PhotometricOffsetsConfig : Configuration options
    brutus.core.sed_utils.get_seds : Model SED generation
    brutus.utils.photometry.phot_loglike : Likelihood computation
    brutus.analysis.individual.BruteForce : Source of fitted parameters

    Notes
    -----
    The photometric offset for each band is computed as:

    1. **Generate model SEDs** for all fitted objects and posterior samples
    2. **Scale by distance**: :math:`F_{\\rm model} = F_0 / d^2`
    3. **Compute flux ratios**: :math:`r = F_{\\rm model} / F_{\\rm obs}`
    4. **Reweight samples**: For bands used in fitting, importance-reweight
       the full-posterior draws to :math:`P(M|D_{-i})` (band i excluded)
       with weights :math:`w \\propto L_{-i}/L_{\\rm full}` to avoid
       circularity
    5. **Bootstrap**: Resample objects and models with weights, compute median
    6. **Uncertainty**: From bootstrap distribution (IQR or std)
    7. **Apply priors** (optional): Bayesian combination with prior

    The reweighting in step 4 is critical: if a band was used in the original
    fit, including it in offset computation would create a circular dependency.
    The input draws target the full posterior :math:`P(M|D)`, so each draw is
    weighted by the likelihood ratio :math:`L_{-i}/L_{\\rm full}` (band i's
    inverse likelihood, up to normalization); weighting by :math:`L_{-i}`
    alone would only partially remove the circularity. Since the chi-square
    is additive over bands, all per-band leave-one-out likelihoods derive
    from a single precomputed pass rather than one full likelihood
    evaluation per band.

    The offsets should be applied as:

    .. math::
        F_{\\rm corrected} = F_{\\rm observed} \\times {\\rm offset}

    For iterative refinement, provide old_offsets from previous iteration.

    References
    ----------
    The bootstrap methodology follows standard non-parametric uncertainty
    estimation. The likelihood-ratio importance reweighting ensures unbiased
    estimates for bands included in the original fit.
    """
    # Handle configuration
    if config is None:
        config = PhotometricOffsetsConfig()

    # Set up random number generator
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    # Validate inputs
    if config.validate_inputs:
        _validate_inputs(phot, err, mask, models, idxs, reds, dreds, dists)

    # Initialize parameters
    nobj, nfilt = phot.shape
    nsamps = idxs.shape[1]

    if sel is None:
        sel = np.ones(nobj, dtype=bool)
    if weights is None:
        weights = np.ones((nobj, nsamps), dtype=float)
    if mask_fit is None:
        mask_fit = np.ones(nfilt, dtype=bool)
    if old_offsets is None:
        old_offsets = np.ones(nfilt)

    # Boolean view of the mask: validation accepts numeric 0/1 masks, but
    # the selection logic below requires bools (float & bool raises).
    mask_bool = np.asarray(mask) > 0

    # Per-filter inputs must be 1-D of length nfilt; a scalar or wrong-length
    # array would otherwise surface much later as an opaque IndexError or
    # broadcasting error.
    mask_fit = np.atleast_1d(np.asarray(mask_fit)).astype(bool)
    if mask_fit.shape != (nfilt,):
        raise ValueError(f"mask_fit must have shape ({nfilt},), got {mask_fit.shape}")
    old_offsets = np.atleast_1d(np.asarray(old_offsets, dtype=float))
    if old_offsets.shape != (nfilt,):
        raise ValueError(
            f"old_offsets must have shape ({nfilt},), got {old_offsets.shape}"
        )
    # Priors are a pair: silently ignoring a lone prior_mean/prior_std would
    # discard information the caller clearly intended to supply.
    if (prior_mean is None) != (prior_std is None):
        raise ValueError("prior_mean and prior_std must be provided together")

    # Generate model SEDs (chunked to bound peak memory)
    if verbose and config.progress_interval > 0:
        print("Generating model SEDs...")

    seds = _generate_seds(models, idxs, reds, dreds, dists)

    # Initialize output arrays
    offsets = np.ones(nfilt)
    offset_errors = np.zeros(nfilt)
    n_objects_used = np.zeros(nfilt, dtype=int)

    ndim = mask_bool.sum(axis=1)
    sample_weight_sums = np.sum(weights, axis=1)

    # One-pass precompute for leave-one-out reweighting: chi-square is
    # additive over bands, so each band's leave-one-out chi-square is the
    # total minus that band's contribution (no per-band likelihood pass).
    # Masked bands contribute 0 (their flux/error placeholders never enter).
    # Only the 2-D running total is stored; each band's 2-D contribution is
    # recomputed on demand in the filter loop below, so no second
    # (nobj, nsamps, nfilt) array is kept alive alongside `seds` (which
    # would double peak memory for large calibration samples).
    if np.any(mask_fit):
        phot_adj = np.where(mask_bool, phot, 0.0) * old_offsets
        var_adj = np.where(mask_bool, err, 1.0) ** 2 * old_offsets**2
        var_safe = np.where(var_adj > 0, var_adj, np.inf)

        def _band_chi2(b, rows=slice(None)):
            """Band b's chi-square contribution, shape (nrows, nsamps)."""
            resid = phot_adj[rows, b][:, None] - seds[rows, :, b]
            np.square(resid, out=resid)
            resid /= var_safe[rows, b][:, None]
            resid *= mask_bool[rows, b][:, None]
            return resid

        chi2_full = np.zeros((nobj, nsamps))
        for b in range(nfilt):
            chi2_full += _band_chi2(b)

    # Process each filter
    for i in range(nfilt):
        if verbose and config.progress_interval > 0:
            print(f"Processing filter {i+1}/{nfilt}...")

        # Select objects with sufficient coverage
        min_bands = config.min_bands_used if mask_fit[i] else config.min_bands_unused

        if mask_fit[i]:
            # Bands available after excluding the current band must reach
            # min_bands (legacy behavior: >= min_bands + 1 bands in total).
            band_counts = ndim - mask_bool[:, i]
        else:
            # Don't exclude current band
            band_counts = ndim

        base_valid = (
            mask_bool[:, i]
            & sel
            & (band_counts >= min_bands)
            & (sample_weight_sums > 0)
        )

        # Ratios divide by the observed flux: zero/negative fluxes (common
        # near the detection limit) would yield inf or sign-flipped
        # "multiplicative" offsets, so they are excluded per band.
        with np.errstate(invalid="ignore"):
            positive_flux = phot[:, i] > 0
        n_nonpositive = int(np.sum(base_valid & ~positive_flux))
        if n_nonpositive > 0 and verbose:
            print(
                f"  Warning: excluded {n_nonpositive} objects with "
                f"non-positive flux for filter {i+1}"
            )
        valid_objects = base_valid & positive_flux

        obj_indices = np.where(valid_objects)[0]
        n = len(obj_indices)

        if n == 0:
            n_objects_used[i] = 0
            if verbose:
                print(f"  Warning: No valid objects for filter {i+1}")
            continue

        # Compute model/data ratios
        ratios = seds[obj_indices, :, i] / phot[obj_indices, None, i]

        # Compute weights (importance-reweight if band was used in fit)
        if mask_fit[i]:
            lnw = _loo_log_weights(
                chi2_full[obj_indices],
                _band_chi2(i, obj_indices),
                ndim[obj_indices],
                dim_prior,
            )
            # Per-object softmax; rows that are entirely -inf collapse to 0.
            lnw_max = np.max(lnw, axis=1, keepdims=True)
            lnw_max = np.where(np.isfinite(lnw_max), lnw_max, 0.0)
            with np.errstate(over="ignore"):
                model_weights = np.exp(lnw - lnw_max)
        else:
            # Use uniform weights
            model_weights = np.ones((n, nsamps))

        # Apply sample weights
        model_weights = model_weights * weights[obj_indices]

        # Objects whose weights collapsed (all zero or non-finite) carry no
        # information about this band; exclude them rather than silently
        # substituting posterior draw 0 with uniform object weight.
        weight_sums = np.sum(model_weights, axis=1)
        good = np.isfinite(weight_sums) & (weight_sums > 0)
        n_bad = n - int(np.sum(good))
        if n_bad > 0:
            if verbose:
                print(
                    f"  Warning: excluded {n_bad} objects with degenerate "
                    f"weights for filter {i+1}"
                )
            ratios = ratios[good]
            model_weights = model_weights[good]
            weight_sums = weight_sums[good]
            n = len(ratios)

        n_objects_used[i] = n
        if n == 0:
            if verbose:
                print(f"  Warning: No informative objects for filter {i+1}")
            continue

        # Normalize weights
        model_weights /= weight_sums[:, None]

        # Object weights for bootstrap (uniform over surviving objects)
        obj_weights = np.full(n, 1.0 / n)

        # Bootstrap uncertainty estimation
        if config.use_vectorized_bootstrap:
            bootstrap_medians = _vectorized_bootstrap_median(
                ratios, model_weights, obj_weights, config.n_bootstrap, rng
            )
        else:
            # Original bootstrap implementation
            bootstrap_medians = []
            for j in range(config.n_bootstrap):
                if (
                    verbose
                    and config.progress_interval > 0
                    and j % config.progress_interval == 0
                ):
                    print(f"  Bootstrap {j+1}/{config.n_bootstrap}")

                # Sample objects
                obj_sample = rng.choice(n, size=n, p=obj_weights)

                # Sample models
                model_sample = np.array(
                    [
                        (
                            rng.choice(nsamps, p=model_weights[k])
                            if np.sum(model_weights[k]) > 0
                            else 0
                        )
                        for k in obj_sample
                    ]
                )

                # Compute median
                sample_ratios = ratios[obj_sample, model_sample]
                bootstrap_medians.append(np.median(sample_ratios))

            bootstrap_medians = np.array(bootstrap_medians)

        # Compute offset and uncertainty
        offsets[i] = np.median(bootstrap_medians)

        if config.uncertainty_method == "bootstrap_std":
            offset_errors[i] = np.std(bootstrap_medians)
        elif config.uncertainty_method == "bootstrap_iqr":
            q25, q75 = np.percentile(bootstrap_medians, [25, 75])
            offset_errors[i] = (q75 - q25) / 1.349  # Convert IQR to std equivalent

    # Apply priors if provided
    if prior_mean is not None and prior_std is not None:
        if len(prior_mean) != nfilt or len(prior_std) != nfilt:
            raise ValueError("Prior arrays must have length n_filters")
        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_std = np.asarray(prior_std, dtype=float)

        # Bands with no informative objects have placeholder 1.0 +/- 0.0
        # estimates that must not act as infinitely precise measurements;
        # with no data, the posterior is the prior.
        estimated = n_objects_used > 0
        var_total = offset_errors**2 + prior_std**2
        combined = (offsets * prior_std**2 + prior_mean * offset_errors**2) / var_total
        combined_err = offset_errors * prior_std / np.sqrt(var_total)
        offsets = np.where(estimated, combined, prior_mean)
        offset_errors = np.where(estimated, combined_err, prior_std)
    else:
        # Without a prior, a band with no informative objects carries no
        # measurement at all: report infinite (not zero) uncertainty so the
        # placeholder offset of 1 cannot masquerade as an infinitely precise
        # estimate downstream.
        offset_errors = np.where(n_objects_used > 0, offset_errors, np.inf)

    if verbose:
        print("Photometric offset computation complete.")

    return offsets, offset_errors, n_objects_used
