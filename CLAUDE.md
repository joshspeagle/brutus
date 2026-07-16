# CLAUDE.md - Repository Guide for brutus

## Overview

`brutus` is a Python package for Bayesian inference of stellar properties, distances, and extinctions from photometry. It supports individual star fitting, cluster analysis, and 3D dust mapping.

**Package name on PyPI**: `astro-brutus`
**Version**: 1.0.0
**Paper**: Speagle et al. (2025), [arXiv:2503.02227](https://arxiv.org/abs/2503.02227)

## Repository Structure

```
brutus/
├── src/brutus/           # Main package source
│   ├── core/             # Stellar modeling (EEPTracks, StarGrid, Isochrone, StellarPop)
│   ├── analysis/         # Fitting (BruteForce, population modeling, photometric offsets)
│   ├── data/             # Data downloading/loading (Pooch-based, Harvard Dataverse)
│   ├── dust/             # 3D dust map utilities
│   ├── priors/           # Bayesian priors (IMF, Galactic structure, extinction)
│   ├── plotting/         # Visualization utilities
│   └── utils/            # Math, photometry, sampling utilities
├── tests/                # Test suite (pytest)
├── tutorials/            # 12 Jupyter tutorial notebooks (00-11)
├── docs/source/          # Sphinx documentation
├── pyproject.toml        # Package configuration
└── CHANGELOG.md          # Version history
```

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
# NUMBA_DISABLE_JIT=1 is REQUIRED: @jit functions show 0% coverage otherwise
NUMBA_DISABLE_JIT=1 pytest --cov=brutus

# Build documentation
cd docs && make html
```

Pre-commit hooks run `black`, `isort`, and `flake8`. If a commit fails on formatting, re-stage the auto-formatted files and commit again.

## Testing

Tests are in `tests/`, organized by module (`test_core/`, `test_analysis/`, etc.).

```bash
pytest tests/test_core/
pytest tests/test_analysis/test_individual.py -v
```

**Important**: When making code changes, flag any tests that may need updating. Check for:
- Tests that directly test modified functions
- Tests that use modified functions as dependencies
- Integration tests that exercise modified code paths

## Documentation

Documentation lives in `docs/source/`:
- **Getting Started**: `installation.rst`, `quickstart.rst`, `glossary.rst`
- **Scientific Background**: `scientific_background.rst`, `stellar_models.rst`, `priors.rst`, `grid_generation.rst`, `population_modeling.rst`, `photometric_offsets.rst`
- **User Guide**: `understanding_results.rst`, `faq.rst`
- **API Reference**: `api/` directory with per-module documentation
- **Development**: `changelog.rst`, `contributing.rst`

Build with `cd docs && make html`. Output in `docs/build/html/`.

## Tutorials

12 Jupyter notebooks in `tutorials/` covering individual stars, populations, grids, priors, fitting, cluster analysis, dust mapping, and photometric calibration. These use real data files (FITS/HDF5) and require downloaded model data.

## Code Style

- `black` for formatting (line length 88)
- `isort` for import sorting
- NumPy-style docstrings

## Unit Conventions and Normalization

Understanding brutus's internal conventions is essential for correct usage:

- **Flux**: "maggies" = `10^(-0.4 * m)` where `m` is the magnitude in native survey units. No absolute zeropoint offset — the conversion is purely relative.
- **Model magnitudes**: Defined at a reference distance of **1 kpc**. Apparent magnitude at distance `d` (kpc) is `m_apparent = m_model + 5 * log10(d_kpc)`.
- **Scale factor**: `scale = (d_ref / d)^2 = 1 / d_kpc^2`. Used internally to shift model flux to the observed distance.
- **Parallax**: Expected in **milliarcseconds** (mas) throughout the fitting pipeline. Survey data in arcseconds must be multiplied by 1000.
- **Extinction**: A(V) in magnitudes. The default R(V) prior is Gaussian with mean 3.32 and std 0.18.
- **Reddening vectors**: In `_get_sed_mle`, `drvecs = (1/A_V) * ∂f/∂R_V` (the A(V) factor is divided out). The Fisher information computation multiplies by A(V) to recover the true derivative `∂f/∂R_V` for correct R(V) uncertainty estimation.
- **Photometric offsets**: Multiplicative flux corrections loaded via `load_offsets()`. Applied as `flux *= offset, err *= offset` in `_setup()`. The offset uncertainties (~0.02 mag optical, ~0.03 mag NIR per Speagle et al. 2025 Table 5) are NOT automatically added to measurement errors. Users should add systematic errors in quadrature before fitting:
  ```python
  sys_err_mag = np.array([0.02, 0.02, ...])  # per filter
  sys_err_flux = flux * sys_err_mag * np.log(10) / 2.5
  err = np.sqrt(err**2 + sys_err_flux**2)
  ```

## Architecture: Grid-Based Bayesian Inference

brutus uses **systematic grid evaluation** rather than MCMC:
- Evaluates likelihood at every grid point — no convergence diagnostics needed
- For each grid point, optimizes (scale, A_V, R_V) via Gauss-Newton and computes a 3x3 Fisher information matrix
- MC samples from the Fisher-based Gaussian approximate the local posterior
- Galactic structure priors, parallax constraints, and IMF are applied as weights
- Final posterior is a weighted mixture over all grid points

The 3x3 Fisher matrix is in (scale, A_V, R_V) space. Internally, `logpost_grid` transforms to **log-distance space** (eta = ln(d)) for MC sampling, applying a Jacobian correction exp(eta) to weights. This prevents the 1/d^3 bias that occurs when sampling in scale space.

The MC prior integral uses **antithetic sampling** (proposal normals drawn in `(z, -z)` pairs via `sample_multivariate_normal(..., antithetic=True)`). The integrand is monotone along `ln(d)` (galactic-density falloff + `log(d)` Jacobian), the regime where antithetic variates cut variance — measured ~20x lower MC-integration variance and lower finite-`Nmc` Jensen bias, unbiased, at half the Gaussian draws. Consequence: the saved `mc_ess` (`1 / sum(w_i^2)`) is a **weight-concentration diagnostic**, not a strict independent-sample count (the antithetic pairs are correlated).

## Internal Constants

- **`MIN_SCALE`** = `1e-20`: Floor for scale factor to prevent log-underflow. Defined in `analysis/individual.py`.
- **`LOG_ZERO`** = `-1e300`: Pseudo-negative-infinity for log-probabilities. Avoids true `-np.inf` arithmetic issues.

## Numba JIT

Performance-critical functions use `@jit(nopython=True, cache=True)`, and the
per-grid-point loops additionally use `parallel=True` with `prange` (each grid
row is independent, so parallel execution is bitwise-identical):
- `_optimize_fit_mag/flux`, `_get_sed_mle`, `_chi2_from_resid`, `_init_mag_resid`
  in `analysis/individual.py`
- `_get_seds` in `core/sed_utils.py`
- `_galactic_prior_fused` in `priors/galactic.py`
- `_batch_min_eig_sym3` and the matrix operations in `utils/math.py`
- `_sample_multivariate_normal_jit` in `utils/sampling.py`

`NUMBA_DISABLE_JIT=1` forces pure-Python fallback for coverage and debugging.
Note: profile and run the test suite with JIT **enabled** to exercise the real
parallel codegen (the coverage workflow disables JIT and would not).

## Plotting Module

- **`corner.py`**: `cornerplot()` — parameter posterior triangle plot
- **`sed.py`**: `posterior_predictive()` — SED residual violin plots
- **`summary.py`**: `summary_plot()` — corner plot + SED inset in one figure
- **`distance.py`**: `dist_vs_red()` — distance vs. reddening 2D posterior
- **`binning.py`**: `bin_pdfs_distred()` — bin PDFs for dust mapping
- **`offsets.py`**: `photometric_offsets()` — offset diagnostic plots
- **`utils.py`**: `hist2d()` — 2D histogram/contour utility

All importable from `brutus.plotting` directly.

## Key APIs

### Input Data Format

brutus expects **linear flux densities**, not magnitudes:
```python
from brutus.utils import inv_magnitude
flux, flux_err = inv_magnitude(mag, mag_err)
```

### Loading Models

```python
from brutus.data import load_models
from brutus.core import StarGrid

models, labels, label_mask = load_models('grid_mist_v9.h5', filters=filters)
grid = StarGrid(models, labels)
```

### Fitting

```python
from brutus.analysis import BruteForce

fitter = BruteForce(grid)
fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask,
    data_labels=obj_ids, save_file='results.h5',
    parallax=parallax, parallax_err=parallax_err,
    data_coords=coords,
    # Performance/accuracy tuning (all optional with sensible defaults):
    max_models=50000,          # Subsample models when Nsel exceeds this
    precision_shrinkage=0.0,   # Shrink off-diagonal precision (try 0.03)
    R_solar=8.2, Z_solar=0.025,  # Solar position for galactic priors
    Ndraws=250,                # Posterior samples per object
    Nmc_prior=50,              # MC samples for prior integration
)
```

### Populations

```python
from brutus.core import Isochrone, StellarPop

iso = Isochrone()
pop = StellarPop(iso, filters=filters)
seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, av=0.1, dist=1000.0)
```

## Fit Quality Metrics

**Do NOT use chi2/Nbands as a quality metric.** It is not variance-stabilizing across different numbers of bands and conflates DOF with band count.

Instead, use **p-values** from the chi-squared distribution with proper degrees of freedom:
```python
from scipy import stats
dof = nbands - 3  # 3 free parameters: scale, A_V, R_V
pvalue = 1.0 - stats.chi2.cdf(chi2min, max(dof, 1))
```

Thresholds:
- `p > 0.001`: Good fit
- `1e-6 < p <= 0.001`: Marginal
- `p <= 1e-6`: Poor fit (model inadequacy, YSOs, binaries, etc.)

The `obj_chi2min` field in HDF5 output includes the parallax contribution (if parallax was provided), and `obj_Nbands` counts parallax as an extra band. So DOF = `obj_Nbands - 3` when parallax is used, `obj_Nbands - 3` otherwise (parallax adds 1 band and 0 free parameters).

For posterior predictive validation, compare the **posterior predictive width** (std of predicted flux across draws) to the **measurement error** per band. The ratio should be ~1; ratios >> 1 indicate poorly constrained parameters.

## Common Pitfalls

- **`load_models` default labels include `afe`**: grids are generated on a 5D
  (mini, eep, feh, afe, smf) lattice; single-afe grids simply carry one
  constant label column.
- **Bayestar reliability masking is on by default**: queries return NaN
  outside each sightline's `DM_reliable_min/max` range and in non-converged
  pixels, degrading the dust prior to uniform there. Pass
  `Bayestar(apply_reliability_mask=False)` for raw map values.
- **EEPTracks caches are versioned** (`*_cachev2.pkl`); unversioned `.pkl`
  caches from older versions are ignored and can be deleted.
- **`quantile` uses the midpoint-CDF convention** for both weighted and
  unweighted samples (not `np.percentile`'s linear interpolation).

- **`np.tile` vs `np.repeat` for label alignment**: When broadcasting labels across MC samples, use `np.tile(labels, Nmc)` (repeats entire array), NOT `np.repeat(labels, Nmc)` (repeats each element). Wrong order silently scrambles label-to-sample mapping.
- **ar_mix cross-term**: The A_V-R_V Fisher cross-term must include the `av[i]` factor to un-normalize `drvecs`. Missing this makes the precision matrix singular.
- **Scale factor guard**: Always clamp scale to `MIN_SCALE` before taking `log(scale)`. Without this, NaN propagates through distance priors.
- **Minimum 4 bands**: Fitting requires at least 4 valid photometric bands (3 free parameters + 1 DOF). Fewer bands produce degenerate Fisher matrices.
- **Diagonal preconditioning**: The 3x3 precision matrix inversion uses diagonal preconditioning (normalize to correlation matrix, invert, un-normalize). This reduces condition numbers from ~30,000 to ~14.
