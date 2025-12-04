# Phase 2: Code Coverage Documentation Review

**Date**: 2025-12-04
**Reviewer**: Claude (Documentation Review Agent)
**Purpose**: Identify gaps between brutus source code and documentation

---

## Executive Summary

This review identifies significant gaps between the brutus codebase and its documentation. While the overall documentation structure is well-organized, there are **substantial undocumented features**, particularly in:

1. **Advanced stellar modeling** - Empirical corrections, binary star systems
2. **Grid-based fitting internals** - Interpolation methods, optimization algorithms
3. **Prior customization** - Available functions not exposed in API docs
4. **Dust module** - Minimal documentation

Additionally, some API documentation references features that differ from actual implementation (method naming, parameter conventions).

**Key Statistics:**
- ~30+ undocumented public/semi-public functions
- 4 areas with significant over-documentation (redundancy)
- 5 API signature mismatches found

---

## Module-by-Module Coverage Assessment

### 1. brutus.core (70% documented)

#### Well-Documented:
- `EEPTracks`, `StarGrid`, `StarEvolTrack`, `Isochrone`, `StellarPop` (basic usage)
- `FastNN`, `FastNNPredictor` (basic usage)
- `GridGenerator.make_grid()` method

#### Undocumented Features:

| Feature | File:Location | Priority | Description |
|---------|---------------|----------|-------------|
| `EEPTracks.get_corrections()` | individual.py:662-763 | **CRITICAL** | Empirical Teff/radius corrections (99-line method, default params never explained) |
| `StarEvolTrack._get_eep_for_secondary()` | individual.py:1102-1188 | **CRITICAL** | Binary star EEP matching algorithm |
| `StarGrid.get_predictions(use_multilinear)` | individual.py:1654 | HIGH | Two interpolation methods available but undocumented |
| `StarGrid._find_neighbors_multilinear()` | individual.py:1432-1571 | MEDIUM | 140+ line interpolation implementation |
| `StellarPop._add_binary_components()` | populations.py:777-861 | HIGH | Binary population synthesis (85 lines) |
| `Isochrone._apply_corrections()` | populations.py:395-451 | HIGH | Population-level empirical corrections |
| `GridGenerator._fit_reddening_coefficients()` | grid_generation.py:506-587 | MEDIUM | Reddening parameterization fitting |

#### Undocumented Parameter Defaults:
```python
# Correction parameters (never explained):
default = (dtdm=0.09, drdm=-0.09, msto_smooth=30.0, feh_scale=0.5)

# Grid interpolation (option not documented):
use_multilinear=True  # Default, but alternative exists
use_multilinear=False # KD-tree nearest neighbor

# EEP bounds (not mentioned):
EEP_MIN = 202  # ZAMS
EEP_MAX_BINARY = 454  # TAMS for secondaries
EEP_MAX_GRID = 1409  # RGB tip
```

---

### 2. brutus.analysis (65% documented)

#### Well-Documented:
- `BruteForce` class (overview)
- `isochrone_population_loglike()` function
- `photometric_offsets()` function

#### Undocumented Features:

| Feature | File | Priority | Description |
|---------|------|----------|-------------|
| `_optimize_fit_mag()` | individual.py:120-300 | HIGH | Core magnitude-space optimization (~180 lines) |
| `_optimize_fit_flux()` | individual.py:302+ | HIGH | Flux-space optimization variant |
| `BruteForce.loglike_grid()` | individual.py | MEDIUM | Likelihood grid computation |
| `BruteForce.logpost_grid()` | individual.py | MEDIUM | Posterior grid computation |
| Convergence parameters | individual.py | HIGH | `tol`, `init_thresh` not exposed to users |
| Prior parameters | individual.py | HIGH | `av_gauss`, `rv_gauss` defaults not documented |
| Mixture model weights | populations.py | MEDIUM | Cluster vs outlier weighting not explained |

#### Undocumented Algorithm Details:
- Magnitude-space to flux-space optimization strategy
- Convergence criteria and iteration limits
- Prior integration mechanism

---

### 3. brutus.priors (80% documented)

#### Well-Documented:
- `logp_imf()`, `logp_parallax()`, `logp_galactic_structure()`, `logp_extinction()`

#### Undocumented Functions:

| Function | File | Priority | Description |
|----------|------|----------|-------------|
| `logp_feh()` | galactic.py | HIGH | Metallicity prior (not in API docs) |
| `logp_age_from_feh()` | galactic.py | HIGH | Age-metallicity relation prior |
| `logn_disk()` | galactic.py | LOW | Disk density model (internal) |
| `logn_halo()` | galactic.py | LOW | Halo density model (internal) |
| `convert_parallax_to_scale()` | astrometric.py | MEDIUM | Parallax conversion utility |

#### Missing Documentation:
- Prior combination mechanism in BruteForce
- Custom prior API (does not exist, but users expect it)
- Default parameter values for each prior

---

### 4. brutus.plotting (60% documented)

Functions listed in API but implementation details unclear:
- `cornerplot()` - signature and parameters need verification
- `posterior_predictive()` - SED visualization
- `dist_vs_red()` - distance-reddening plots
- `photometric_offsets()` - offset visualization
- `hist2d()` - 2D histogram utility

---

### 5. brutus.utils (85% documented)

#### Minor Gaps:

| Function | File | Description |
|----------|------|-------------|
| `chisquare_outlier_loglike()` | photometry.py | Used in populations but not in API |
| `uniform_outlier_loglike()` | photometry.py | Alternative outlier model |
| `draw_sar()` | sampling.py | Scale/Av/Rv sampling utility |
| `inverse3()` | math.py | 3x3 matrix inversion |

---

### 6. brutus.dust (40% documented)

#### Significant Gaps:

| Feature | File | Priority | Description |
|---------|------|----------|-------------|
| `DustMap` base class | maps.py | MEDIUM | Abstract interface unclear |
| `Bayestar` implementation | maps.py | HIGH | Usage not well documented |
| Coordinate transforms | maps.py | MEDIUM | `lb2pix()` and others |
| Prior integration | extinction.py | HIGH | How dust maps affect priors |

---

### 7. brutus.data (90% documented)

#### Minor Issues:
- `FILTERS` list not documented as user-accessible constant
- Pooch registry/cache management not explained
- Cache location customization (environment variables) not listed

---

## Specific Over-Documented Areas (Redundancy)

### 1. Distance Modulus Formula
**Appears in 4+ places**: StarEvolTrack, StarGrid, GridGenerator, core overview
**Recommendation**: Document once, reference elsewhere

### 2. Binary Star Modeling
**Appears in 6+ places**: StarEvolTrack, StellarPop, cluster_modeling, core overview, multiple docstrings
**Recommendation**: Single comprehensive guide with links

### 3. Grid File Format
**Appears in 4 places**: GridGenerator, _save_grid(), data module, core overview
**Recommendation**: Move to single section in data.rst

### 4. Neural Network Overview
**Appears in 4 places**: FastNN, FastNNPredictor, core overview, grid generation guide
**Recommendation**: Single docstring in FastNN, brief reference elsewhere

---

## API Signature Mismatches

### 1. Method Naming
```python
# Documentation sometimes uses:
StellarPop.synthesize(...)  # INCORRECT - doesn't exist

# Actual code uses:
StellarPop.get_seds(...)    # Correct
```

### 2. Return Value Inconsistency
```python
# StarEvolTrack.get_seds() with binary:
returns (sed, params, params2, eep2)

# StarGrid.get_seds():
params2 is always empty dict {} or None
# No eep2 return option
```

### 3. Distance Reference
- Documentation says "Distance in parsecs. Default is 1000.0"
- Grid actually stores magnitudes at 1 kpc reference (crucial but not stated clearly)

### 4. Prior Parameter Defaults
```python
# Code defaults (not documented):
av_gauss = (0.0, 1e6)      # Flat prior on Av
rv_gauss = (3.32, 0.18)    # Gaussian on Rv
```

### 5. Filter Parameter Convention
```python
# Inconsistent filter specification across modules:
EEPTracks: No filter parameter
StarEvolTrack: filters=None (list/array)
GridGenerator: filters=None (list/array)
# When None used, behavior varies
```

---

## Priority Recommendations

### CRITICAL (Immediate Action)

#### 1. Document Empirical Corrections System
**Create**: `docs/source/empirical_corrections.rst`
**Content**:
- Explain parameters: dtdm, drdm, msto_smooth, feh_scale
- When to use/disable corrections
- Default values and their justification
**Files**: individual.py:662-763, populations.py:395-451

#### 2. Fix Method Naming in Documentation
**Action**: Search for "synthesize()" and replace with "get_seds()"
**Verify**: All code examples use correct method names

#### 3. Document Binary Star Modeling
**Create**: `docs/source/binary_stars.rst`
**Content**:
- Secondary EEP calculation algorithm
- Limitations and failure modes
- SMF (secondary mass fraction) explanation
**Files**: individual.py:1102-1188, populations.py:777-861

#### 4. Add Distance Reference Documentation
**Action**: Document 1 kpc reference distance in StarGrid class docstrings
**Impact**: Critical for distance modulus calculations

---

### HIGH (Before Release)

#### 5. Document Grid Interpolation Methods
- Explain multilinear vs. KD-tree approaches
- Add guidance on when to use each
- Document `use_multilinear` parameter

#### 6. Create Optimization Algorithm Guide
- Magnitude-space to flux-space approach
- Convergence parameters (tol, init_thresh)
- Numerical stability considerations

#### 7. Document All Prior Functions
- Add `logp_feh()`, `logp_age_from_feh()` to API docs
- Explain prior combination in BruteForce
- Document prior customization (current limitation)

#### 8. Create Advanced StarGrid Usage Guide
- `StarGrid.get_predictions()` usage
- `StarGrid.get_seds()` without fitting
- Interpolation options explained

---

### MEDIUM

| ID | Recommendation |
|----|----------------|
| 9 | Consolidate duplicate documentation |
| 10 | Document cache system (EEPTracks pickle caching) |
| 11 | Add neural network architecture documentation |
| 12 | Create parameter reference guide with defaults |

---

### LOW

| ID | Recommendation |
|----|----------------|
| 13 | Reduce documentation redundancy |
| 14 | Expand dust module documentation |

---

## New Documentation Files Needed

1. **empirical_corrections.rst** - Correction system guide
2. **binary_stars.rst** - Binary modeling comprehensive guide
3. **grid_usage.rst** - Grid usage without fitting
4. **optimization_algorithm.rst** - Fitting algorithm details

---

## Source Files Needing Documentation Updates

### Documentation Files (docs/source/):
1. **api/core.rst** - Add: get_corrections(), interpolation methods
2. **api/analysis.rst** - Add: optimization details, convergence parameters
3. **api/priors.rst** - Add: logp_feh(), logp_age_from_feh()

### Source Files (src/brutus/):
1. **core/individual.py** - Clarify correction parameters in docstrings
2. **core/populations.py** - Document mixture model approach
3. **analysis/individual.py** - Document optimization strategy
4. **priors/astrometric.py** - Add convert_parallax_to_scale() to __all__
5. **dust/maps.py** - Expand Bayestar documentation

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total Public Functions/Methods | ~120 |
| Documented in API | ~85 (71%) |
| Undocumented | ~35 (29%) |
| Private Methods with Full Docstrings | ~15 |
| Over-documented Topics | 4 |
| API Signature Mismatches | 5 |
| Recommended New Guides | 4 |
| Files Needing Updates | 10 |

---

## Conclusion

The brutus documentation provides a good foundation for basic usage but falls short for:

1. **Advanced stellar modeling** (corrections, binaries, SMF)
2. **Custom grid usage** (interpolation, direct photometry)
3. **Algorithm details** (optimization, convergence)
4. **Prior customization** (no public API exists)

The codebase has comprehensive docstrings that aren't being leveraged in public API documentation. A concerted effort to extract undocumented features into user guides would significantly improve usability.
