# Brutus Documentation Audit

This file tracks issues found during the documentation audit and serves as a checklist for systematic revision.

**Created**: 2025-12-03
**Paper Reference**: Speagle et al. (2025), arXiv:2503.02227
**Local paper copy**: `/tmp/brutus-reference/brutus_paper.txt`

---

## Audit Status

| File | Audited | Issues Found | Revised |
|------|---------|--------------|---------|
| `index.rst` | [x] | 5 | [x] Stage 2 |
| `installation.rst` | [x] | 3 | [x] Stage 2 |
| `quickstart.rst` | [x] | 7 | [x] Stage 1+2 |
| `scientific_background.rst` | [x] | 2 | [x] Stage 3 |
| `stellar_models.rst` | [x] | 5 | [x] Stage 3 |
| `priors.rst` | [x] | 8 | [x] Stage 3 (59% reduction: 449→183) |
| `grid_generation.rst` | [x] | 6 | [x] Stage 4 (60% reduction: 517→207) |
| `cluster_modeling.rst` | [x] | 5 | [x] Stage 4 (59% reduction: 488→202) |
| `understanding_results.rst` | [x] | 10 | [x] Stage 4 (62% reduction: 594→226) |
| `choosing_options.rst` | [x] | 12 | [x] Stage 4 (54% reduction: 619→282) |
| `faq.rst` | [x] | 12 | [x] Stage 5 (72% reduction: 701→198) |
| `api/core.rst` | [x] | 3 | [x] Stage 6 |
| `api/analysis.rst` | [x] | 4 | [x] Stage 6 |
| `api/data.rst` | [x] | 1 | [x] (autodoc) |
| `api/priors.rst` | [x] | 1 | [x] (autodoc) |
| `api/plotting.rst` | [x] | 1 | [x] (autodoc) |
| `api/utils.rst` | [x] | 1 | [x] (autodoc) |
| `api/dust.rst` | [x] | 1 | [x] (autodoc) |

---

## Issue Categories

- **[IMPL]** - Implementation mismatch (documented feature doesn't exist or works differently)
- **[VERB]** - Excessive verbosity (can be significantly condensed)
- **[DUP]** - Duplicate content (same info repeated elsewhere)
- **[ERR]** - Factual error or misleading claim
- **[MISSING]** - Missing important information
- **[EXAMPLE]** - Code example needs verification/fixing
- **[ORG]** - Organizational issue (wrong location, poor structure)

---

## Detailed Issue Log

### `index.rst`

**Line count**: 116 lines (reasonable length)

1. **[EXAMPLE] L79-91**: Quick Start individual star example - `star.get_seds()` call needs verification of exact parameter names (does it use `afe` or just the 4 shown?)
2. **[EXAMPLE] L95-108**: StellarPop example uses `pop.get_seds()` - actual implementation may use `pop.synthesize()` based on `__init__.py` docstring
3. **[MISSING]**: No mention of data requirements (need to download grids/isos first)
4. **[ORG]**: References `photometric_offsets.rst` in toctree but need to verify this file exists
5. **[MINOR]**: Uses emoji (🌟) in features list - acceptable but noted for style consistency

**Recommendation**: Mostly okay, verify examples run, add note about data downloads.

---

### `installation.rst`

**Line count**: 77 lines (concise, good)

1. **[IMPL] L48**: `conda install -c conda-forge astro-brutus` - UNVERIFIED if package exists on conda-forge. Need to check or remove.
2. **[EXAMPLE] L75-76**: Testing example `from brutus import Isochrone, load_models` - works, but `Isochrone()` may fail if data files not downloaded first (no warning about this)
3. **[MISSING]**: No mention of disk space requirements for data files (grids can be several GB)

**Recommendation**: Verify conda-forge availability; add note about data file requirements.

---

### `quickstart.rst`

**Line count**: 104 lines (concise)

1. **[IMPL] L59-69**: `fetch_grids()`, `fetch_isos()`, `fetch_dustmaps()` shown with no arguments - actual signature is `fetch_*(target_dir=".", ...)`. Example will work but may confuse users about where files go.
2. **[IMPL] L74-85**: "Working with Results" section shows:
   - `results['dist_samples']` - UNVERIFIED output key name
   - `results['av_samples']` - UNVERIFIED output key name
   - `results['stellar_params']` - UNVERIFIED output key name
   - Need to check actual `BruteForce.fit()` output format (saves to HDF5 file, not dict)
3. **[IMPL] L85**: `cornerplot(results, show_titles=True)` - WRONG SIGNATURE. Actual function is `cornerplot(idxs, data, params, ...)` - completely different interface!
4. **[MISSING] L32**: "For large-scale fitting with pre-computed grids" - links to tutorials/api but doesn't show the actual `BruteForce` workflow which is the main use case
5. **[EXAMPLE] L46-51**: `iso.get_predictions()` - method exists but parameters may differ (`feh`, `afe`, `loga` are correct)
6. **[ORG] L87-96**: "Common Workflows" section is too vague to be useful - doesn't actually show the workflow
7. **[MISSING]**: No mention that `BruteForce.fit()` saves directly to HDF5 file (doesn't return a dict)

**Recommendation**: Major revision needed. The "Working with Results" section is fundamentally wrong about the API. Need to show actual BruteForce workflow.

---

### `scientific_background.rst`

**Line count**: 189 lines (reasonable, well-organized)

1. **[ORG]**: Generally well-written and accurate. Good alignment with paper §2.
2. **[VERB] L131-147**: "Why Bayesian Inference?" section could be condensed - lists advantages that are somewhat obvious for the target audience

**Recommendation**: Minor cleanup only. This is one of the better-written docs. Consider slightly condensing the "Why Bayesian" section.

---

### `stellar_models.rst`

**Line count**: 335 lines (lengthy but covers complex topic)

1. **[EXAMPLE] L81**: `tracks.get_predictions([1.0, 454, 0.0, 0.0])` - need to verify exact parameter format (list vs individual args)
2. **[EXAMPLE] L207-211**: `FastNNPredictor` example - need to verify `filters` parameter and `predict()` method signature
3. **[IMPL] L261**: `load_models('grid_mist_v9.h5')` - file name needs verification against actual available files
4. **[VERB] L159-228**: Extinction and neural network sections are quite detailed - could be condensed
5. **[MISSING]**: No mention of what happens if required data files are not downloaded

**Recommendation**: Verify code examples; consider condensing technical implementation details.

---

### `priors.rst`

**Line count**: 429 lines (VERY VERBOSE - longest conceptual doc)

1. **[IMPL] L56**: References `brutus.priors.stellar.logp_imf` - actual location is `brutus.priors.logp_imf` (no `stellar` submodule exported)
2. **[IMPL] L124**: References `brutus.priors.galactic.logp_galactic_structure` - actual is `brutus.priors.logp_galactic_structure`
3. **[IMPL] L172**: References `brutus.priors.galactic.logp_metallicity` - DOES NOT EXIST. Actual function is `logp_feh`
4. **[IMPL] L208**: References `brutus.priors.galactic.logp_age` - DOES NOT EXIST. Actual function is `logp_age_from_feh`
5. **[IMPL] L239]: References `brutus.priors.extinction.logp_extinction` and `brutus.dust.maps.get_dust_prior` - need verification
6. **[IMPL] L311-314**: `BruteForce(grid, use_galactic_prior=False)` and `use_dust_prior=False` - THESE PARAMETERS DO NOT EXIST!
7. **[IMPL] L349-360**: Code example references `results['dist_samples']` which is wrong (see quickstart.rst issues)
8. **[VERB]**: Overall document is excessively verbose - 430 lines for priors is too much. Much of this repeats scientific_background.rst.

**Recommendation**: MAJOR REVISION NEEDED. Fix incorrect function references. Remove non-existent parameters. Condense significantly (target: ~200 lines). Remove duplication with scientific_background.rst.

---

### `grid_generation.rst`

**Line count**: ~400 lines (verbose)

1. **[EXAMPLE]**: `generator.make_grid()` - need to verify exact method signature and parameters
2. **[IMPL]**: Grid file names referenced (`grid_mist_v9.h5`, etc.) need verification
3. **[VERB]**: Lengthy explanations of grid concepts that overlap with stellar_models.rst
4. **[IMPL]**: `GridGenerator` class interface needs verification
5. **[EXAMPLE]**: Memory-mapped loading (`memmap=True` in `load_models`) - needs verification
6. **[DUP]**: Significant overlap with stellar_models.rst "Grid Pre-computation vs On-the-Fly" section

**Recommendation**: Verify GridGenerator API; condense and remove duplication with stellar_models.rst.

---

### `cluster_modeling.rst`

**Line count**: ~500 lines (VERY VERBOSE)

1. **[VERB]**: Extensive theoretical background that duplicates scientific_background.rst and the paper
2. **[IMPL]**: `isochrone_population_loglike()` signature and parameters need verification
3. **[IMPL]**: `generate_isochrone_population_grid()` parameters need verification
4. **[DUP]**: "Mixture-Before-Marginalization" section duplicated in faq.rst (verbatim or nearly so)
5. **[EXAMPLE]**: MCMC/emcee examples may not match actual usage patterns

**Recommendation**: MAJOR REDUCTION needed. Focus on practical API usage, remove theory duplication.

---

### `understanding_results.rst`

**Line count**: 581 lines (EXTREMELY VERBOSE)

1. **[IMPL] L16-45**: Output structure shows `results['dist_samples']`, `results['av_samples']`, etc. - THESE ARE FABRICATED. `BruteForce.fit()` saves to HDF5 file, NOT a dict! This entire section is wrong.
2. **[IMPL] L46-63**: Cluster fitting output format - may also be incorrect
3. **[IMPL] L330-354**: References `use_galactic_prior=True/False` and `use_dust_prior=True/False` - THESE PARAMETERS DO NOT EXIST!
4. **[IMPL] L310-315**: `fitter.compute_lnlike_grid()`, `fitter.compute_lnprior_grid()` - methods may not exist with this interface
5. **[IMPL] L357]: `load_models(..., memmap=True)` - unverified parameter
6. **[IMPL] L382-386]: `EEPTracks(use_cache=True)`, `Isochrone(use_cache=True)` - parameters likely don't exist
7. **[IMPL] L447]: `results['R_g']` - unverified key
8. **[IMPL] L461]: `results['M_bol_samples']` - unverified key
9. **[VERB]**: Overall extremely verbose - ASCII art diagrams are nice but take space
10. **[DUP]**: Degeneracy discussion overlaps with scientific_background.rst and faq.rst

**Recommendation**: CRITICAL FIX NEEDED. The output structure section is fundamentally wrong. Verify actual fit() output format from HDF5 file. Condense significantly.

---

### `choosing_options.rst`

**Line count**: 619 lines (EXTREMELY VERBOSE - longest user guide doc)

1. **[IMPL] L67]: `generator.make_grid('my_grid.h5')` - need to verify signature
2. **[IMPL] L119-125]: `generator.make_grid()` with `mini_range`, `eep_range`, etc. - parameters need verification
3. **[IMPL] L146-151]: `BruteForce(grid, use_galactic_prior=True, use_dust_prior=True, use_imf_prior=True)` - THESE PARAMETERS DO NOT EXIST!
4. **[IMPL] L191-198]: `brutus.dust.maps.use_dust_map()` - function likely doesn't exist
5. **[IMPL] L218-224]: `fitter.fit(..., dist_bounds=..., av_max=..., rv_bounds=...)` - parameters may not match actual signature
6. **[IMPL] L269-273]: `fitter.fit(..., ftol=..., maxiter=...)` - parameters may not exist
7. **[IMPL] L289-293]: `fitter.fit(..., n_samples=...)` - parameter may not exist
8. **[IMPL] L357]: `load_models(..., memmap=True)` - unverified
9. **[IMPL] L382-386]: `use_cache=True` parameters - unverified
10. **[VERB]**: Decision trees and trade-off explanations are helpful but take too much space
11. **[DUP]**: Cluster modeling section duplicates cluster_modeling.rst
12. **[DUP]**: Grid vs on-the-fly discussion duplicates stellar_models.rst and grid_generation.rst

**Recommendation**: MAJOR REVISION NEEDED. Fix non-existent parameters. Verify all fit() parameters against actual implementation. Condense significantly (target: ~300 lines).

---

### `faq.rst`

**Line count**: 691 lines (EXTREMELY VERBOSE - longest single doc file)

1. **[IMPL] L159-165**: `BruteForce(grid, use_galactic_prior=False)` and `results['dist_median']` - BOTH NON-EXISTENT
2. **[IMPL] L213-220]: `generator.make_grid()` with `n_mini`, `n_eep` parameters - needs verification
3. **[IMPL] L227-231]: `fitter.fit(..., dist_bounds=..., av_max=...)` - parameters need verification
4. **[IMPL] L246]: `fitter.fit(..., n_samples=1000)` - parameter may not exist
5. **[IMPL] L268]: `load_models(..., memmap=True)` - unverified
6. **[IMPL] L489]: `fitter.fit(..., maxiter=2000)` - parameter may not exist
7. **[IMPL] L504-509]: `generator.make_grid()` with various parameters - needs verification
8. **[VERB]**: Overall extremely verbose - 691 lines for FAQ is excessive
9. **[DUP]**: "What is mixture-before-marginalization" section duplicates cluster_modeling.rst verbatim
10. **[DUP]**: Model selection section largely duplicates stellar_models.rst
11. **[DUP]**: Performance section duplicates choosing_options.rst
12. **[DUP]**: Results interpretation duplicates understanding_results.rst

**Recommendation**: MAJOR REDUCTION needed. Target: ~300 lines. Remove all content duplicated from other docs. Fix non-existent parameters.

---

### API Documentation

The API docs use Sphinx autodoc directives which is good - they'll pull from actual docstrings. However, the narrative examples have issues.

#### `api/core.rst`

**Line count**: 170 lines (reasonable)

1. **[EXAMPLE] L35-38**: `star.get_seds()` example - verify exact signature
2. **[EXAMPLE] L65-67]: `pop.get_seds()` - may be `synthesize()` instead
3. **[ORG]**: Autodoc should work, but verify docstrings are present in source

**Recommendation**: Verify examples match implementation; mostly okay due to autodoc.

---

#### `api/analysis.rst`

**Line count**: 158 lines (reasonable)

1. **[IMPL] L48-53]: `fitter.fit(..., n_samples=10000)` - parameter may not exist
2. **[IMPL] L56-57]: `results['dist_median']`, `results['dist_std']`, `results['av_median']` - WRONG OUTPUT FORMAT
3. **[EXAMPLE] L79-87]: `isochrone_population_loglike()` signature - need to verify
4. **[ORG]**: Autodoc will show actual signatures, which is good

**Recommendation**: Fix narrative examples; autodoc handles the rest.

---

#### `api/data.rst`

**Line count**: ~80 lines

1. **[ORG]**: Should be okay with autodoc - verify docstrings exist

**Recommendation**: Minor, just verify autodoc works.

---

#### `api/priors.rst`

**Line count**: ~80 lines

1. **[ORG]**: Should be okay with autodoc - verify actual function names match exports

**Recommendation**: Minor, verify autodoc generates correct output.

---

#### `api/plotting.rst`

**Line count**: ~80 lines

1. **[ORG]**: Should be okay with autodoc - verify functions exist with correct signatures

**Recommendation**: Minor cleanup only.

---

#### `api/utils.rst`

**Line count**: ~80 lines

1. **[ORG]**: Verify autodoc works correctly

**Recommendation**: Minor, standard autodoc structure.

---

#### `api/dust.rst`

**Line count**: ~60 lines

1. **[ORG]**: Verify dust module is properly importable and has docstrings

**Recommendation**: Minor, verify module is accessible.

---

## Global Issues

These issues appear across multiple documentation files:

### G1. Fabricated Output Format (CRITICAL)
- **Files affected**: quickstart.rst, understanding_results.rst, choosing_options.rst, priors.rst, faq.rst, api/analysis.rst
- **Issue**: Docs claim `BruteForce.fit()` returns a dict with keys like `'dist_samples'`, `'dist_median'`, `'av_samples'`, etc.
- **Reality**: `BruteForce.fit()` saves to an HDF5 file and returns something different
- **Fix**: Document actual output format from HDF5 file structure

### G2. Non-Existent BruteForce Parameters (CRITICAL)
- **Files affected**: priors.rst, understanding_results.rst, choosing_options.rst, faq.rst
- **Issue**: Docs reference `BruteForce(grid, use_galactic_prior=..., use_dust_prior=..., use_imf_prior=...)`
- **Reality**: These constructor parameters do not exist
- **Fix**: Remove references or implement the parameters

### G3. Non-Existent fit() Parameters (HIGH)
- **Files affected**: choosing_options.rst, faq.rst, understanding_results.rst
- **Issue**: Docs reference `fitter.fit(..., dist_bounds=..., av_max=..., rv_bounds=..., n_samples=..., ftol=..., maxiter=...)`
- **Reality**: Many of these parameters may not exist
- **Fix**: Verify actual fit() signature and update all references

### G4. Excessive Verbosity (HIGH)
- **Total line count**: ~4,000+ lines across all user guide docs
- **Target**: ~2,000 lines (50% reduction)
- **Worst offenders**: faq.rst (691), choosing_options.rst (619), understanding_results.rst (581), cluster_modeling.rst (~500), priors.rst (429)

### G5. Content Duplication (MEDIUM)
- **Issue**: Same content appears in multiple places
- **Examples**:
  - "Grid vs on-the-fly" in stellar_models.rst, grid_generation.rst, choosing_options.rst
  - "Mixture-before-marginalization" in cluster_modeling.rst, faq.rst
  - Model selection guidance in stellar_models.rst, faq.rst, choosing_options.rst
  - Degeneracy discussion in scientific_background.rst, understanding_results.rst, faq.rst
- **Fix**: Consolidate in one authoritative location, cross-reference elsewhere

### G6. Incorrect Import Paths (MEDIUM)
- **Files affected**: priors.rst
- **Issue**: References `brutus.priors.stellar.logp_imf`, `brutus.priors.galactic.logp_galactic_structure`
- **Reality**: Correct import is `brutus.priors.logp_imf`, `brutus.priors.logp_galactic_structure`
- **Fix**: Update all import path references

### G7. Missing Data Download Context (LOW)
- **Files affected**: index.rst, quickstart.rst, installation.rst
- **Issue**: Examples don't mention that data files must be downloaded first
- **Fix**: Add notes about data requirements and how to download

---

## Priority Order for Revisions

### 1. **Critical** (blocks basic usage):
- [x] **G1**: Fix fabricated output format in all files - document actual HDF5 structure ✓ Stage 1
- [x] **G2**: Remove or fix non-existent BruteForce constructor parameters ✓ Stage 1
- [x] **quickstart.rst**: Fix fundamentally wrong "Working with Results" section ✓ Stage 1

### 2. **High** (affects common workflows):
- [x] **G3**: Verify and fix all fit() parameters across docs ✓ Stage 1
- [x] **priors.rst**: Fix incorrect function references ✓ Stage 1 (reduction pending)
- [x] **understanding_results.rst**: Complete rewrite of output format section ✓ Stage 1
- [x] **choosing_options.rst**: Remove non-existent parameters ✓ Stage 1 (50% reduction pending)

### 3. **Medium** (improves clarity):
- [x] **G4/G5**: Reduce verbosity and remove duplication ✓ Stage 4+5 (60%+ reduction achieved)
- [x] **faq.rst**: Reduced from 701 to 198 lines (72% reduction) ✓ Stage 5
- [x] **cluster_modeling.rst**: Reduced to 202 lines, API-focused ✓ Stage 4
- [x] **grid_generation.rst**: Reduced to 207 lines ✓ Stage 4

### 4. **Low** (nice to have):
- [x] **G6**: Fix import path references in priors.rst ✓ Stage 1
- [x] **G7**: Add data download notes to getting started docs ✓ Stage 2
- [x] **API docs**: Fixed narrative examples ✓ Stage 6
- [x] **installation.rst**: Mark conda-forge availability as varies ✓ Stage 2
- [x] **scientific_background.rst**: Minor condensing ✓ Stage 3

---

## Notes

- Paper sections for reference:
  - §2: Statistical Framework (priors, likelihood, posterior)
  - §3: Implementation (brute-force algorithm)
  - §4: Stellar and Extinction Models (MIST, Bayestar)
  - §5: Empirical Calibration (cluster fitting, photometric offsets)
  - §6: Validation Tests
  - Appendix A: Detailed prior descriptions
  - Appendix B: Algorithm details

- Target total documentation length: ~2,000 lines (down from ~4,000+)
- Key principle: Docs should be self-contained but concise; link to paper for deep theory

---

## Revision Complete - Final Stats

**Completed**: 2025-12-03

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| understanding_results.rst | 594 | 226 | 62% |
| choosing_options.rst | 619 | 282 | 54% |
| grid_generation.rst | 517 | 207 | 60% |
| cluster_modeling.rst | 488 | 202 | 59% |
| priors.rst | 449 | 183 | 59% |
| faq.rst | 701 | 198 | 72% |
| **Total (major files)** | **3368** | **1298** | **61%** |

**All critical issues resolved:**
- G1: Fabricated output format → Fixed with correct HDF5 structure
- G2: Non-existent BruteForce constructor params → Removed
- G3: Non-existent fit() params → Fixed with correct API
- G4/G5: Verbosity and duplication → 61% reduction achieved
- G6: Incorrect import paths → Fixed
- G7: Missing data download notes → Added

