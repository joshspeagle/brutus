# Documentation Revision Plan

This document outlines the multi-stage plan for systematically revising the brutus documentation.

**Goal**: Fix all errors, reduce verbosity by ~50%, eliminate duplication, and ensure accuracy against actual implementation.

**Reference**: Speagle et al. (2025), arXiv:2503.02227 (local copy at `/tmp/brutus-reference/`)

---

## Stage 0: Ground Truth Discovery

**Purpose**: Before fixing docs, establish what the actual implementation does.

### Tasks:
1. **Document actual `BruteForce.fit()` behavior**
   - Read the fit() method implementation thoroughly
   - Identify actual parameters and their defaults
   - Document actual return value / HDF5 output structure
   - Create a "truth table" of parameters vs what docs claim

2. **Document actual `BruteForce.__init__()` parameters**
   - Verify which constructor parameters exist
   - Note what priors are actually controllable and how

3. **Verify other key APIs**
   - `load_models()` signature and parameters (memmap?)
   - `GridGenerator.make_grid()` signature
   - `isochrone_population_loglike()` signature
   - `cornerplot()` actual signature
   - `EEPTracks`, `Isochrone`, `StellarPop` constructors

4. **Create reference document**
   - Write `docs/API_TRUTH.md` with verified signatures
   - Use this as source of truth for all subsequent revisions

### Deliverable:
- `docs/API_TRUTH.md` - Verified API reference for doc writers

---

## Stage 1: Critical Fixes (G1 + G2)

**Purpose**: Fix the most fundamental errors that completely mislead users.

### 1A: Fix Output Format Documentation (G1)

**Paper reference**: §3 (Implementation), particularly the output description

**Files to update** (in order):
1. `understanding_results.rst` - Primary location, rewrite "Output Structure" section
2. `quickstart.rst` - Fix "Working with Results" section
3. `choosing_options.rst` - Remove incorrect output references
4. `priors.rst` - Fix code examples
5. `faq.rst` - Fix code examples
6. `api/analysis.rst` - Fix narrative examples

**Approach**:
- First, examine actual HDF5 output from fit()
- Write accurate output documentation once in understanding_results.rst
- Update all other files to reference or mirror this correct version

### 1B: Fix Non-Existent Parameters (G2)

**Files to update**:
1. `priors.rst` - Remove `use_galactic_prior`, `use_dust_prior` references
2. `understanding_results.rst` - Same
3. `choosing_options.rst` - Same, plus `use_imf_prior`
4. `faq.rst` - Same

**Approach**:
- Search-and-remove all instances of fabricated parameters
- Document actual way to control priors (if any exists)
- If no way exists, note this as a limitation or future feature

### Deliverable:
- All critical G1/G2 issues resolved
- Accurate output format documented

---

## Stage 2: User Entry Points

**Purpose**: Fix the docs users see first - these set expectations.

**Paper reference**: Abstract and §1 (Introduction) for positioning/scope

### 2A: `quickstart.rst` (Major revision)

**Changes**:
1. Fix "Working with Results" section completely
2. Show actual BruteForce workflow with correct output handling
3. Fix `cornerplot()` example or remove it
4. Add note about data download requirements
5. Keep concise - this should be ~100 lines

### 2B: `installation.rst` (Minor revision)

**Changes**:
1. Verify conda-forge availability or remove that section
2. Add note about data file requirements (disk space)
3. Keep concise - ~80 lines is good

### 2C: `index.rst` (Minor revision)

**Changes**:
1. Verify code examples work
2. Add brief note about data downloads
3. Verify `photometric_offsets.rst` exists (or fix toctree)
4. Keep length similar

### Deliverable:
- Clean, accurate entry point documentation
- Users can actually follow quickstart successfully

---

## Stage 3: Scientific Background Docs

**Purpose**: Revise conceptual docs using paper as authoritative source.

### 3A: `scientific_background.rst` (Minor revision)

**Paper reference**: §2 (Statistical Framework)

This is already well-written. Minor changes:
1. Condense "Why Bayesian Inference?" section slightly
2. Ensure alignment with paper terminology
3. Keep as template for tone/style of other docs

**Target**: ~170 lines (from 189)

### 3B: `stellar_models.rst` (Medium revision)

**Paper reference**: §4 (Stellar and Extinction Models)

**Changes**:
1. Verify all code examples against implementation
2. Condense extinction/neural network sections
3. Remove duplication with grid_generation.rst
4. Add note about data file requirements

**Target**: ~280 lines (from 335)

### 3C: `priors.rst` (Major revision)

**Paper reference**: §2.4 and Appendix A (Prior descriptions)

**Changes**:
1. Fix ALL incorrect function references (G6)
2. Remove fabricated BruteForce parameters
3. Fix code examples with wrong output format
4. Significantly condense - much duplicates scientific_background.rst
5. Focus on: what priors exist, how to use them, when to customize
6. Remove detailed Galactic model math (reference paper instead)

**Target**: ~200 lines (from 429) - 53% reduction

### Deliverable:
- Accurate, concise scientific background
- Paper properly referenced for deep theory

---

## Stage 4: Workflow Documentation

**Purpose**: Fix the practical "how to use" guides.

### 4A: `understanding_results.rst` (Major rewrite)

**Paper reference**: §3 and §6 (Implementation, Validation)

**Changes**:
1. COMPLETELY REWRITE "Output Structure" section with actual format
2. Remove non-existent parameters
3. Keep diagnostic guidance (chi-squared, residuals) - this is useful
4. Condense degeneracy discussion (already in scientific_background.rst)
5. Remove duplicate reliability checklist content
6. Keep ASCII art diagrams if space permits (they're helpful)

**Target**: ~350 lines (from 581) - 40% reduction

### 4B: `choosing_options.rst` (Major revision)

**Paper reference**: §3 (Implementation details)

**Changes**:
1. Remove ALL non-existent parameters
2. Verify actual fit() parameters and document correctly
3. Remove cluster modeling section (belongs in cluster_modeling.rst)
4. Remove grid vs on-the-fly discussion (in stellar_models.rst)
5. Focus on: actual configurable options, when to use each
6. Keep decision trees but make more concise

**Target**: ~300 lines (from 619) - 52% reduction

### 4C: `grid_generation.rst` (Medium revision)

**Paper reference**: §3.1 (Grid construction)

**Changes**:
1. Verify GridGenerator API
2. Remove overlap with stellar_models.rst
3. Focus on practical grid creation workflow
4. Document actual available grids and naming

**Target**: ~250 lines (from ~400) - 38% reduction

### 4D: `cluster_modeling.rst` (Major revision)

**Paper reference**: §5 (Empirical Calibration, cluster fitting)

**Changes**:
1. Remove extensive theoretical background (in scientific_background.rst)
2. Verify `isochrone_population_loglike()` signature
3. Focus on practical cluster fitting workflow
4. Keep mixture-before-marginalization explanation (brief version)
5. Remove duplicate content with faq.rst

**Target**: ~300 lines (from ~500) - 40% reduction

### Deliverable:
- Practical, accurate workflow guides
- No duplicate content between files

---

## Stage 5: FAQ Consolidation

**Purpose**: After other docs are fixed, trim FAQ to truly unique Q&A.

**Approach**: The FAQ should ONLY contain:
- Questions not answered elsewhere
- Brief answers that link to detailed docs
- Troubleshooting specific error messages
- Citation/contribution info

### Changes:
1. Remove "Model Selection" section (now in stellar_models.rst)
2. Remove "Performance" section (now in choosing_options.rst)
3. Remove "Results Interpretation" (now in understanding_results.rst)
4. Remove "mixture-before-marginalization" detail (in cluster_modeling.rst)
5. Keep: Getting Started basics, Error messages, Data formats, Citation
6. Convert remaining sections to brief Q&A with links

**Target**: ~250 lines (from 691) - 64% reduction

### Deliverable:
- Concise FAQ with no duplication
- Links to authoritative sections elsewhere

---

## Stage 6: API Documentation

**Purpose**: Ensure API docs are accurate and autodoc works.

### Tasks:
1. Fix narrative examples in `api/analysis.rst` and `api/core.rst`
2. Verify all autodoc directives work (build docs locally)
3. Ensure docstrings in source code are accurate
4. Fix `pop.get_seds()` vs `pop.synthesize()` confusion

### Deliverable:
- Working autodoc-generated API reference
- Accurate narrative examples

---

## Stage 7: Final Review & Polish

**Purpose**: Ensure consistency and completeness.

### Tasks:
1. Cross-check all internal links work
2. Verify terminology consistency across docs
3. Run spell check
4. Build docs locally and review
5. Update DOCUMENTATION_AUDIT.md to mark items complete
6. Final line count check against targets

### Deliverable:
- Complete, consistent documentation
- All audit items checked off

---

## Execution Strategy

### Context Management
- Work on one stage at a time to avoid context overflow
- Keep `API_TRUTH.md` and paper sections accessible
- Commit after each major file revision

### Paper Usage
For each section, pull relevant context from paper:
- §2: Statistical framework → scientific_background.rst, priors.rst
- §3: Implementation → understanding_results.rst, choosing_options.rst, grid_generation.rst
- §4: Models → stellar_models.rst
- §5: Calibration → cluster_modeling.rst, photometric_offsets
- Appendix A: Priors → priors.rst

### Iteration Order
1. Stage 0 first (ground truth) - required for all other stages
2. Stage 1 next (critical fixes) - unblocks users
3. Stages 2-4 can be done file-by-file
4. Stage 5 must come after 2-4 (FAQ depends on other docs being done)
5. Stages 6-7 are cleanup

### Progress Tracking
- Update `DOCUMENTATION_AUDIT.md` "Revised" column after each file
- Commit frequently with descriptive messages
- Keep running line count to track verbosity reduction

---

## Target Outcomes

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Total lines (user guides) | ~4,000+ | ~2,000 | 50% |
| faq.rst | 691 | 250 | 64% |
| choosing_options.rst | 619 | 300 | 52% |
| understanding_results.rst | 581 | 350 | 40% |
| priors.rst | 429 | 200 | 53% |
| cluster_modeling.rst | ~500 | 300 | 40% |

| Quality Metric | Before | After |
|----------------|--------|-------|
| Critical errors (G1, G2) | Multiple | 0 |
| Fabricated parameters | 10+ instances | 0 |
| Duplicate sections | 5+ major | 0 |
| Incorrect import paths | 4+ | 0 |

---

## Ready to Begin?

Start with **Stage 0: Ground Truth Discovery** to establish what the actual implementation does before making any documentation changes.
