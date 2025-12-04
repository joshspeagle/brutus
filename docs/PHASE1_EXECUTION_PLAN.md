# Phase 1 UX Improvement Execution Plan

**Date**: 2025-12-04
**Objective**: Improve documentation accessibility and user experience without adding bloat

---

## Guiding Principles

1. **Conciseness over comprehensiveness** - Add clarity, not volume
2. **Actionable over descriptive** - Decision trees and checklists, not prose
3. **Progressive disclosure** - Key info first, details for those who need them
4. **Consistent terminology** - Glossary as single source of truth

---

## Stage 1: Foundation - Glossary

**Goal**: Create shared vocabulary that all other docs can reference

**Files**:
- CREATE: `docs/source/glossary.rst`
- EDIT: `docs/source/index.rst` (add to toctree)

**Content** (glossary.rst):
- ~25 key terms with concise definitions
- Organized by category: Stellar Parameters, Extinction, Data, Methods
- Each definition: 1-2 sentences max
- Cross-references to detailed docs where relevant

**Terms to include**:
- EEP (Equivalent Evolutionary Point)
- IMF (Initial Mass Function)
- A_V (V-band extinction)
- E(B-V) (color excess)
- R_V (extinction curve parameter)
- MIST (stellar models)
- Parallax, Distance modulus
- Isochrone, Evolutionary track
- Magnitude, Flux density
- Posterior, Prior, Likelihood

---

## Stage 2: Critical Onboarding - Installation & Data

**Goal**: Prevent "it doesn't work" frustration for first-time users

**Files**:
- EDIT: `docs/source/installation.rst`

**Changes**:
1. Add prominent checklist after "Data Files" section
2. Distinguish mandatory vs optional downloads
3. Add disk space table
4. Add verification step

**Format**:
```
┌─────────────────────────────────────────────────────┐
│ Before You Start: Data Download Checklist           │
├─────────────────────────────────────────────────────┤
│ [ ] Grids (REQUIRED) - 1-5 GB                       │
│ [ ] Isochrones (REQUIRED) - ~100 MB                 │
│ [ ] Dust maps (optional) - ~1 GB                    │
└─────────────────────────────────────────────────────┘
```

---

## Stage 3: Navigation - Getting Started Decision Tree

**Goal**: Help users immediately identify the right workflow

**Files**:
- EDIT: `docs/source/index.rst`

**Location**: After the introduction, before "Key Features"

**Format**: Text-based decision tree
```
Which workflow should I use?
├── I want to fit stellar parameters to photometry
│   ├── Large sample (>100 stars): Use BruteForce + StarGrid
│   └── Small sample or custom filters: Use StarEvolTrack
├── I want to model a star cluster
│   └── Use Isochrone + StellarPop with MCMC
└── I want to generate synthetic photometry
    ├── For individual stars: Use StarEvolTrack
    └── For populations: Use StellarPop
```

---

## Stage 4: Core Flow - Quickstart Restructure

**Goal**: Lead with recommended workflow, not historical development order

**Files**:
- EDIT: `docs/source/quickstart.rst`

**New Structure**:
1. Workflow Selection (NEW - brief decision guidance)
2. Data Download (move up, make prominent)
3. Fitting with BruteForce (MOVE UP - main workflow)
4. Working with Results (keep)
5. Generating Photometry (MOVE DOWN - secondary)
6. Common Workflows (keep)
7. Next Steps (keep)

**Key Changes**:
- Add 5-line "Which approach?" intro
- Move BruteForce fitting BEFORE on-the-fly generation
- Keep existing code examples (they're good)

---

## Stage 5: Decision Support - Enhanced Diagnostics

**Goal**: Help users interpret results with clear thresholds

**Files**:
- EDIT: `docs/source/understanding_results.rst`
- EDIT: `docs/source/choosing_options.rst`

**Changes to understanding_results.rst**:
1. Add chi-squared interpretation table:
   - χ² < 0.5: Errors likely overestimated
   - χ² 0.8-1.2: Good fit
   - χ² 1.2-2.0: Acceptable, check residuals
   - χ² 2.0-3.0: Marginal, investigate
   - χ² > 3.0: Poor fit, check data quality

2. Enhance degeneracy solutions with specific recommendations

**Changes to choosing_options.rst**:
Add "Expected Runtimes" section:
- Single star: 1-10 seconds
- 1000 stars (parallel): 15-30 minutes
- Cluster MCMC: 1-8 hours

---

## Stage 6: Polish & Cross-references

**Goal**: Ensure consistency and good navigation

**Files**:
- Multiple (light edits)

**Tasks**:
1. Add `:doc:`glossary`` references where helpful
2. Verify terminology consistency
3. Final read-through for flow
4. Commit with clear message

---

## Commit Strategy

Will commit after each stage:
1. `Stage 1: Add glossary for consistent terminology`
2. `Stage 2: Improve installation with data checklist`
3. `Stage 3: Add getting started decision tree`
4. `Stage 4: Restructure quickstart for clarity`
5. `Stage 5: Enhance diagnostics with decision guidance`
6. `Stage 6: Add cross-references and polish`

Final push after all stages complete.

---

## Estimated Changes

| Stage | Files Modified | Lines Added | Lines Removed |
|-------|---------------|-------------|---------------|
| 1     | 2             | ~120        | 0             |
| 2     | 1             | ~30         | 0             |
| 3     | 1             | ~40         | 0             |
| 4     | 1             | ~20         | ~10           |
| 5     | 2             | ~40         | ~10           |
| 6     | 3-5           | ~10         | ~5            |
| **Total** | **6-8**   | **~260**    | **~25**       |

Net addition: ~235 lines (mostly the glossary)

---

## Success Criteria

After Phase 1 completion:
- [ ] New user can identify correct workflow in <30 seconds
- [ ] Installation page has clear mandatory/optional distinction
- [ ] All key terms defined in one place
- [ ] Quickstart leads with recommended approach
- [ ] Diagnostics have actionable thresholds
- [ ] No jargon without glossary reference
