# brutus Audit, Cleanup & Release-Prep — Design & Plan

**Date:** 2026-05-29
**Branch:** `audit/v1.1.0-cleanup`
**Author:** Claude (autonomous overnight session, ultracode mode)
**Status:** APPROVED-BY-PREAUTHORIZATION (user asked to review the final suite of changes in the morning)

---

## 1. Context & Goals

Systematic audit, cleanup, and polish of `brutus` (PyPI `astro-brutus`) ahead of a PyPI
release. Six dimensions requested by the maintainer:

1. Clean, bug-free code (fix *verified* bugs).
2. A test suite that exercises realistic scientific use cases (not coverage theater);
   add/remove/combine tests for efficiency.
3. Properly documented & cross-referenced code internals (for humans, agents, and autodoc).
4. Clean, comprehensive, polished documentation; tutorials properly finalized/run/integrated;
   improved website UX.
5. Cleaned-up, polished tutorials that each serve a clear pedagogical purpose.
6. Proper versioning and PyPI release preparation.

## 2. Operating Mode & Guardrails (autonomous overnight)

The maintainer is asleep and will review the result in the morning. Therefore:

- **Work on the existing branch `audit/v1.1.0-cleanup`.** No push, no PR, nothing
  outward-facing. Local commits only.
- **Prepare the release but DO NOT publish.** Build + `twine check` only. The exact publish
  command is documented for the maintainer to run after review.
- **Keep the test suite green throughout.** Run the relevant subset after every change set;
  never commit a red suite.
- **Bug-fix risk policy (maintainer's choice "Fix + isolate + document"):** verified bugs are
  fixed; any fix that changes *numerical/scientific output* goes in its **own commit**, tagged
  `[output-change]`, with before/after notes, so each can be reviewed or reverted independently.
- **Adversarially verify bugs before fixing** (no acting on false positives).
- **Deviation from the brainstorming skill's interactive approval gate** is intentional and
  authorized by the maintainer's "work autonomously … review in the morning" instruction.

### Workflow-orchestration strategy (ultracode)

- **Workflows (parallel agents) are used for read-only work:** adversarial bug verification,
  finding all instances of a pattern, validating doc examples against live signatures, and
  reviewing my own diffs.
- **Interdependent / numeric edits and notebook execution are done in the main loop** for tight
  control and per-step verification. Parallel file-mutating agents are deliberately avoided here
  because the changes are interdependent and correctness-critical in a scientific package;
  parallel edits would conflict and undermine verifiability. (This is a considered deviation from
  "orchestrate everything with workflows," consistent with the maintainer's invitation to push
  back where it conflicts with doing the job well.)

## 3. Resolved Decisions (ground-truth verified this session)

| Decision | Resolution | Evidence |
|---|---|---|
| **Release version** | **1.1.0** | PyPI has only `0.8.2`; 0.9.0/1.0.0 never published. Branch named `v1.1.0`. New public features since the 1.0.0 changelog entry (combine_seds, summary_plot, max_models, precision_shrinkage default, systematic errors) → minor bump per semver. |
| **Python floor** | **3.9** (drop 3.8) | 3.8 is EOL, untested in CI, advertised in metadata. Add 3.13 to CI (works locally). |
| **numpy support** | **Support numpy 2.x** | Package imports & no-data tests pass on numpy 2.2.6 locally. Raise numba floor to a numpy-2-compatible version; add a numpy-2.x CI leg. |
| **Canonical Orion field** | **`Orion_l209.1_b-19.9`** (207 objs, committed) | The `l204.7_b-19.2` field referenced by tuts 10/11 exists nowhere locally. Regenerate `_mist`/`_nodust` results from the committed 207-object file + local `grid_mist_v9.h5`. |
| **Tutorial data delivery** | **Commit regenerated results to repo** | Matches existing pattern (`Orion_l209.1_b-19.9.h5` already committed, ~20 KB; results ~small). No Dataverse upload possible/needed. |
| **Test data policy** | **skip-if-missing** + `requires_data` marker | Suite must run offline; CI still enforces the data path. |
| **napoleon vs numpydoc** | **Keep napoleon, drop numpydoc** | Avoid double-processing; add explicit `:func:`/`:class:` inline roles for cross-linking (more reliable than relying on numpydoc See-Also resolution). |
| **chi²/Nbands** | Replace everywhere with **p-value** methodology | Matches CLAUDE.md; fixes tut 07 + understanding_results.rst. |

### Local environment (verified)
Python 3.13.5 · numpy 2.2.6 · scipy 1.16.1 · numba 0.61.2 · h5py 3.14 · healpy 1.18.1 · pooch 1.8.2.
Model data present: `~/.cache/astro-brutus/{grid_mist_v9.h5, nn_c3k.h5, offsets_*.txt}` + symlinked
bayestar; full 8.5 GB DATAFILES at `/mnt/d/.../data/DATAFILES/`. **Fits can run.**

### Confirmed environment blocker (fixed first)
`pyproject.toml` `addopts` includes `--cov-report=html`; on this WSL+Dropbox (DrvFs) path every
`pytest` invocation (even `--collect-only`) dies with `OSError: [Errno 22]` writing `htmlcov/`.
**Remove `--cov*` from default `addopts`** (invoke coverage explicitly, as CI does) — prerequisite
for running tests at all locally.

## 4. State Assessment (from 8-explorer survey, 72 issues: 2C/15H/19M/36L)

The package is **fundamentally healthy**: scientific core verified correct (Fisher matrix matches
finite differences to ~1e-6, scipy cross-checks, correct unit conventions), 100% module/class +
92% function docstring coverage, modern build config, well-organized docs. This is a **polish**
job, not a rescue. Issues cluster in: stale CHANGELOG/release process, stale doc/rst code
examples, a handful of real code bugs, test-infra hygiene, and unfinalized tutorials.

## 5. Confirmed Bug List & Risk Classification

**Crash / non-numeric (fix freely, normal commits, +regression test):**
- `load_offsets()` crashes on single-row offsets file (`loader.py`) — verified.
- `cornerplot()` mutates caller's `labels` list in place (`corner.py`) — verified.
- `logp_imf` ZeroDivisionError for slope == 1.0 (`stellar.py`) — verified.
- Dead/misleading `scipy.polyfit` import (`grid_generation.py`).
- `dist_vs_red` `interpolation=None` → `'none'`; add `ax`/`fig` (`distance.py`).
- `_fetch()` no symlink fallback + stale-symlink `FileExistsError` (`download.py`) — relevant on WSL.
- `Nmc` can collapse to 0 for tiny `mem_lim`; guard `dim_prior` for `Ndim<=0` (`individual.py`).
- Mutable default arg, dead attrs (`self.null`, `self.binwidths`, `_n_pix`, `get_query_size`).
- Numerous docstring/behavior mismatches (quantile example, bin defaults, `photometric_offsets_2d`
  required-args + "In not provided" typo, galactic module-docstring signature, `inverse3` example,
  BruteForce.fit missing 5 params).

**Output-changing (isolated `[output-change]` commits, before/after notes):**
- `FastNN.encode` mis-scales a batch of *exactly 6* stars via broadcasting-failure dispatch
  (`neural_nets.py`) — replace with explicit `ndim`/orientation check. Verified reproducible.
- `los_dust` kernels collapse per-object means via `.flat[0]`, breaking the `template_reds` path
  (`los_dust.py`) — make kernels element-wise. Verified reproducible.
- Binary SED combination NaNs a valid primary when the secondary-EEP solve misses a very tight
  tolerance (`individual.py`) — relax/expose tolerance or fall back to primary-only + warn.

**Judgment/contract items (document or guard; flag if behavior-affecting):**
- Sub-4-band / all-masked fits proceed without erroring → guard + document contract.
- StarGrid unspecified-dimension fallback to `grid_axes[...][0]` → document.
- Gumbel-max subsampling without inclusion-prob reweighting → document as approximation.
- `photometric_offsets` name collision (plotting vs analysis) → document distinction.
- `ASTRO_BRUTUS_DATA_DIR` vs `BRUTUS_DATA_DIR` env-var inconsistency → unify + document.

Every fix is re-confirmed by an adversarial verification pass before editing.

## 6. Phased Plan

**Phase 0 — Baseline & inventory** ✅ (this session): branch confirmed, data inventoried, PyPI
checked, stack verified, green baseline established, `addopts` crash reproduced.

**Phase 1 — Code audit & bug fixes:** adversarially verify the bug list; fix crash/non-numeric
bugs + regression tests; fix output-changing bugs in isolated commits; remove dead code; fix
in-code docstrings; guard edge cases. Verify with targeted test runs + numeric before/after checks.

**Phase 2 — Test-suite overhaul:** remove `--cov*` from `addopts` (unblock local runs); reconcile
the NUMBA_DISABLE_JIT story (make `=1` canonical, fix/remove misleading conftest block, delete
unused `run_coverage.py` story if obsolete); convert 5 AssertionError-on-missing-data sites to
`skipif` + add `requires_data` marker; register & apply the `slow` marker; tighten
`filterwarnings`; delete/consolidate vacuous import/version tests; add a golden recovery test on
the real MIST subset; add the missing-but-documented tutorial test harness (with Phase 5).

**Phase 3 — Internal documentation:** fix all in-code docstring examples to match signatures; add
the 5 missing `BruteForce.fit` params; add inline `:func:`/`:class:` cross-reference roles to the
most-referenced symbols; ensure `__all__` symbols are reflected in curated autodoc.

**Phase 4 — Documentation & website:** fix all stale `.rst` code examples (priors/utils/data/
grid_generation/quickstart/stellar_models/api/core/api/analysis); add `summary_plot` to autodoc;
add a p-value fit-quality subsection (replace chi²/Nbands framing); resolve napoleon/numpydoc;
fix Furo→pydata CSS variable; remove dead autosummary config (or adopt tables); set RTD
`fail_on_warning: true` after warnings are cleared; update CLAUDE.md/MEMORY (faq.rst, env var,
tutorial-test claims); **build docs locally (`make html`) and drive warnings to zero.**

**Phase 5 — Tutorials:** standardize on the `l209.1_b-19.9` field; fix the three-way results-file
naming mismatch across `tutorial_utils.py` / producer (tut 05) / consumers (tuts 07/10/11);
regenerate results by actually running the fit; **re-run all 12 notebooks end-to-end with outputs**
(data is local); fix tut 07 chi²→p-value; remove author-specific absolute paths from
`tutorial_utils.find_brutus_data_file`; integrate a real pytest-collected tutorial test that fails
on genuine errors + a CI job (skips when data absent); verify docs render the executed notebooks.

**Phase 6 — Release prep (no publish):** bump version → 1.1.0 (pyproject + `__init__`); rewrite
CHANGELOG with an accurate, dated 1.1.0 section covering all post-1.0.0 work; set
`requires-python>=3.9`, fix classifiers/README/black/mypy targets; resolve numpy/numba deps + add
numpy-2.x & py3.13 CI legs; fix `MANIFEST.in` path; add a tag-triggered publish workflow file
(Trusted Publishing) **without triggering it**; clean stale build artifacts; `python -m build` +
`twine check dist/*`; document the publish runbook. **Do not upload.**

## 7. Verification Strategy

- After each code change: run the directly-affected test module(s) with
  `NUMBA_DISABLE_JIT=1 pytest -o addopts="-p no:cacheprovider" …`.
- Output-changing fixes: capture explicit numeric before/after in the commit message.
- Before final handoff: full suite (with data) under both JIT-on and `NUMBA_DISABLE_JIT=1`;
  `make html` docs build with zero warnings; `python -m build` + `twine check`.
- A final review workflow audits the cumulative diff for regressions/inconsistencies.

## 8. Out of Scope / Deferred / Flagged for Maintainer

- Actual PyPI/TestPyPI upload (maintainer runs post-review).
- Dataverse uploads of any data (cannot perform; example data committed to repo instead).
- Large local clutter that is already git-ignored (`tutorials/calib`, `.ipynb_checkpoints`,
  `data/`, `work/`) — left untouched on disk; not a repo concern.
- Any behavior change where the "right" answer is a genuine science/product decision
  (e.g. whether `template_reds`/binary-grid paths are supported) — fixed correctly but flagged
  with a note for confirmation.

## 9. Risks & Rollback

- All work is on `audit/v1.1.0-cleanup`; `git reset`/branch-delete fully reverts.
- Output-changing fixes are isolated commits → individually revertible.
- Notebook re-execution is resource/time-bound; any notebook that cannot complete (memory/time)
  is reported rather than committed half-run.

## 10. Definition of Done (morning deliverable)

A clean branch with logically-grouped commits, a green test suite (offline-skippable + full),
zero-warning docs build, finalized & executed tutorials, version 1.1.0 prepared with an accurate
CHANGELOG and validated build artifacts, and a written summary covering: what changed, what was
found-but-deliberately-not-changed (and why), what needs a maintainer decision, and the exact
publish command.
