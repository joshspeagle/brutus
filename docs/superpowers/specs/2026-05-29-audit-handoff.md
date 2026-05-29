# brutus Audit — Morning Handoff (2026-05-29)

Branch: `audit/v1.1.0-cleanup` — pushed to GitHub (`origin`) and checked out in the
WSL-native ext4 clone `~/brutus_clean`. **All work is on GitHub.**

## TL;DR

- 14 commits implementing verified bug fixes, doc accuracy, release prep, and
  tutorial corrections — all pushed.
- The release (v1.1.0) is **prepared but not published** (your instruction):
  `python -m build` + `twine check` PASS. Publish steps below.
- **Read the "Needs your decision" section** — three fixes change numerical
  output on specific paths, and there are a few deferred items.

## ⚠️ Environment root cause (important)

The repo lives at `/mnt/c/Users/joshs/Dropbox/GitHub/brutus` — a **Dropbox-synced
Windows drive via WSL/DrvFs**. Mid-session, git there began failing hard
(`pack-objects` SIGBUS, `Input/output error` reading `.git/objects`, empty reads
of "online-only" placeholder objects). This is a known-bad place to host an
active git repo: Dropbox syncing/placeholdering `.git` corrupts git's mmap/reads.

**Recovery:** cloned clean `master` from GitHub to ext4, grafted my (readable,
locally-written) commit objects on top, `fsck`-verified the full branch, and
pushed. No work was lost.

**Recommendation:** do future git work from a **non-Dropbox, native filesystem**
path (e.g. keep using `~/brutus_clean`, or re-clone to `~/brutus`). If you want
Dropbox backup of source, exclude `.git/` from Dropbox sync, and rely on GitHub
for history. The `/mnt/c` copy is frozen at commit `daf8c2c` and should be
considered stale.

## Per-phase status

| Phase | Status | Notes |
|---|---|---|
| 0 Baseline | ✅ done | branch, data inventory, PyPI check (only 0.8.2 was ever published), green baseline |
| 1 Code bugs | ✅ done | 9 verified bugs fixed + regression tests; minor correctness fixes |
| 2 Tests | ◑ partial | regression tests added; `addopts`/`conftest` coverage reconciled; **deferred:** missing-data→skip conversions, golden recovery test |
| 3 Internal docs | ✅ mostly | docstring examples fixed, `BruteForce.fit` params documented, footnote refs de-duped; **deferred:** broader inline `:func:`/`:class:` cross-ref roles |
| 4 Docs site | ✅ done | stale `.rst` examples fixed, `summary_plot` added, single docstring engine, **zero-warning build**; **deferred:** RTD `fail_on_warning: true` (after you confirm a clean RTD build) |
| 5 Tutorials | ◑ source-fixed | naming/chi²/utils fixed in source; **NOT executed — see below** |
| 6 Release | ✅ prepared | v1.1.0, deps, MANIFEST, publish workflow, build+twine PASS; **not published** |

## The 9 fixed bugs (all adversarially verified + regression-tested)

Output-affecting (isolated commits, tagged `[output-change]`):
1. `FastNN.encode` — silent mis-scaling for exactly-6-sample batches.
2. `los_dust` kernels — collapsed per-object cloud means (template path).
3. Binary companion EEP — discarded valid primary; relaxed default `tol` 1e-6→1e-2.

Crash/robustness (no change to valid-path output):
4. `load_offsets` single-row crash · 5. `logp_imf` slope==1 div-by-zero ·
6. `_fetch` stale-symlink/no-symlink fallback · 7. `Nmc` floor ·
8. `dim_prior` log(0) + fully-masked fail-fast · 9. `cornerplot` label mutation.
Plus: `dist_vs_red` raw-bin imshow, `hist2d` smoothing guard, dead `scipy.polyfit` import.

## ❗ Needs your decision / review

- **Binary-grid tolerance (fix #3):** I relaxed `StarEvolTrack.get_seds(tol=...)`
  from 1e-6 to 1e-2 dex and added a primary-only fallback. This **changes the
  content of regenerated binary (smf>0) grids** (more valid models; some now
  primary-only). Single-star fitting and the shipped default grids are
  unaffected. Please confirm the 1e-2 dex choice; regenerate binary grids if used.
- **`los_dust` template path (fix #2)** and **`encode` N=6 (fix #1):** both
  changed silently-wrong outputs to correct ones — confirm no downstream analysis
  relied on the (buggy) old behavior.
- **Tutorials are source-corrected but NOT re-executed.** Reliable execution here
  was blocked by the same Dropbox-placeholder data problem (the multi-GB MIST
  track/iso files and the bayestar symlink resolve to flaky `/mnt/c`|`/mnt/d`
  reads). They are now *ready* to run: do a "Restart & Run All" on a machine
  where the data is reliably local (ext4 cache + downloaded grids). Order:
  run `tutorial_05` first (it produces the `_mist`/`_mist_nodust` results that
  07/10/11 consume); `tutorial_06` also needs `pip install emcee`.

## Deferred (recommended follow-ups)

- Phase 2: convert the 5 missing-data `AssertionError` sites to `pytest.skip` +
  a `requires_data` marker (offline-runnable suite); add a golden recovery test.
- CI (`tests.yml`): add a numpy-2.x leg + Python 3.13, and drop the `numpy<2.0`
  pin so CI tests the real install path. (Left unchanged because I can't run GH
  Actions here to verify; the package works on numpy 2.2/py3.13 locally.)
- RTD `fail_on_warning: true` once a clean RTD build is confirmed.
- Optional hygiene: remove unused `EEPTracks.null`/`binwidths`, `maps._n_pix`;
  decide on `get_query_size` (public but unused).

## To publish v1.1.0 (when ready)

The built artifacts are validated (`dist/astro_brutus-1.1.0.{tar.gz,whl}`,
`twine check` PASS). Two options:

1. **Automated (recommended):** configure PyPI Trusted Publishing for
   `astro-brutus` (owner `joshspeagle`, repo `brutus`, workflow `release.yml`,
   environment `pypi`), then `git tag v1.1.0 && git push origin v1.1.0`. The new
   `release.yml` builds, checks the tag==version, and publishes.
2. **Manual:** `python -m build && twine upload dist/*` with your PyPI token.

Before tagging: merge this branch (review the diff / open a PR from
`audit/v1.1.0-cleanup`).

## Test status

- 414/414 pass across every module I changed, including all new regression tests
  (run on ext4 with the local grid cache).
- The full suite was **728 passed** earlier (on `/mnt/c`, before the env failure;
  the single failure then was my own test, since corrected).
- 3 `EEPTracks` init tests fail **only** in the ext4 environment because the
  multi-GB MIST track file resolves to a flaky Dropbox-placeholder read — not a
  code regression (that code was untouched). They pass with reliably-local data.
