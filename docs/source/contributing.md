# Contributing

Contributions to `brutus` are welcome. `CLAUDE.md` (symlinked as `AGENTS.md`) at the repository root defines the full engineering, numerical, testing, and documentation standards.

## Development Setup

1. Fork the repository on GitHub.

2. Clone your fork locally:

   ```bash
   git clone https://github.com/yourusername/brutus.git
   cd brutus
   ```

3. Install the package in editable mode with development and documentation dependencies:

   ```bash
   pip install -e ".[dev,docs]" pre-commit
   pre-commit install
   ```

4. Open an issue that states a single goal, then create a branch named `<issue-number>-<short-slug>`:

   ```bash
   git checkout -b 42-fix-parallax-units
   ```

## Running Tests

Run the test suite with `pytest`:

```bash
# Download the model data used by CI; without it, data-dependent tests skip
python .github/scripts/fetch_test_data.py

pytest

# Skip slow tests that load full model grids
pytest -m "not slow"

# Measure test coverage (NUMBA_DISABLE_JIT=1 is required so coverage can
# trace lines inside @jit functions)
NUMBA_DISABLE_JIT=1 pytest --cov=brutus
```

CI runs the test suite with `NUMBA_DISABLE_JIT=1` for coverage across Python 3.10 to 3.13, plus an Ubuntu Python 3.13 leg with Numba JIT enabled. When modifying a `@jit` function, always run its tests locally with JIT enabled before pushing.

## Code Style

Pre-commit hooks run **black** (formatting), **isort** (import sorting), and **flake8** (linting) on each commit:

```bash
pre-commit run --all-files
```

## Documentation

Every public module, class, and function requires a NumPy-style docstring documenting parameter and return types, array shapes, and physical units. Any symbol added to a subpackage's `__all__` must also be listed on its corresponding page in `docs/source/api/` (verified by `tests/test_init.py`). Build the Sphinx documentation with warnings treated as errors, matching CI:

```bash
cd docs
make html SPHINXOPTS="-W --keep-going"
```

## Submitting Changes

1. Make small, self-contained commits with imperative subjects (`type(scope): summary`, such as `fix(priors): normalize the masked-bin dust prior`). Add `[output-change]` before the colon when a change alters numerical outputs for the same inputs and seed.
2. Push the branch to your fork on GitHub.
3. Open a pull request using the repository template and link the issue with `Closes #<issue-number>`.
4. Include unit tests covering new or fixed behaviour and add an entry under `Unreleased` in `CHANGELOG.md` for user-visible changes.

## Reporting Bugs and Requesting Features

- **Bug reports:** Include the `brutus` version, Python and dependency versions, a minimal reproducible example, and the full traceback on [GitHub Issues](https://github.com/joshspeagle/brutus/issues).
- **Feature requests:** Open an issue describing the scientific use case, input/output contracts, and how the feature fits into `brutus`'s architecture.

## Releases

Releases follow Semantic Versioning (`MAJOR.MINOR.PATCH`). The lead maintainer approves each release and publishes distributions to PyPI.

## Contact

- **GitHub Issues:** <https://github.com/joshspeagle/brutus/issues>
- **Email:** <mailto:j.speagle@utoronto.ca>
