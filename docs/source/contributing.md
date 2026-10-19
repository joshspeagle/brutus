# Contributing

We welcome contributions to brutus! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository on GitHub

2. Clone your fork locally:

   ```bash
   git clone https://github.com/yourusername/brutus.git
   cd brutus
   ```

3. Install in development mode:

   ```bash
   pip install -e ".[dev]" pre-commit
   pre-commit install
   ```

4. Open an issue that states one goal, then create a branch for it named
   `<issue number>-<short-slug>`:

   ```bash
   git checkout -b 42-fix-parallax-units
   ```

## Running Tests

We use pytest for testing. Run tests with:

```bash
# Basic tests
pytest

# Skip slow tests (marked with the "slow" marker; they load large
# model data files and skip automatically if the data are absent)
pytest -m "not slow"

# With coverage (NUMBA_DISABLE_JIT=1 is required: numba-compiled
# functions otherwise report 0% coverage even when exercised)
NUMBA_DISABLE_JIT=1 pytest --cov=brutus
```

## Code Style

The pre-commit hooks run **black** (formatting), **isort** (import order)
and **flake8** (linting) on each commit. To run them on every file:

```bash
pre-commit run --all-files
```

## Documentation

- All public functions and classes should have NumPy-style docstrings

- Update documentation when adding new features

- Build docs locally to test:

  ```bash
  cd docs
  make html
  ```

## Submitting Changes

1. Make small commits, one change each. Write the subject as
   `type(scope): summary` in the imperative, for example
   `fix(priors): normalize the masked-bin dust prior`. Add
   `[output-change]` before the colon when the change alters results.
2. Push to your fork on GitHub.
3. Open one pull request per goal. Fill in the template, and close the issue
   with `Closes #<issue number>`.
4. Include tests for new behaviour and a line under "Unreleased" in
   `CHANGELOG.md` for each user-visible change.

The full rules for maintainers and coding agents are in `CLAUDE.md` at the
repository root.

## Issue Reporting

When reporting bugs:

- Include brutus version and Python environment details
- Provide a minimal example that reproduces the issue
- Include full error tracebacks

## Feature Requests

For feature requests:

- Describe the use case and motivation
- Consider if it fits the scope of brutus
- Be willing to help implement or test

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Contact

- GitHub Issues: <https://github.com/joshspeagle/brutus/issues>
- Email: <mailto:j.speagle@utoronto.ca>

Thank you for contributing to brutus!
