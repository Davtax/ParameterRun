# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

## [0.4.0] - 2026-04-21

### Added

- Added `tests/test_mpi_integration.py` with MPI numerical-equivalence coverage for `parameterrun(..., backend="mpi")`.
- Added a dedicated Linux PR CI job that runs MPI tests with 4 ranks.

### Changed

- Updated default test commands in docs and CI to run non-MPI tests with `-m "not mpi"`.
- Registered a pytest `mpi` marker in `pyproject.toml` for explicit MPI test selection.
- Update how the indices to compute are shared among workers. Previously, all the indices where computed by all workers,
  which caused a significan overhead in the memory usage. Now, the indices are computed on the fly.

## [0.3.0] - 2026-04-01

### Changed

- Added `result_as_array` to `parameterrun(...)`: outputs are numpy arrays by default and can be converted to Python
  lists with `result_as_array=False`.
- Removed the `pbar_kwargs` optional argument from `parameterrun(...)`, as it was unused.
- Added more tests for the depth of the parameter grid.
- Added the possibility to add parameter values as numpy arrays, which are converted to lists internally.

## [0.2.1] - 2026-04-01

### Added

- Modified GitHub Actions workflow to trigger on pull requests to the main branch, ensuring that all changes are
  reviewed before merging.

## [0.2.0] - 2026-03-26

### Added

- Automated version management: single source of truth in `src/parameterrun/__init__.py`
- Version bump script `scripts/bump_version.py` for automatic version updates across all files
- VERSIONING.md guide for release workflows and version management
- `__version__` exported from main package: `from parameterrun import __version__`
- Dynamic version extraction in `pyproject.toml` (no more manual version updates needed)

### Changed

- `parameterrun(..., reshape=True)` now reshapes outputs to the parameter-grid dimensions automatically (including
  multi-output functions).
- Added early validation for duplicate parameter names, unknown swept parameters, unknown kwargs, and kwarg/sweep-name
  conflicts.

## [0.1.0] - 2026-03-25

### Added

- Initial public release of `parameterrun`
