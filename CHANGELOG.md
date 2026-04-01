# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

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
