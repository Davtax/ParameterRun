# Contributing

Thanks for contributing to `parameterrun`.

## Development setup

```bash
uv sync --group dev
```

If you do not use `uv`, create a virtual environment and install equivalent dev dependencies from `pyproject.toml`.

## Run quality checks

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run python -m build
uv run python -m twine check dist/*
```

## Pull request guidelines

- Keep PRs focused and small when possible.
- Add or update tests for behavior changes.
- Update `README.md` and `CHANGELOG.md` when user-facing behavior changes.
- Ensure all checks pass before requesting review.

## Release notes

- Add entries to the `[Unreleased]` section in `CHANGELOG.md`.
- At release time, move those entries under a versioned heading.

