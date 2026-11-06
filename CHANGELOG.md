# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [0.5.0] - 2026-11-06

### Added

- `shuffle_seed` parameter: seeds the permutation that distributes points among MPI ranks, making the distribution
  reproducible. It was accepted by the MPI backend but never passed through. A keyword argument named `seed` is still
  forwarded to the user function.
- `reshape="strict"` raises a `ValueError` when the result cannot be reshaped to the parameter grid.
- `RuntimeWarning` when some points return `None`, with the count and the parameters of the first one.
- `RuntimeWarning` when `backend="joblib"` with a negative `n_workers` runs inside an MPI job with more than one rank
  (every rank would start a full-node pool).
- `on_error` parameter (`"raise"` by default, or `"nan"`) for exceptions raised by `fun`.
  - `"raise"`: the exception keeps its type and gets the parameter values of the failing point as a note. Under MPI,
    the failing rank prints the traceback once and calls `comm.Abort(1)`, instead of leaving the other ranks waiting
    until the walltime limit.
  - `"nan"`: failed points are filled with NaN of the right shape per output channel, and a `RuntimeWarning` reports
    how many points failed and the traceback of the first one. If every point fails, a `RuntimeError` is raised.
- Dynamic MPI schedule (`mpi_schedule="dynamic"`): rank 0 becomes a manager that hands out chunks of points on demand,
  and the other ranks compute. Fast ranks get more work, so the job no longer waits for the slowest rank. Numeric
  outputs go straight into preallocated arrays on rank 0, the progress bar counts every point (not rank 0's share),
  and with `on_error="raise"` the manager prints the failing point's traceback once and aborts the job. Results are
  identical to the static schedule.
- `mpi_schedule` parameter: `"auto"` (default; dynamic with 8 ranks or more, static below), `"static"` or `"dynamic"`.
- `mpi_chunk_size` parameter: points per chunk for the dynamic schedule. The default (None) uses guided chunks that
  shrink to one point at the end.
- `return_on` parameter: `"root"` (default), `"all"` (every MPI rank gets the result; numeric arrays are broadcast as
  raw bytes, not pickled) or `"none"` (only the result file is written).
- `result_file` parameter: writes the result, reshaped to the grid, to a `.npy` file (one per output channel). Under
  MPI, numeric outputs go into a memory-mapped file while the points arrive, so the result is no longer limited by
  rank 0's memory.
- Checkpoints for long runs: `checkpoint` (a directory), `checkpoint_every` (points), `checkpoint_interval` (seconds,
  600 by default), `checkpoint_resume` (`"auto"`, `"force"`, `"restart"`) and `keep_checkpoint`. The finished points
  are saved regularly, and calling `parameterrun` again with the same directory computes only the missing points, with
  any backend, MPI schedule or number of ranks. Resuming checks the parameters, the extra keyword arguments and the code
  of `fun` (bytecode, constants, defaults, closures, and helper functions of its module). The static MPI schedule
  works in synchronised rounds between saves; the dynamic schedule saves without stopping the workers. `SIGUSR1`
  saves at once (e.g. Slurm's `--signal=USR1@600`); `SIGTERM` saves, then ends the run. Snapshots are `.npz` files:
  numeric outputs are stored without pickle, and outputs already in a memory-mapped `result_file` are not copied.

### Changed

- `on_error`, `mpi_schedule`, `mpi_chunk_size`, `return_on`, `result_file` and the `checkpoint*` options are now
  parameters of `parameterrun` itself, so keyword arguments with these names are no longer forwarded to `fun`.
- MPI, static schedule: results are no longer gathered as one pickled message per output channel. Numeric outputs are
  sent as raw bytes in blocks of at most 64 MiB straight into preallocated arrays on rank 0. This removes the 2 GB
  limit of pickled messages (`MPI_ERR_ARG` on large results) and most of rank 0's extra copies; non-numeric outputs
  are still pickled. Both schedules now share this result store, so they return the same arrays.
- MPI: every call uses its own duplicated communicator, so messages of one call can never reach another.
- MPI runs with 8 ranks or more now use the dynamic schedule by default, which leaves rank 0 to the manager. Pass
  `mpi_schedule="static"` for the previous behaviour.
- **Breaking:** dropped support for Python 3.9 and 3.10; `parameterrun` now requires Python >= 3.11.
- Added support for Python 3.13, 3.14 and 3.15 (classifiers and CI test matrix).
- A failed reshape now emits a `RuntimeWarning` that explains why, instead of printing "Could not reshape the result".
- `pbar_bool` now defaults to `None`: progress bars are shown on a terminal or in a notebook, hidden otherwise, and
  `TQDM_DISABLE` is honoured. Forced bars on a non-terminal output (e.g. Slurm logs) print one plain line per refresh,
  every 30 s by default (`TQDM_MININTERVAL` overrides it).
- MPI: rank 0 now waits for the workers with a blocking receive instead of a busy `Iprobe` loop.
- MPI: results are sorted with `np.argsort` instead of Python's `sorted`.

### Removed

- The Open MPI 5.0.6 version-string workaround for the progress bar. Bars on MPI ranks whose stdout is a terminal now
  end each refresh with a newline, so MPI launchers that forward output line by line show them with any MPI version.

## [0.4.1] - 2026-04-21

### Fixed

- Fix the display of the progress bar when running with `backend="joblib"`, since the previous implementation didn't
  have access to the total number of runs to compute, which is required for the progress bar to work properly.

### Added

- Added a test for the progress bar display when running with `backend="joblib"`.

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
