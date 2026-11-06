# ParameterRun

`parameterrun` is a lightweight utility for running parameter sweeps in Python with a simple, function-based API.

It is designed for scientific and numerical workflows where you want to evaluate a function over one or more parameter
grids, optionally in parallel, while keeping progress bars and output reshaping convenient.

The package supports:

- single-parameter scans
- multi-parameter scans
- grouped parameters that must vary together
- parallel execution with `joblib`
- MPI execution with `mpi4py`
- automatic reshaping of outputs into arrays
- functions returning either one output or multiple outputs

## Installation

### With `pip`

```bash
pip install parameterrun
```

### With optional MPI support

```bash
pip install parameterrun[mpi]
```

## Basic idea

The main entry point is:

```python
from parameterrun import parameterrun
```

The function `parameterrun(...)` evaluates a user-defined function over a set of parameter values. The parameters to
sweep are identified by their names exactly as they appear in the target function signature. Additional fixed arguments
can be passed as keyword arguments.

## Quick start

### 1. Single parameter

```python
from parameterrun import parameterrun


def square(x):
    return x ** 2


result = parameterrun(square, param_names="x", param_values=[0, 1, 2, 3], n_workers=1, )

print(result)  # array([0, 1, 4, 9])
```

### 2. Multiple independent parameters

This creates a Cartesian product of the parameter groups.

```python
from parameterrun import parameterrun


def f(x, y, scale=1):
    return scale * (x + y)


result = parameterrun(f, param_names=["x", "y"], param_values=[[1, 2, 3], [10, 20], ], scale=2, )

print(result.shape)  # (3, 2)
```

### 3. Grouped parameters

Parameters in the same group vary together and must have the same number of values.

```python
from parameterrun import parameterrun


def f(x, y, z):
    return x + y + z


result = parameterrun(f, param_names=[["x", "y"], ["z"]], param_values=[[[1, 2], [10, 20]],  # (x, y) = (1,10), (2,20)
                                                                        [[100, 200, 300]],  # z varies independently
                                                                        ], n_workers=1, )

print(result.shape)  # (2, 3)
```

This grouped-input mode is useful when some parameters are logically linked and should not be combined independently.

## Multiple outputs

If the target function returns a tuple, parameterrun returns a list of arrays, one per output channel.

```python
from parameterrun import parameterrun


def stats(x):
    return x + 1, x ** 2


mean_like, square_like = parameterrun(stats, param_names="x", param_values=[1, 2, 3], n_workers=1, )

print(mean_like)
# array([2, 3, 4])

print(square_like)  # array([1, 4, 9])

# Ask for plain Python lists instead of numpy arrays
mean_like, square_like = parameterrun(stats, param_names="x", param_values=[1, 2, 3], n_workers=1,
                                      result_as_array=False, )

print(mean_like)  # [2, 3, 4]
```

If the function returns a single object, the result is returned as a single array when reshaping is possible.
Set `result_as_array=False` to convert outputs to Python lists.

For multi-dimensional sweeps, reshaping is automatic when `reshape=True` (default): the first dimensions always match
the parameter grid shape.

If the outputs cannot be stacked into an array (for example, arrays of different shapes), `parameterrun` emits a
`RuntimeWarning` with the reason and returns the flat list(s) of results instead. Pass `reshape="strict"` to raise a
`ValueError` instead.

If `fun` returns `None` for some points (for example, a failed solver), a `RuntimeWarning` reports how many points and
the parameters of the first one, because the result is then no longer a numeric array. Returning a NaN array of the
right shape for failed points keeps the result numeric.

## Errors in the function

`on_error` controls what happens when `fun` raises an exception at some point:

- `on_error="raise"` (default): the exception propagates with its original type, plus a note with the parameter values
  of the failing point, for example `parameterrun: raised by my_function at {'x': 2}`. Under MPI with more than one
  rank, the failing rank prints the traceback once and aborts the whole job (`comm.Abort(1)`), so the other ranks don't
  wait for it until the walltime limit.
- `on_error="nan"`: the failing point is filled with NaN of the right shape for each output channel, and the sweep
  continues. At the end, a `RuntimeWarning` reports how many points failed, with the parameters and traceback of the
  first one. Integer and boolean outputs become `float64` so they can hold NaN; floating-point and complex outputs keep
  their dtype; non-numeric outputs get `None`. If every point fails, a `RuntimeError` is raised.

```python
result = parameterrun(my_function, param_names="x", param_values=[1, 2, 3], on_error="nan", )
```

Only `Exception` subclasses are caught: `KeyboardInterrupt` and `SystemExit` always propagate.

## Validation behavior

`parameterrun` validates inputs before launching workers and raises clear `ValueError`s when:

- parameter names are duplicated
- a swept parameter does not exist in the target function signature
- extra keyword arguments do not match the target function signature
- a keyword argument conflicts with a swept parameter name

## Parallel backend

`parameterrun` supports two execution backends:

- `joblib`: a simple, local parallelization library that works well for most use cases.
- `mpi`: a distributed parallelization approach using MPI, suitable for running on clusters or supercomputers.

If `backend=None`, the package tries to detect whether MPI is available. If MPI is not available, it falls back to
`joblib`; otherwise it uses MPI.

### Joblib backend

```python
result = parameterrun(my_function, param_names=["x", "y"], param_values=[[1, 2, 3], [4, 5]], backend="joblib",
                      n_workers=-1, )
```

If `n_workers=1`, the function runs serially without spawning joblib workers.

Inside an MPI job (more than one rank), `backend="joblib"` with a negative `n_workers` makes every rank start a pool
with one worker per core, which oversubscribes the node. `parameterrun` warns in that case; pass `n_workers`
explicitly if you really run one rank per node.

### MPI backend

```python
result = parameterrun(my_function, param_names=["x", "y"], param_values=[[1, 2, 3], [4, 5]], backend="mpi", )
```

Run the script with MPI, for example:

```bash
mpirun -n 4 python my_script.py
```

Under MPI, only rank 0 returns the final result; the other ranks return `None`.

The points are distributed among ranks by a pseudo-random permutation. Pass `shuffle_seed` to make that distribution
reproducible:

```python
result = parameterrun(my_function, param_names="x", param_values=[1, 2, 3], backend="mpi", shuffle_seed=1234, )
```

`shuffle_seed` is a parameter of `parameterrun` itself. A keyword argument called `seed` is still forwarded to `fun`.

#### Static and dynamic schedules

`mpi_schedule` chooses how the points are distributed among the ranks:

- `"static"`: every rank computes a fixed share of the points, and rank 0 gathers the results at the end. All ranks
  compute, but the job lasts as long as its slowest rank, and differences in speed between nodes and cores are common
  on clusters.
- `"dynamic"`: rank 0 becomes a manager that hands out chunks of points on demand, and the other ranks compute. Fast
  ranks simply ask for more work, so no rank waits at the end. Numeric outputs are written straight into preallocated
  arrays on rank 0, and the progress bar counts every point. This costs one rank, and needs at least 2 (with 1 rank,
  the static schedule runs).
- `"auto"` (default): dynamic with 8 ranks or more, static otherwise.

Both schedules give identical results. With the dynamic schedule, chunks are guided by default: large at first (few
messages) and shrinking to one point at the end. For very cheap points, set a fixed size to cut the number of messages:

```python
result = parameterrun(my_function, param_names="x", param_values=range(10**6), backend="mpi",
                      mpi_schedule="dynamic", mpi_chunk_size=256, )
```

`shuffle_seed` also fixes the order in which the dynamic schedule hands out the points.

#### Which ranks get the result

By default only rank 0 returns the result. If every rank needs it, use `return_on="all"` instead of a `comm.bcast` of
the result afterwards: numeric arrays are broadcast as raw bytes, without pickling and without its 2 GB limit.

```python
result = parameterrun(my_function, param_names="x", param_values=[1, 2, 3], backend="mpi", return_on="all", )
```

If rank 0 fails while finishing the result (for example with `reshape="strict"`), every rank raises instead of waiting.

## Saving the result to a file

`result_file` writes the result, reshaped to the grid, to a `.npy` file on rank 0. With several outputs there is one
file per output: `result.npy` becomes `result_0.npy`, `result_1.npy`, ... Existing files are overwritten.

```python
result = parameterrun(my_function, param_names=["x", "y"], param_values=[xs, ys], result_file="sweep.npy", )
data = np.load("sweep.npy")  # Same as result
```

- Under MPI, numeric outputs are written to the file while the points arrive (a memory-mapped array), so the size of
  the result is limited by the disk, not by rank 0's memory. The returned arrays then read from the file,
  copy-on-write: changing them never changes the file.
- With `return_on="none"`, every rank returns `None` and only the file is written. Useful for results larger than
  memory.
- Under joblib, the files are written at the end.
- Outputs that cannot form one array without Python objects (ragged arrays, `None`, ...) are not written, with a
  `RuntimeWarning`. String outputs are written as a NumPy string array.
- The directory is checked before anything is computed.

## Progress bars and verbosity

By default (`pbar_bool=None`), the progress bar is shown when the output is a terminal or a Jupyter notebook, and hidden
otherwise (for example, in a Slurm log). The `TQDM_DISABLE` environment variable overrides this default. Force the bar
on or off with `pbar_bool=True` or `pbar_bool=False`:

```python
result = parameterrun(my_function, param_names="x", param_values=[1, 2, 3], pbar_bool=True, )
```

When a bar is forced on and the output is not a terminal, it prints one plain line per refresh instead of carriage-return
redraws, every 30 s by default. Set `TQDM_MININTERVAL` to change the interval.

Verbose logging can be enabled with:

```python
result = parameterrun(my_function, param_names="x", param_values=[1, 2, 3], verbose=True, )
```

The progress-bar description is generated automatically from the function name and parameter names, unless you provide
`desc` explicitly.

## API reference

`parameterrun`

```python
parameterrun(fun, param_names, param_values, n_workers=-1, pbar_bool=None, pbar_leave=True, verbose=False,
             reshape=True, result_as_array=True, backend=None, desc=None, shuffle_seed=None, on_error="raise",
             mpi_schedule="auto", mpi_chunk_size=None, return_on="root", result_file=None, **kwargs, )
```

* `fun`: function to evaluate
* `param_names`: parameter name or grouped parameter names
* `param_values`: values corresponding to `param_names` (any iterable, e.g. list, tuple, range, numpy array)
* `n_workers`: number of workers for `joblib`
* `pbar_bool`: `True`/`False` to show/hide progress bars; `None` shows them only on a terminal or in a notebook
* `pbar_leave`: keep the progress bar after completion
* `verbose`: print backend and timing information
* `reshape`: reshape results into arrays when possible (warns otherwise); `"strict"` raises instead; `False` returns
  the flat lists
* `result_as_array`: return numpy arrays by default; if `False`, convert outputs to Python lists
* `backend`: `"joblib"`, `"mpi"`, or `None`
* `desc`: custom progress-bar label
* `shuffle_seed`: MPI only, seed of the permutation that distributes the points among ranks
* `on_error`: `"raise"` (default; aborts the whole MPI job) or `"nan"` (fill failed points with NaN and warn)
* `mpi_schedule`: MPI only, `"auto"` (default), `"static"` or `"dynamic"`
* `mpi_chunk_size`: dynamic MPI schedule only, points per chunk (`None`: guided chunks)
* `return_on`: MPI ranks that get the result: `"root"` (default), `"all"` or `"none"` (needs `result_file`)
* `result_file`: `.npy` path to write the result to (one file per output channel)
* `**kwargs`: extra fixed keyword arguments forwarded to `fun`. They cannot reuse the names of the parameters above.

## Input formats

`parameterrun` accepts three input styles.

At every style, value containers can be any iterable (for example lists, tuples, ranges, generators, or numpy
arrays).

### A. Single parameter

```python
param_names = "x"
param_values = [1, 2, 3]
```

### B. Multiple independent parameters

```python
param_names = ["x", "y"]
param_values = [[1, 2, 3], [10, 20], ]
```

### C. Multiple groups of parameters

```python
param_names = [["x", "y"], ["z"]]
param_values = [[[1, 2], [10, 20]], [[100, 200, 300]], ]
```

## Development checks

Before opening a PR or publishing a release, run:

```bash
pytest -m "not mpi"
ruff check .
mypy
python -m build
python -m twine check dist/*
```

For MPI numerical-equivalence checks (Linux/OpenMPI, 4 ranks):

```bash
uv sync --extra mpi
mpirun -n 4 uv run pytest -m mpi -o addopts="-q -m mpi"
```

