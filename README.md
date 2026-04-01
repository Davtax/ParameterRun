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

### MPI backend

```python
result = parameterrun(my_function, param_names=["x", "y"], param_values=[[1, 2, 3], [4, 5]], backend="mpi", )
```

Run the script with MPI, for example:

```bash
mpirun -n 4 python my_script.py
```

Under MPI, only rank 0 returns the final result; the other ranks return `None`.

## Progress bars and verbosity

By default, progress bars are enabled:

```python
result = parameterrun(my_function, param_names="x", param_values=[1, 2, 3], pbar_bool=True, )
```

Verbose logging can be enabled with:

```python
result = parameterrun(my_function, param_names="x", param_values=[1, 2, 3], verbose=True, )
```

The progress-bar description is generated automatically from the function name and parameter names, unless you provide
`desc` explicitly.

## API reference

`parameterrun`

```python
parameterrun(fun, param_names, param_values, n_workers=-1, pbar_bool=True, verbose=False,
             reshape=True, result_as_array=True, backend=None, desc=None, **kwargs, )
```

* `fun`: function to evaluate
* `param_names`: parameter name or grouped parameter names
* `param_values`: values corresponding to `param_names` (any iterable, e.g. list, tuple, range, numpy array)
* `n_workers`: number of workers for `joblib`
* `pbar_bool`: enable or disable progress bars
* `verbose`: print backend and timing information
* `reshape`: reshape results into arrays when possible
* `result_as_array`: return numpy arrays by default; if `False`, convert outputs to Python lists
* `backend`: `"joblib"`, `"mpi"`, or `None`
* `desc`: custom progress-bar label
* `**kwargs`: extra fixed keyword arguments forwarded to `fun`

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
pytest
ruff check .
mypy
python -m build
python -m twine check dist/*
```
