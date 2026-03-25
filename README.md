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

result = result.reshape(3, 2)
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

result = result.reshape(2, 3)
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
```

If the function returns a single object, the result is returned as a single array when reshaping is possible.

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

You can also pass extra keyword arguments to `tqdm` through `pbar_kwargs`:

```python
result = parameterrun(my_function, param_names="x", param_values=[1, 2, 3], pbar_kwargs={"leave": False}, )
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
parameterrun(fun, param_names, param_values, n_workers=-1, pbar_bool=True, verbose=False, pbar_kwargs=None,
    reshape=True, backend=None, desc=None, **kwargs, )
```

* `fun`: function to evaluate
* `param_names`: parameter name or grouped parameter names
* `param_values`: values corresponding to `param_names`
* `n_workers`: number of workers for `joblib`
* `pbar_bool`: enable or disable progress bars
* `verbose`: print backend and timing information
* `pbar_kwargs`: additional keyword arguments passed to tqdm
* `reshape`: reshape results into arrays when possible
* `backend`: `"joblib"`, `"mpi"`, or `None`
* `desc`: custom progress-bar label
* `**kwargs`: extra fixed keyword arguments forwarded to `fun`

## Input formats

`parameterrun` accepts three input styles.

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
mypy src/parameterrun
python -m build
python -m twine check dist/*
```
