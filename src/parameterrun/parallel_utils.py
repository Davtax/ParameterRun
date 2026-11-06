"""The public entry point, `parameterrun`: input handling, backend selection, execution and post-processing."""

import os
import sys
import warnings
from collections.abc import Callable, Iterable
from itertools import product
from pathlib import Path
from time import time
from typing import Any, Literal

import numpy as np

from ._checkpoint import CheckpointResume, _check_checkpoint_options, _Checkpoint, _save_triggers
from ._delivery import _RETURN_ON, _broadcast_result, _check_result_file, _write_result_files
from ._errors import _ON_ERROR_MODES, _fill_failed_points
from ._inputs import _format_input, _validate_function_arguments
from ._joblib_backend import _parameterrun_joblib, _parameterrun_joblib_checkpointed
from ._mpi_dynamic import _parameterrun_mpi_dynamic
from ._mpi_static import _parameterrun_mpi, _parameterrun_mpi_rounds
from ._progress import _resolve_pbar_bool
from ._results import _as_flat_lists, _convert_result_to_lists, _reshape_channel_result, _warn_none_outputs
from ._utils import _callable_name, _log, _mpi_world_size_from_env, _normalized_time

_MPI_SCHEDULES = ("auto", "static", "dynamic")

# mpi_schedule="auto" picks the dynamic schedule from this many ranks on. Below it, giving up one rank to the manager
# costs more than the load balancing gains (and static is what CI's 4-rank runs exercise by default)
_DYNAMIC_MIN_RANKS = 8


def _resolve_mpi_schedule(mpi_schedule: str, size: int) -> str:
    if size < 2:  # The dynamic schedule needs a manager and at least one worker
        return "static"
    if mpi_schedule == "auto":
        return "dynamic" if size >= _DYNAMIC_MIN_RANKS else "static"
    return mpi_schedule


def parameterrun(
    fun: Callable[..., Any],
    param_names: str | list[str] | list[list[str]],
    param_values: Iterable[Any] | Iterable[Iterable[Any]] | Iterable[Iterable[Iterable[Any]]],
    n_workers: int | None = -1,
    pbar_bool: bool | None = None,
    pbar_leave: bool = True,
    verbose: bool | None = False,
    reshape: bool | Literal["strict"] = True,
    result_as_array: bool = True,
    backend: str | None = None,
    desc: str | None = None,
    shuffle_seed: int | None = None,
    on_error: Literal["raise", "nan"] = "raise",
    mpi_schedule: Literal["auto", "static", "dynamic"] = "auto",
    mpi_chunk_size: int | None = None,
    return_on: Literal["root", "all", "none"] = "root",
    result_file: str | os.PathLike[str] | None = None,
    checkpoint: str | os.PathLike[str] | None = None,
    checkpoint_every: int | None = None,
    checkpoint_interval: float | None = None,
    checkpoint_resume: CheckpointResume = "auto",
    keep_checkpoint: bool = True,
    **fun_kwargs: Any,
) -> list[Any] | np.ndarray | None:
    """
    Run a function with multiple parameters in parallel. To indentify the parameters of interest, user must provide its
    name as is written in the function definition. If more parameters should be pass to the function, they can be
    provided as kwargs. The backend used for the parallelization is chosen automatically between joblib and mpi4py.
    If mpi4py is used, the function must be run with mpirun -n n_workers python script.py. Finally, the result can be
    reshaped as a (hyper)matrix if the reshape parameter is set to True, if the functions return multiple values,
    the result is a list of arrays.

    The parameters can be grouped in different ways, the possible options are:
    - A single parameter: The function is run with a single parameter. In this case, param_names='param_name' and
    param_values=[value1, value2, ...]
    - Multiple parameters: The function is run with multiple parameters. In this case, param_names=['param_name1',
    'param_name2', ...] and param_values=[[value1, value2, ...], [value1, value2, ...], ...]
    - Multiple groups of parameters: The function is run with multiple groups of parameters, running the parameters in
    the same group at the same time. In this case, param_names=[['param_name1', 'param_name2', ...], ['param_name1',
    'param_name2', ...], ...] and param_values=[[[value1, value2, ...], [value1, value2, ...], ...], [[value1, value2,
    ...], [value1, value2, ...], ...], ...]. Note that in this case, the number of values for each parameter in the same
    group must be the same.

    Parameters
    ----------
    fun : callable
        Function to run in parallel.
    param_names : str or list
        Name of the parameters to run in parallel.
    param_values : iterable
        Values of the parameters to run in parallel. Lists, tuples, ranges, generators, and numpy arrays are
        accepted as long as they follow one of the supported input shapes.
    n_workers : int, optional (default=-1)
        Number of workers to use in parallel. If -1, the number of workers is the maximum number of cores available.
    pbar_bool : bool, optional (default=None)
        If True, show a progress bar; if False, hide it. If None, honour the TQDM_DISABLE environment variable, and
        otherwise show the bar only when the output is a terminal or a notebook. When the output is not a terminal
        (e.g. a Slurm log), the bar prints one plain line per refresh, every 30 s by default (TQDM_MININTERVAL
        overrides it).
    pbar_leave : bool, optional (default=True)
        If True, leave the progress bar after completion.
    backend : str, optional (default=None)
        Backend to use for the parallelization. If None, the backend is chosen automatically between joblib and mpi.
    verbose : bool, optional (default=False)
        If True, print information about the parallelization.
    reshape : bool or "strict", optional (default=True)
        If True, reshape the result as a (hyper)matrix. If that is not possible (e.g. outputs of different shapes), a
        RuntimeWarning explains why and the flat list(s) of results are returned instead. With "strict", a ValueError
        is raised instead. If False, the flat list(s) are returned.
    result_as_array : bool, optional (default=True)
        If True, return numpy arrays when possible (default behavior). If False, convert numpy-array outputs to Python
        lists.
    desc: str, optional (default=None)
        Description of the progress bar. If None, the description denotes the function and parameters names.
    shuffle_seed: int, optional (default=None)
        MPI backend only: seed of the permutation that distributes the points among ranks. Pass it to make the
        assignment reproducible; if None, a random seed is drawn on rank 0. Ignored by the joblib backend.
    on_error: "raise" or "nan", optional (default="raise")
        What to do when `fun` raises an exception at some point.
        - "raise": the exception propagates with the parameter values of the failing point added as a note (its type is
          kept). Under MPI with more than one rank, the failing rank prints the traceback once and aborts the whole MPI
          job (`comm.Abort(1)`), so the other ranks don't wait until the walltime limit.
        - "nan": the point is filled with NaN of the right shape per output channel (integer and boolean outputs become
          floating point, non-numeric outputs get None), and a RuntimeWarning reports the number of failed points and
          the traceback of the first one. If every point fails, a RuntimeError is raised.
        Only `Exception` subclasses are caught; e.g. KeyboardInterrupt always propagates.
    mpi_schedule: "auto", "static" or "dynamic", optional (default="auto")
        MPI backend only: how the points are distributed among ranks.
        - "static": every rank computes a fixed share of the points (a seeded pseudo-random permutation), and the
          results are gathered at the end. All ranks compute.
        - "dynamic": rank 0 is a manager that hands out chunks of points on demand, and the other ranks compute. Fast
          ranks get more work, so the job no longer waits for the slowest rank; results go straight into
          preallocated arrays on rank 0. Costs one rank, and needs at least 2 (with 1 rank, "static" runs).
        - "auto": "dynamic" with 8 ranks or more, "static" otherwise.
    mpi_chunk_size: int, optional (default=None)
        Dynamic MPI schedule only: number of points per chunk. If None, chunks are guided: a quarter of each worker's
        share of the remaining points, so they shrink to one point at the end. Set it for very cheap points, to cut
        the number of messages.
    return_on: "root", "all" or "none", optional (default="root")
        Which MPI ranks get the result. "root": rank 0 only, the other ranks get None. "all": every rank gets the same
        result; numeric arrays are broadcast as raw bytes (no pickling), which is much cheaper than a `comm.bcast` of
        the result afterwards. "none": every rank gets None; requires `result_file`. Under joblib, "all" is the same
        as "root".
    result_file: str or path, optional (default=None)
        Write the result to this `.npy` file (on rank 0), reshaped to the grid: one file per output channel, so
        `result.npy` becomes `result_0.npy`, `result_1.npy`, ... for several outputs. Existing files are overwritten.
        Under MPI, numeric outputs are written to the file while the points arrive (a memory-mapped array), so the
        size of the result is limited by the disk, not by rank 0's memory; the returned arrays then read from the file
        (copy-on-write: changing them never changes the file). Under joblib, the files are written at the end. Outputs
        that cannot form one array without Python objects (e.g. ragged arrays, None) are not written, with a
        RuntimeWarning.
    checkpoint: str or path, optional (default=None)
        Directory for snapshots of a long run. The finished points are saved there regularly (by rank 0), and calling
        `parameterrun` again with the same directory resumes from the last snapshot: only the missing points are
        computed. A snapshot records which points are done, so it can be resumed with another backend, MPI schedule or
        number of ranks. The parameter names and values and `fun_kwargs` must be the same, otherwise a ValueError is
        raised. Failed points (`on_error="nan"`) count as done. With the static MPI schedule, the ranks work in rounds
        between snapshots, and fast ranks wait for the slowest one at the end of each round; the dynamic schedule
        saves without stopping the workers. Under joblib, and on the dynamic manager, the points finished before `fun`
        raised are saved before the exception propagates. SIGUSR1 saves at once and goes on (e.g. with Slurm's
        `--signal=USR1@600`); SIGTERM saves at once, then ends the run (the whole MPI job) as the signal would have.
        Numeric outputs are stored without pickle, and outputs that live in the memory-mapped `result_file` are not
        copied into the snapshot (resuming then needs the same `result_file`). Non-numeric outputs are pickled: load
        snapshots only from sources you trust.
    checkpoint_every: int, optional (default=None)
        Save after this many points (over all workers or ranks).
    checkpoint_interval: float, optional (default=None)
        Save after this many seconds. If neither `checkpoint_every` nor `checkpoint_interval` is given, the run is
        saved every 600 seconds. With both, whichever comes first.
    checkpoint_resume: "auto", "force" or "restart", optional (default="auto")
        - "auto": resume if `checkpoint` holds a snapshot of the same run. Raise a ValueError if the code of `fun`
          changed since: its bytecode, constants, default values, closure variables, and the functions of its own
          module that it calls. Changes in other modules or libraries cannot be detected.
        - "force": resume even if the code of `fun` changed, with a RuntimeWarning (e.g. after a harmless edit).
        - "restart": discard the snapshot and start over.
    keep_checkpoint: bool, optional (default=True)
        Keep the snapshot after a successful run (calling `parameterrun` again then returns the result at once). If
        False, it is deleted once the result is ready.
    fun_kwargs :
        Additional parameters to pass to the function. They cannot use the names of the parameters of
        `parameterrun` itself (`fun`, `param_names`, `param_values`, `n_workers`, `pbar_bool`, `pbar_leave`,
        `verbose`, `reshape`, `result_as_array`, `backend`, `desc`, `shuffle_seed`, `on_error`, `mpi_schedule`,
        `mpi_chunk_size`, `return_on`, `result_file`, `checkpoint`, `checkpoint_every`, `checkpoint_interval`,
        `checkpoint_resume`, `keep_checkpoint`). Any other name, including `seed`, is passed to the function.

    Returns
    -------
    result : list or np.ndarray
        Result of the function run in parallel. The shape of the (hyper)matrix is (n_values1, n_values2, ...,
         n_valuesN), where n_valuesI is the number of values in the I-th group.
    """

    time_start = time()

    if reshape not in (True, False, "strict"):
        raise ValueError(f'reshape must be True, False or "strict", got {reshape!r}')
    if on_error not in _ON_ERROR_MODES:
        raise ValueError(f'on_error must be "raise" or "nan", got {on_error!r}')
    if mpi_schedule not in _MPI_SCHEDULES:
        raise ValueError(f'mpi_schedule must be "auto", "static" or "dynamic", got {mpi_schedule!r}')
    if mpi_chunk_size is not None and (
        isinstance(mpi_chunk_size, bool) or not isinstance(mpi_chunk_size, int) or mpi_chunk_size < 1
    ):
        raise ValueError(f"mpi_chunk_size must be a positive integer or None, got {mpi_chunk_size!r}")
    if return_on not in _RETURN_ON:
        raise ValueError(f'return_on must be "root", "all" or "none", got {return_on!r}')
    if return_on == "none" and result_file is None:
        raise ValueError('return_on="none" needs a result_file, otherwise the result would be lost')
    if result_file is not None and not Path(result_file).suffix == ".npy":
        raise ValueError(f"result_file must end in .npy, got {str(result_file)!r}")
    _check_checkpoint_options(checkpoint_every, checkpoint_interval, checkpoint_resume)

    formatted_param_names, formatted_param_values = _format_input(param_names, param_values)
    _validate_function_arguments(fun, formatted_param_names, fun_kwargs)

    # Create the list of dictionaries with the parameters in a nested loop
    n_values = [len(group[0]) for group in formatted_param_values]
    if any(n_value == 0 for n_value in n_values):
        raise ValueError("Parameter groups must contain at least one value.")

    # Choose the backed
    if backend is not None:
        available_backends = ["joblib", "mpi"]
        if backend not in available_backends:
            raise ValueError(f"Unknown backend {backend}. Available backends are {available_backends}")

    size = 1
    rank = 0

    mpi_module = None
    if backend is None or backend == "mpi":
        try:
            from mpi4py import MPI

            mpi_module = MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
        except ImportError as err:
            if backend == "mpi":
                raise ImportError('backend="mpi" requires mpi4py to be installed and available.') from err

        if size == 1 and backend is None:
            backend = "joblib"
        else:
            backend = "mpi"

    if backend == "joblib" and n_workers is not None and n_workers < 0:
        mpi_size = _mpi_world_size_from_env()
        if mpi_size > 1:
            warnings.warn(
                f'backend="joblib" with n_workers={n_workers} inside an MPI job of {mpi_size} ranks: every rank starts '
                f"a pool with one worker per core, which oversubscribes the node. Use the MPI backend, or pass "
                f"n_workers explicitly (e.g. int(os.environ['SLURM_CPUS_PER_TASK'])) if you run one rank per node.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Check the result file before computing anything. Under MPI only rank 0 writes it, and every rank must agree
    if result_file is not None:
        problem = _check_result_file(result_file) if rank == 0 else None
        if backend == "mpi" and size > 1:
            problem = comm.bcast(problem, root=0)
        if problem is not None:
            raise ValueError(problem)

    # Open the checkpoint on rank 0 (resuming from it if it holds this run). Every rank raises if that fails
    snapshot: _Checkpoint | None = None
    if checkpoint is not None:
        open_error: Exception | None = None
        if rank == 0:
            try:
                snapshot = _Checkpoint.open(
                    checkpoint,
                    fun,
                    formatted_param_names,
                    formatted_param_values,
                    fun_kwargs,
                    result_file=result_file,
                    every=checkpoint_every,
                    interval=checkpoint_interval,
                    resume=checkpoint_resume,
                )
            except (ValueError, OSError) as err:
                open_error = err
        if backend == "mpi" and size > 1:
            message = comm.bcast(None if open_error is None else str(open_error), root=0)
            if message is not None and open_error is None:
                raise ValueError(message)
        if open_error is not None:
            raise open_error

    # tqdm writes to stderr under joblib, and the MPI backend writes to stdout
    pbar_bool = _resolve_pbar_bool(pbar_bool, sys.stdout if backend == "mpi" else sys.stderr)

    if desc is None:
        param_names_pbar = [parameters_names_i for parameters_names_i in formatted_param_names]
        auto_desc = f"{_callable_name(fun)}: {param_names_pbar}"

        if len(auto_desc) > 40:
            auto_desc = auto_desc[:40] + "(...)]"

        desc = auto_desc

    # Execute the parallel run
    result: list[Any]  # One flat list or array per output channel

    if backend == "joblib":
        _log("Running under joblib", verbose)

        if snapshot is not None:
            result, n_outputs = _parameterrun_joblib_checkpointed(
                fun,
                formatted_param_names,
                formatted_param_values,
                desc,
                snapshot,
                n_workers=n_workers,
                pbar_bool=pbar_bool,
                pbar_leave=pbar_leave,
                on_error=on_error,
                verbose=verbose,
                fun_kwargs=fun_kwargs,
            )
        else:
            indices = [list(range(n_value)) for n_value in n_values]
            result, n_outputs = _parameterrun_joblib(
                fun,
                formatted_param_names,
                formatted_param_values,
                product(*indices),
                desc,
                n_workers=n_workers,
                pbar_bool=pbar_bool,
                pbar_leave=pbar_leave,
                on_error=on_error,
                **fun_kwargs,
            )

    elif backend == "mpi":
        schedule = _resolve_mpi_schedule(mpi_schedule, size)
        _log(
            f"Running under mpi with {size} ranks, {schedule} schedule",
            verbose and rank == 0,
            hostname=mpi_module.Get_processor_name() if mpi_module is not None else None,
        )

        common: dict[str, Any] = {
            "pbar_bool": pbar_bool,
            "pbar_leave": pbar_leave,
            "verbose": verbose,
            "seed": shuffle_seed,
            "fun_kwargs": fun_kwargs,
            "on_error": on_error,
        }
        if schedule == "dynamic":
            result, n_outputs = _parameterrun_mpi_dynamic(
                fun,
                formatted_param_names,
                formatted_param_values,
                desc,
                chunk_size=mpi_chunk_size,
                result_file=result_file,
                checkpoint=snapshot,
                checkpointed=checkpoint is not None,
                **common,
            )
        elif checkpoint is not None:  # Static schedule in rounds. Every rank knows; only rank 0 holds the snapshot
            result, n_outputs = _parameterrun_mpi_rounds(
                fun,
                formatted_param_names,
                formatted_param_values,
                desc,
                snapshot,
                *_save_triggers(checkpoint_every, checkpoint_interval),
                **common,
            )
        else:
            result, n_outputs = _parameterrun_mpi(
                fun, formatted_param_names, formatted_param_values, desc, result_file=result_file, **common
            )

    else:
        raise ValueError("Unknown backend")

    final_result: list[Any] | np.ndarray | None = None
    root_error: BaseException | None = None
    if rank == 0:
        try:
            final_result = _finish_on_root(
                result, n_outputs, n_values, formatted_param_names, formatted_param_values, reshape, result_file
            )
        except Exception as err:
            if not (return_on == "all" and size > 1):
                raise
            root_error = err  # Re-raised by _broadcast_result, after telling the other ranks
        else:
            if snapshot is not None and not keep_checkpoint:
                snapshot.remove()

        _log(f"Total time: {_normalized_time(time() - time_start)}", verbose)

    if return_on == "all" and size > 1:
        final_result = _broadcast_result(comm, final_result, root_error)
    elif rank != 0 or return_on == "none":
        return None
    if not result_as_array:
        final_result = _convert_result_to_lists(final_result)
    return final_result


def _finish_on_root(
    result: list[Any],
    n_outputs: int,
    n_values: list[int],
    param_names: list[list[str]],
    param_values: list[list[list[Any]]],
    reshape: bool | Literal["strict"],
    result_file: str | os.PathLike[str] | None,
) -> list[Any] | np.ndarray:
    """Post-processing on rank 0: NaN filling, checks, result files and reshaping (warnings point at the caller)."""
    _fill_failed_points(result, param_names, param_values)
    _warn_none_outputs(result, param_names, param_values)

    if result_file is not None:
        _write_result_files(result, n_values, result_file)

    # Reshape the result to get the (hyper)matrix
    if reshape:
        channel = 0
        try:
            reshaped = []
            for channel in range(n_outputs):
                reshaped.append(_reshape_channel_result(result[channel], n_values))
            return reshaped[0] if n_outputs == 1 else reshaped
        except ValueError as err:
            message = f"Could not reshape output channel {channel} to the grid shape {tuple(n_values)}: {err}"
            if reshape == "strict":
                raise ValueError(message) from err
            warnings.warn(
                f"{message}. Returning the flat list(s) of results instead of arrays.", RuntimeWarning, stacklevel=3
            )

    return _as_flat_lists(result)  # Not reshaped: one flat list per channel, whatever the backend
