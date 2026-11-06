"""Exceptions raised by the user function: the on_error="raise" and on_error="nan" modes."""

import sys
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._inputs import _params_at
from ._utils import _callable_name

_ON_ERROR_MODES = ("raise", "nan")


@dataclass(frozen=True)
class _FailedPoint:
    """
    Placeholder for a point whose call to ``fun`` raised with ``on_error="nan"``. Defined at module level so it can be
    pickled between joblib and MPI processes. Rank 0 replaces it by NaN once the output shapes are known.
    """

    summary: str  # Exception type and message, e.g. "ValueError: solver diverged"
    traceback: str


def _evaluate_point(fun: Callable[..., Any], params: dict[str, Any], fun_kwargs: dict[str, Any], on_error: str) -> Any:
    """
    Call ``fun`` at one grid point. If it raises, ``"raise"`` adds the parameter values to the exception as a note
    (keeping its type), and ``"nan"`` returns a ``_FailedPoint`` instead.
    """
    try:
        return fun(**{**params, **fun_kwargs})
    except Exception as err:
        if on_error == "raise":
            err.add_note(f"parameterrun: raised by {_callable_name(fun)} at {params}")
            raise
        return _FailedPoint(
            summary="".join(traceback.format_exception_only(err)).strip(),
            traceback="".join(traceback.format_exception(err)),
        )


def _n_outputs_of_first_success(raw: list[Any]) -> int:
    """Number of output channels of the first point that did not fail, or 0 if every point failed."""
    for value in raw:
        if not isinstance(value, _FailedPoint):
            return len(value) if isinstance(value, tuple) else 1
    return 0


def _split_channels(raw: list[Any], n_outputs: int) -> list[list[Any]]:
    """
    Turn the per-point returns into one list per output channel. A ``_FailedPoint`` goes into every channel.
    """
    channels: list[list[Any]] = [[] for _ in range(n_outputs)]
    for value in raw:
        if isinstance(value, _FailedPoint):
            for channel in channels:
                channel.append(value)
        else:
            values = value if isinstance(value, tuple) else (value,)
            for channel, channel_value in zip(channels, values, strict=True):
                channel.append(channel_value)
    return channels


def _all_points_failed_error(
    n_total: int, failure: _FailedPoint | None = None, params: dict[str, Any] | None = None
) -> RuntimeError:
    message = f'All {n_total} points raised an exception (on_error="nan"), so the output shapes are unknown.'
    if failure is not None:
        message += f" Failure at {params}:\n{failure.traceback}"
    return RuntimeError(message)


def _nan_like(template: Any) -> Any:
    """
    NaN with the shape of ``template``. Floating-point and complex outputs keep their dtype; integer and boolean outputs
    become float64. Non-numeric outputs get None.
    """
    array = np.asarray(template)
    if array.dtype.kind not in "biufc":
        return None
    dtype = array.dtype if array.dtype.kind in "fc" else np.dtype(np.float64)
    filled = np.full(array.shape, np.nan, dtype=dtype)
    return filled[()] if filled.ndim == 0 else filled


def _warn_failed_points(
    n_failed: int,
    n_total: int,
    first_index: int,
    first_failure: _FailedPoint,
    param_names: list[list[str]],
    param_values: list[list[list[Any]]],
) -> None:
    warnings.warn(
        f'{n_failed} of {n_total} points raised an exception and were filled with NaN (on_error="nan"). '
        f"First failure at {_params_at(first_index, param_names, param_values)}: {first_failure.summary}\n"
        f"{first_failure.traceback}",
        RuntimeWarning,
        stacklevel=4,
    )


def _fill_failed_points(result: list[Any], param_names: list[list[str]], param_values: list[list[list[Any]]]) -> None:
    """
    Replace every ``_FailedPoint`` by NaN of the right shape per channel (in place), then warn once with the number of
    failed points and the traceback of the first one. Results are ordered by flat (C-order) index for every backend.

    Only flat lists can hold ``_FailedPoint``s (a failed point is in every channel). Array channels come from the
    dynamic MPI backend, whose result store has already filled them.
    """
    if not result or not isinstance(result[0], list):
        return
    failed = [j for j, value in enumerate(result[0]) if isinstance(value, _FailedPoint)]
    if not failed:
        return

    first_failure: _FailedPoint = result[0][failed[0]]
    for channel in result:
        fill = _nan_like(next(value for value in channel if not isinstance(value, _FailedPoint)))
        for j in failed:
            channel[j] = fill

    _warn_failed_points(len(failed), len(result[0]), failed[0], first_failure, param_names, param_values)


def _abort_header(rank: int, size: int, hostname: str) -> str:
    return (
        f'parameterrun: rank {rank} of {size} on {hostname} raised an exception (on_error="raise"). '
        f"Aborting the MPI job.\n"
    )


def _print_and_abort(comm: Any, message: str) -> None:  # pragma: no cover
    """Print ``message`` once to stderr, then abort the whole MPI job, so no rank waits until the walltime limit."""
    sys.stdout.flush()
    sys.stderr.write(message)
    sys.stderr.flush()
    comm.Abort(1)


def _abort_on_error(comm: Any, err: BaseException) -> None:  # pragma: no cover
    """
    Print the traceback (whose note holds the parameter values) once from the failing rank, then abort the whole MPI
    job, so the other ranks don't wait for this one until the walltime limit.
    """
    from mpi4py import MPI

    header = _abort_header(comm.Get_rank(), comm.Get_size(), MPI.Get_processor_name())
    _print_and_abort(comm, header + "".join(traceback.format_exception(err)))
