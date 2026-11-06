"""Local backend: serial loop or a joblib pool."""

import contextlib
import sys
from collections.abc import Callable, Iterable
from math import prod
from typing import Any

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from . import _store
from ._checkpoint import _Checkpoint
from ._errors import (
    _all_points_failed_error,
    _evaluate_point,
    _n_outputs_of_first_success,
    _split_channels,
)
from ._inputs import _get_iteration, _params_at, _product_element_from_index
from ._progress import _pbar_options, tqdm_joblib
from ._store import Channel, _ChunkBuffer
from ._utils import _log

# Points kept in the worker-side buffer before they move into the result store (bounds the buffer's Python objects)
_FLUSH_POINTS = 4096


def _parameterrun_joblib(
    fun: Callable[..., Any],
    param_names: list[list[str]],
    param_values: list[list[list[Any]]],
    indices_iterate: Iterable[tuple[int, ...]],
    desc: str | None,
    n_workers: int | None = -1,
    pbar_bool: bool = True,
    pbar_leave: bool | None = True,
    on_error: str = "raise",
    **fun_kwargs: Any,
) -> tuple[list[list[Any]], int]:
    n_values = [len(group[0]) for group in param_values]
    n_total = prod(n_values)  # Compute the total number of iterations

    result: list[Any] = []
    pbar_options = _pbar_options(pbar_bool, sys.stderr)
    if n_workers == 1:  # If only one worker, do not use joblib
        pbar = tqdm(indices_iterate, total=n_total, desc=desc, leave=pbar_leave, **pbar_options)
        for index in pbar:
            result.append(_evaluate_point(fun, _get_iteration(index, param_names, param_values), fun_kwargs, on_error))
    else:
        with tqdm_joblib(tqdm(indices_iterate, total=n_total, desc=desc, leave=pbar_leave, **pbar_options)) as _:
            result = Parallel(n_jobs=n_workers)(
                delayed(_evaluate_point)(fun, _get_iteration(index, param_names, param_values), fun_kwargs, on_error)
                for index in indices_iterate
            )

    n_output = _n_outputs_of_first_success(result)
    if n_output == 0:
        raise _all_points_failed_error(n_total, result[0], _params_at(0, param_names, param_values))

    return _split_channels(result, n_output), n_output


def _parameterrun_joblib_checkpointed(
    fun: Callable[..., Any],
    param_names: list[list[str]],
    param_values: list[list[list[Any]]],
    desc: str | None,
    checkpoint: _Checkpoint,
    n_workers: int | None = -1,
    pbar_bool: bool = True,
    pbar_leave: bool | None = True,
    on_error: str = "raise",
    verbose: bool | None = False,
    fun_kwargs: dict[str, Any] | None = None,
) -> tuple[list[Channel], int]:
    """
    Like ``_parameterrun_joblib``, but the results go into the checkpoint's result store and are saved regularly.
    ``Parallel(return_as="generator")`` yields results in order while the pool keeps working ahead, so saving never
    makes the workers wait. Only the points the checkpoint does not hold yet are computed. If ``fun`` raises (or the
    run is interrupted), the points finished so far are saved before the exception propagates.
    """
    fun_kwargs = fun_kwargs or {}
    n_values = [len(group[0]) for group in param_values]
    axes = [list(range(n_value)) for n_value in n_values]
    n_total = prod(n_values)
    remaining = checkpoint.remaining()
    todo: Iterable[int] = range(n_total) if remaining is None else remaining.tolist()
    n_done = 0 if remaining is None else n_total - len(remaining)

    def params_of(index: int) -> dict[str, Any]:
        return _get_iteration(_product_element_from_index(axes, index), param_names, param_values)

    pbar_options = _pbar_options(pbar_bool, sys.stderr)
    pbar = tqdm(total=n_total, initial=n_done, desc=desc, leave=pbar_leave, **pbar_options)
    if n_done:
        _log(f"Resuming from checkpoint {checkpoint.directory}: {n_done} of {n_total} points done", verbose)

    if n_workers == 1:
        values: Iterable[Any] = (_evaluate_point(fun, params_of(index), fun_kwargs, on_error) for index in todo)
    else:
        values = Parallel(n_jobs=n_workers, return_as="generator")(
            delayed(_evaluate_point)(fun, params_of(index), fun_kwargs, on_error) for index in todo
        )

    buffer = _ChunkBuffer()

    def flush() -> None:
        if len(buffer):
            checkpoint.add(buffer.payload())
            buffer.clear()

    signals = checkpoint.signals  # SIGUSR1: save now; SIGTERM: save, then terminate
    try:
        with signals:
            for index, value in zip(todo, values, strict=False):
                buffer.add(index, value, params_of(index))
                pbar.update(1)
                if buffer.nbytes >= _store._MAX_MESSAGE_BYTES or len(buffer) >= _FLUSH_POINTS:
                    flush()
                if checkpoint.due(len(buffer)):
                    stop = signals.stop
                    flush()
                    if checkpoint.commit():
                        _log(f"Checkpoint saved: {checkpoint.n_saved} of {n_total} points", verbose)
                    if stop:
                        signals.terminate()
    except BaseException:  # Including the SystemExit of terminate, KeyboardInterrupt, and dead joblib workers
        with contextlib.suppress(Exception):
            flush()
        checkpoint.commit_quietly()
        raise
    finally:
        pbar.close()
        close = getattr(values, "close", None)
        if close is not None:  # Stop the pool working ahead if we leave early
            close()

    flush()
    if checkpoint.commit():
        _log(f"Checkpoint saved: {checkpoint.n_saved} of {n_total} points", verbose)
    return checkpoint.store.finalize(param_names, param_values)
