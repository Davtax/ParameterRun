"""Local backend: serial loop or a joblib pool."""

import sys
from collections.abc import Callable, Iterable
from math import prod
from typing import Any

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ._errors import _all_points_failed_error, _evaluate_point, _n_outputs_of_first_success, _split_channels
from ._inputs import _get_iteration, _params_at
from ._progress import _pbar_options, tqdm_joblib


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
