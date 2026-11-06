"""Post-processing on rank 0: checks on the outputs and reshaping to the parameter grid."""

import warnings
from typing import Any

import numpy as np

from ._inputs import _params_at


def _warn_none_outputs(result: list[Any], param_names: list[list[str]], param_values: list[list[list[Any]]]) -> None:
    """
    Warn when some points returned None (e.g. a failed solver), since the result then stops being a numeric array.
    Results are ordered by flat (C-order) index for every backend.
    """
    for channel, values in enumerate(result):
        if isinstance(values, np.ndarray) and values.dtype != object:  # Numeric arrays cannot hold None
            continue
        none_positions = [j for j, value in enumerate(values) if value is None]
        if none_positions:
            first = _params_at(none_positions[0], param_names, param_values)
            warnings.warn(
                f"Output channel {channel}: {len(none_positions)} of {len(values)} results are None (first one at "
                f"{first}). The result will not be a numeric array. Return a NaN array of the right shape for failed "
                f"points instead of None.",
                RuntimeWarning,
                stacklevel=3,
            )


def _reshape_channel_result(channel_values: list[Any] | np.ndarray, n_values: list[int]) -> np.ndarray:
    array_result = np.asarray(channel_values)  # No copy for the arrays of the dynamic MPI backend
    if array_result.ndim == 0:
        raise ValueError("Could not reshape scalar output")

    target_shape = tuple(n_values) + tuple(array_result.shape[1:])
    reshaped: np.ndarray = array_result.reshape(target_shape)
    return reshaped


def _as_flat_lists(result: list[Any]) -> list[list[Any]]:
    """
    One flat list of per-point values per channel. Array channels (dynamic MPI backend) become Python scalars for 1-D
    arrays, and per-point arrays otherwise, matching what the other backends return.
    """
    flat: list[list[Any]] = []
    for channel in result:
        if isinstance(channel, np.ndarray):
            flat.append(channel.tolist() if channel.ndim == 1 else list(channel))
        else:
            flat.append(channel)
    return flat


def _convert_result_to_lists(result: Any) -> Any:
    if isinstance(result, np.ndarray):
        return result.tolist()
    if isinstance(result, list):
        return [_convert_result_to_lists(item) for item in result]
    return result
