"""Normalisation and validation of the parameter grid, and mapping from grid indices to parameter values."""

import inspect
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

from ._utils import _callable_name


def _format_input(
    param_names: str | list[str] | list[list[str]], param_values: Any
) -> tuple[list[list[str]], list[list[list[Any]]]]:
    # Check the depth of the parameters

    if isinstance(param_names, str):
        if not param_names:
            raise ValueError("Parameter names must be non-empty strings")
        depth = 1
    elif isinstance(param_names, list) and len(param_names) > 0 and all(isinstance(name, str) for name in param_names):
        if any(not name for name in param_names):
            raise ValueError("Parameter names must be non-empty strings")
        depth = 2
    elif (
        isinstance(param_names, list) and len(param_names) > 0 and all(isinstance(group, list) for group in param_names)
    ):
        if any(len(group) == 0 for group in param_names):
            raise ValueError("Parameter name groups must contain at least one parameter")
        if any(any(not isinstance(name, str) or not name for name in group) for group in param_names):
            raise ValueError("Parameter names must be non-empty strings")
        depth = 3
    else:
        raise ValueError("Unknown input format for the parameters names")

    normalized_param_names: list[list[str]]
    normalized_param_values: list[list[list[Any]]]

    # Correct the input if the depth is not equal to 3
    if depth == 1:
        # Only one parameter is provided
        values = _to_list_iterable(param_values, "The parameters values must be an iterable of values")
        normalized_param_names = [[cast(str, param_names)]]
        normalized_param_values = [[values]]
    elif depth == 2:
        # Groups with a single parameters are provided
        values = _to_list_iterable(param_values, "The parameters values must be an iterable of values")
        normalized_param_names = [[param_name_i] for param_name_i in cast(list[str], param_names)]
        normalized_param_values = [
            [_to_list_iterable(param_values_i, "Each parameter values entry must be an iterable")]
            for param_values_i in values
        ]
    else:
        groups_values = _to_list_iterable(param_values, "The parameters values must be an iterable of values")
        groups_values = [
            _to_list_iterable(values_group, "Each parameter group values must be provided as an iterable")
            for values_group in groups_values
        ]
        normalized_param_names = cast(list[list[str]], param_names)
        normalized_param_values = [
            [
                _to_list_iterable(parameter_values, "Each parameter values entry must be an iterable")
                for parameter_values in values_group
            ]
            for values_group in groups_values
        ]

    # Check that the number of groups of parameters is the same as the number of groups of values
    if len(normalized_param_names) != len(normalized_param_values):
        raise ValueError("The number of groups of parameters must be the same as the number of groups of values")

    for names_group, values_group in zip(normalized_param_names, normalized_param_values, strict=True):
        if len(names_group) != len(values_group):
            raise ValueError("Each parameter group must include one values list per parameter")

    # Check that all parameters in the same group have the same length
    for group in normalized_param_values:
        if len(set([len(param) for param in group])) != 1:
            raise ValueError("All parameters in the same group must have the same length")

    all_names = _flatten_param_names(normalized_param_names)
    duplicated = sorted({name for name in all_names if all_names.count(name) > 1})
    if duplicated:
        raise ValueError(f"Duplicate parameter names are not allowed: {duplicated}")

    return normalized_param_names, normalized_param_values


def _flatten_param_names(param_names: list[list[str]]) -> list[str]:
    return [name for group in param_names for name in group]


def _to_list_iterable(values: Any, error_message: str) -> list[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(error_message)
    return list(values)


def _validate_function_arguments(
    fun: Callable[..., Any], param_names: list[list[str]], fun_kwargs: dict[str, Any]
) -> None:
    signature = inspect.signature(fun)
    parameters = signature.parameters
    accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())

    sweep_names = _flatten_param_names(param_names)

    duplicated = sorted({name for name in sweep_names if sweep_names.count(name) > 1})
    if duplicated:
        raise ValueError(f"Duplicate parameter names are not allowed: {duplicated}")

    overlapping_kwargs = sorted(set(sweep_names).intersection(fun_kwargs))
    if overlapping_kwargs:
        raise ValueError(
            f"Keyword arguments conflict with swept parameters: {overlapping_kwargs}. "
            f"Remove them from kwargs or param_names."
        )

    if accepts_var_kwargs:
        return

    known_names = set(parameters)
    unknown_sweep = sorted(name for name in sweep_names if name not in known_names)
    if unknown_sweep:
        raise ValueError(f"Unknown swept parameter names for function {_callable_name(fun)}: {unknown_sweep}")

    unknown_kwargs = sorted(name for name in fun_kwargs if name not in known_names)
    if unknown_kwargs:
        raise ValueError(f"Unknown keyword arguments for function {_callable_name(fun)}: {unknown_kwargs}")


def _get_iteration(
    index: Sequence[int], param_names: list[list[str]], param_values: list[list[list[Any]]]
) -> dict[str, Any]:
    n_groups = len(param_values)
    n_parameters = [len(group) for group in param_values]

    dic_temp: dict[str, Any] = {}
    for group_index in range(n_groups):
        for parameter_index in range(n_parameters[group_index]):
            dic_temp[param_names[group_index][parameter_index]] = param_values[group_index][parameter_index][
                index[group_index]
            ]

    return dic_temp


def _product_element_from_index(lists: list[list[int]], k: int) -> list[int]:
    lengths = [len(L) for L in lists]

    indices = [0] * len(lists)
    for i in range(len(lists) - 1, -1, -1):
        k, indices[i] = divmod(k, lengths[i])

    return list(L[j] for L, j in zip(lists, indices, strict=True))


def _params_at(flat_index: int, param_names: list[list[str]], param_values: list[list[list[Any]]]) -> dict[str, Any]:
    """Parameter values of the point at a flat (C-order) index."""
    indices = [list(range(len(group[0]))) for group in param_values]
    return _get_iteration(_product_element_from_index(indices, flat_index), param_names, param_values)
