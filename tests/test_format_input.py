import pytest

from parameterrun.parallel_utils import _format_input


def test_format_input_single_parameter():
    names, values = _format_input("x", [1, 2, 3])
    assert names == [["x"]]
    assert values == [[[1, 2, 3]]]


def test_format_input_multiple_parameters():
    names, values = _format_input(["x", "y"], [[1, 2], [10, 20]])
    assert names == [["x"], ["y"]]
    assert values == [[[1, 2]], [[10, 20]]]


def test_format_input_group_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        _format_input([["x", "y"]], [[[1, 2], [10]]])


def test_format_input_rejects_empty_parameter_name():
    with pytest.raises(ValueError, match="non-empty"):
        _format_input("", [1, 2])


def test_format_input_rejects_non_list_group_values():
    with pytest.raises(ValueError, match="values entry must be a list"):
        _format_input([["x"]], [[(1, 2)]])


def test_format_input_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate parameter names"):
        _format_input(["x", "x"], [[1, 2], [3, 4]])

