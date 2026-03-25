from parameterrun.parallel_utils import _get_iteration


def test_get_iteration_grouped():
    param_names = [["x", "y"], ["z"]]
    param_values = [[[1, 2], [10, 20]], [[100, 200, 300]], ]
    out = _get_iteration((1, 2), param_names, param_values)
    assert out == {"x": 2, "y": 20, "z": 300}
