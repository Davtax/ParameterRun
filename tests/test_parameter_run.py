import numpy as np
import pytest

import parameterrun.parallel_utils as pu


def test_parameterrun_single_parameter_returns_1d_array():
    def square(x):
        return x ** 2

    result = pu.parameterrun(square, param_names="x", param_values=[0, 1, 2, 3], n_workers=1, pbar_bool=False,
                             backend="joblib", )

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([0, 1, 4, 9]))


def test_parameterrun_single_parameter_can_return_list():
    def square(x):
        return x ** 2

    result = pu.parameterrun(square, param_names="x", param_values=[0, 1, 2, 3], n_workers=1, pbar_bool=False,
                             backend="joblib", result_as_array=False, )

    assert isinstance(result, list)
    assert result == [0, 1, 4, 9]


def test_parameterrun_grouped_parameters_builds_expected_grid():
    def f(x, y, z, offset=0):
        return x + y + z + offset

    result = pu.parameterrun(f, param_names=[["x", "y"], ["z"]],
                             param_values=[[[1, 2], [10, 20]],  # grouped pairs: (1,10), (2,20)
                                           [[100, 200, 300]],  # second group scanned independently
                                           ], n_workers=1, pbar_bool=False, backend="joblib", offset=5, )

    expected = np.array([[116, 216, 316],  # (x,y) = (1,10)
                         [127, 227, 327],  # (x,y) = (2,20)
                         ])

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, expected)


def test_parameterrun_multiple_outputs_returns_list_of_arrays():
    def f(x):
        return x + 1, x ** 2

    result = pu.parameterrun(f, param_names="x", param_values=[1, 2, 3], n_workers=1, pbar_bool=False,
                             backend="joblib", )

    assert isinstance(result, list)
    assert len(result) == 2

    np.testing.assert_array_equal(result[0], np.array([2, 3, 4]))
    np.testing.assert_array_equal(result[1], np.array([1, 4, 9]))


def test_parameterrun_multiple_outputs_can_return_list_of_lists():
    def f(x):
        return x + 1, x ** 2

    result = pu.parameterrun(f, param_names="x", param_values=[1, 2, 3], n_workers=1, pbar_bool=False,
                             backend="joblib", result_as_array=False, )

    assert isinstance(result, list)
    assert result == [[2, 3, 4], [1, 4, 9]]


def test_parameterrun_multiple_outputs_keep_grid_shape():
    def f(x, y):
        return x + y, x * y

    result = pu.parameterrun(f, param_names=["x", "y"], param_values=[[1, 2], [10, 20, 30]], n_workers=1,
                             pbar_bool=False, backend="joblib", )

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 3)
    assert result[1].shape == (2, 3)

    np.testing.assert_array_equal(result[0], np.array([[11, 21, 31], [12, 22, 32]]))
    np.testing.assert_array_equal(result[1], np.array([[10, 20, 30], [20, 40, 60]]))


def test_parameterrun_rejects_empty_parameter_groups():
    with pytest.raises(ValueError, match="at least one value"):
        pu.parameterrun(lambda x: x, param_names="x", param_values=[], n_workers=1, pbar_bool=False, backend="joblib", )


def test_parameterrun_explicit_mpi_requires_mpi4py(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "mpi4py":
            raise ImportError("mocked missing mpi4py")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match='backend="mpi"'):
        pu.parameterrun(lambda x: x, param_names="x", param_values=[1], n_workers=1, pbar_bool=False, backend="mpi", )


def test_parameterrun_invalid_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        pu.parameterrun(lambda x: x, "x", [1, 2], backend="bad", pbar_bool=False)


def test_parameterrun_rejects_unknown_swept_parameters():
    def f(x):
        return x

    with pytest.raises(ValueError, match="Unknown swept parameter"):
        pu.parameterrun(f, param_names="y", param_values=[1, 2], n_workers=1, pbar_bool=False, backend="joblib", )


def test_parameterrun_rejects_unknown_kwargs():
    def f(x):
        return x

    with pytest.raises(ValueError, match="Unknown keyword arguments"):
        pu.parameterrun(f, param_names="x", param_values=[1, 2], extra=3, n_workers=1, pbar_bool=False,
                        backend="joblib", )


def test_parameterrun_rejects_kwargs_conflicting_with_swept_parameters():
    def f(x):
        return x

    with pytest.raises(ValueError, match="conflict with swept parameters"):
        pu.parameterrun(f, param_names="x", param_values=[1, 2], x=99, n_workers=1, pbar_bool=False, backend="joblib", )
