import numpy as np

from parameterrun import parameterrun


def test_depth_1():
    def fun(x):
        return x ** 2

    xs = np.linspace(0, 10, 5)
    result = parameterrun(fun, 'x', xs, n_workers=1, pbar_bool=False)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, xs ** 2)


def test_depth_2():
    def fun(x):
        return x ** 2

    xs = np.linspace(0, 10, 5)
    result = parameterrun(fun, ['x'], [xs], n_workers=1, pbar_bool=False)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, xs ** 2)


def test_depth_3():
    def fun(x):
        return x ** 2

    xs = np.linspace(0, 10, 5)
    result = parameterrun(fun, [['x']], [[xs]], n_workers=1, pbar_bool=False)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, xs ** 2)
