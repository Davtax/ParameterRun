import numpy as np

from parameterrun import parameterrun


def test_joblib_2_workers() -> None:
    def f(x: float) -> float:
        return x**2

    xs = np.linspace(0, 10, 5)
    result = parameterrun(f, "x", xs, n_workers=2, pbar_bool=False, backend="joblib")

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, xs**2)
