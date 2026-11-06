"""Error handling in `fun`: on_error="raise" and on_error="nan" (phase 2)."""

import os
import shutil
import subprocess
import sys
import textwrap
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import parameterrun._errors as errors
import parameterrun._utils as utils
import parameterrun.parallel_utils as pu


def _run(fun: Any, values: Any = (1, 2, 3), names: Any = "x", n_workers: int = 1, **kwargs: Any) -> Any:
    return pu.parameterrun(fun, names, list(values), n_workers=n_workers, pbar_bool=False, backend="joblib", **kwargs)


def _fails_at_two(x: int) -> float:
    if x == 2:
        raise ValueError("solver diverged")
    return float(10 * x)


# ---------------------------------------------------------------- validation


def test_on_error_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="on_error must be"):
        _run(lambda x: x, on_error="skip")


def test_on_error_is_not_forwarded_to_fun() -> None:
    def f(x: int, **kwargs: Any) -> int:
        assert "on_error" not in kwargs
        return x

    np.testing.assert_array_equal(_run(f, on_error="nan"), [1, 2, 3])


# ---------------------------------------------------------------- on_error="raise"


@pytest.mark.parametrize("n_workers", [1, 2])
def test_raise_keeps_exception_type_and_adds_parameters(n_workers: int) -> None:
    with pytest.raises(ValueError, match="solver diverged") as excinfo:
        _run(_fails_at_two, n_workers=n_workers)

    assert excinfo.value.__notes__ == ["parameterrun: raised by _fails_at_two at {'x': 2}"]


def test_raise_is_the_default() -> None:
    with pytest.raises(ValueError, match="solver diverged"):
        _run(_fails_at_two)


def test_raise_reports_grouped_parameters_and_kwargs_are_not_listed() -> None:
    def f(a: int, b: int, c: int, offset: int) -> int:
        if (a, c) == (2, 300):
            raise ZeroDivisionError("bad point")
        return a + b + c + offset

    with pytest.raises(ZeroDivisionError) as excinfo:
        _run(f, values=[[[1, 2], [10, 20]], [[100, 300]]], names=[["a", "b"], ["c"]], offset=5)

    assert excinfo.value.__notes__ == ["parameterrun: raised by f at {'a': 2, 'b': 20, 'c': 300}"]


def test_base_exceptions_are_never_caught() -> None:
    class Interrupt(BaseException):
        pass

    def f(x: int) -> int:
        raise Interrupt

    with pytest.raises(Interrupt):
        errors._evaluate_point(f, {"x": 1}, {}, "nan")


# ---------------------------------------------------------------- on_error="nan"


@pytest.mark.parametrize("n_workers", [1, 2])
def test_nan_fills_only_the_failing_point_and_warns_with_count(n_workers: int) -> None:
    with pytest.warns(RuntimeWarning) as record:
        result = _run(_fails_at_two, n_workers=n_workers, on_error="nan")

    np.testing.assert_array_equal(result, [10.0, np.nan, 30.0])
    messages = [str(w.message) for w in record]
    assert len(messages) == 1
    assert messages[0].startswith(
        "1 of 3 points raised an exception and were filled with NaN (on_error=\"nan\"). First failure at {'x': 2}: "
        "ValueError: solver diverged"
    )
    assert "Traceback (most recent call last)" in messages[0]


def test_nan_when_the_first_point_fails() -> None:
    def f(x: int) -> int:
        if x == 1:
            raise RuntimeError("first")
        return x

    with pytest.warns(RuntimeWarning, match="1 of 3 points"):
        result = _run(f, on_error="nan")

    # Integer outputs become floating point so they can hold NaN
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [np.nan, 2.0, 3.0])


def test_nan_counts_several_failures_and_reports_the_first() -> None:
    def f(x: int, y: int) -> float:
        if x + y > 21:
            raise ValueError(f"too large: {x + y}")
        return float(x + y)

    with pytest.warns(RuntimeWarning, match=r"3 of 6 points .*First failure at \{'x': 1, 'y': 30\}: ValueError: too"):
        result = _run(f, values=[[1, 2], [10, 20, 30]], names=["x", "y"], on_error="nan")

    expected = np.array([[11.0, 21.0, np.nan], [12.0, np.nan, np.nan]])
    np.testing.assert_array_equal(result, expected)


def test_nan_with_multiple_array_outputs_keeps_shapes_and_dtypes() -> None:
    def f(x: int) -> tuple[np.ndarray, int, complex]:
        if x == 2:
            raise ValueError("bad")
        return np.full((2, 2), x, dtype=complex), x, 1j * x

    with pytest.warns(RuntimeWarning, match="1 of 3 points"):
        matrices, ints, scalars = _run(f, on_error="nan")

    assert matrices.shape == (3, 2, 2)
    assert matrices.dtype == np.complex128
    assert np.isnan(matrices[1]).all()
    np.testing.assert_array_equal(matrices[2], np.full((2, 2), 3, dtype=complex))

    assert ints.dtype == np.float64
    np.testing.assert_array_equal(ints, [1.0, np.nan, 3.0])

    assert scalars.dtype == np.complex128
    assert np.isnan(scalars[1])


def test_nan_without_failures_emits_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _run(lambda x: x, on_error="nan")
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_nan_with_flat_lists() -> None:
    with pytest.warns(RuntimeWarning, match="1 of 3 points"):
        result = _run(_fails_at_two, on_error="nan", reshape=False)

    assert isinstance(result, list)
    assert result[0][0] == 10.0
    assert np.isnan(result[0][1])


def test_nan_with_non_numeric_output_falls_back_to_none() -> None:
    def f(x: int) -> str:
        if x == 2:
            raise ValueError("bad")
        return f"point {x}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(f, on_error="nan")

    assert list(result) == ["point 1", None, "point 3"]
    messages = [str(w.message) for w in caught]
    assert any("1 of 3 points raised an exception" in m for m in messages)
    assert any("1 of 3 results are None" in m for m in messages)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_nan_raises_when_every_point_fails(n_workers: int) -> None:
    def f(x: int) -> int:
        raise KeyError(f"missing {x}")

    with pytest.raises(
        RuntimeError, match=r"(?s)All 3 points raised an exception .*Failure at {'x': 1}.*KeyError: 'missing 1'"
    ):
        _run(f, on_error="nan", n_workers=n_workers)


@pytest.mark.parametrize(
    ("template", "expected_shape", "expected_dtype"),
    [
        (1, (), np.float64),
        (True, (), np.float64),
        (np.float32(1.0), (), np.float32),
        (1 + 1j, (), np.complex128),
        (np.zeros((2, 3), dtype=np.int32), (2, 3), np.float64),
        (np.zeros(4, dtype=np.complex64), (4,), np.complex64),
    ],
)
def test_nan_like(template: Any, expected_shape: tuple[int, ...], expected_dtype: Any) -> None:
    fill = errors._nan_like(template)
    assert np.shape(fill) == expected_shape
    assert np.asarray(fill).dtype == expected_dtype
    assert np.isnan(fill).all()


def test_nan_like_non_numeric_is_none() -> None:
    assert errors._nan_like("text") is None
    assert errors._nan_like({"a": 1}) is None


def test_failed_point_survives_pickling() -> None:
    import pickle

    failure = errors._evaluate_point(_fails_at_two, {"x": 2}, {}, "nan")
    assert isinstance(failure, errors._FailedPoint)
    assert pickle.loads(pickle.dumps(failure)) == failure
    assert failure.summary == "ValueError: solver diverged"


# ---------------------------------------------------------------- on_error="raise" under mpiexec (subprocess)

_FAILING_MPI_SCRIPT = textwrap.dedent(
    """
    import time
    import sys
    from parameterrun import parameterrun

    def f(x):
        if x == 5:
            raise ValueError("boom at five")
        time.sleep(0.01)
        return x

    parameterrun(f, "x", list(range(40)), backend="mpi", pbar_bool=False, mpi_schedule=sys.argv[1])
    print("UNREACHABLE", flush=True)
    """
)


@pytest.mark.mpi_subprocess
@pytest.mark.parametrize("schedule", ["static", "dynamic"])
def test_mpi_raise_aborts_the_job_with_one_traceback(tmp_path: Path, schedule: str) -> None:
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None or find_spec("mpi4py") is None:
        pytest.skip("Needs mpiexec on PATH and mpi4py")
    if utils._mpi_world_size_from_env() > 1:
        pytest.skip("Already inside an MPI job: don't nest mpiexec")

    script = tmp_path / "failing_sweep.py"
    script.write_text(_FAILING_MPI_SCRIPT, encoding="utf-8")

    # Make the working tree importable even if parameterrun is not installed in the interpreter
    src_dir = str(Path(pu.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src_dir, os.environ.get("PYTHONPATH")]))}

    completed = subprocess.run(
        [mpiexec, "-n", "4", sys.executable, str(script), schedule],
        capture_output=True,
        text=True,
        timeout=120,  # The point of the test: the other ranks must not hang
        env=env,
        check=False,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode != 0, output
    assert "UNREACHABLE" not in output
    assert output.count('raised an exception (on_error="raise"). Aborting the MPI job.') == 1, output
    assert output.count("Traceback (most recent call last)") == 1, output
    assert "ValueError: boom at five" in output
    assert "parameterrun: raised by f at {'x': 5}" in output
