"""Loud reshape failures, None outputs and joblib-inside-MPI detection (phase 1.2, 1.3, 1.7)."""

import sys
import types
import warnings
from typing import Any

import numpy as np
import pytest

import parameterrun._utils as utils
import parameterrun.parallel_utils as pu


def _run(fun: Any, values: Any = (1, 2, 3), names: Any = "x", **kwargs: Any) -> Any:
    return pu.parameterrun(fun, names, list(values), n_workers=1, pbar_bool=False, backend="joblib", **kwargs)


# ---------------------------------------------------------------- reshape


def _ragged(x: int) -> np.ndarray:
    return np.ones(x)


def test_reshape_failure_warns_with_reason_and_returns_flat_lists() -> None:
    with pytest.warns(RuntimeWarning, match=r"Could not reshape output channel 0 to the grid shape \(3,\): .*inhomog"):
        result = _run(_ragged)

    assert isinstance(result, list)
    assert len(result) == 1  # One channel
    assert [len(item) for item in result[0]] == [1, 2, 3]


def test_reshape_failure_names_the_failing_channel() -> None:
    def f(x: int) -> tuple[int, np.ndarray]:
        return x, np.ones(x)

    with pytest.warns(RuntimeWarning, match="output channel 1"):
        result = _run(f)

    assert isinstance(result, list)
    assert result[0] == [1, 2, 3]


def test_reshape_strict_raises() -> None:
    with pytest.raises(ValueError, match=r"Could not reshape output channel 0 to the grid shape \(3,\)"):
        _run(_ragged, reshape="strict")


def test_reshape_strict_returns_arrays_when_possible() -> None:
    result = _run(lambda x: x * 2, reshape="strict")
    np.testing.assert_array_equal(result, [2, 4, 6])


def test_reshape_false_returns_flat_lists_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _run(_ragged, reshape=False)
    assert isinstance(result, list)


def test_reshape_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="reshape must be"):
        _run(lambda x: x, reshape="loose")


def test_successful_run_emits_no_warning() -> None:
    def f(x: int, y: int) -> tuple[int, np.ndarray]:
        return x + y, np.full(2, x)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _run(f, values=[[1, 2], [10, 20, 30]], names=["x", "y"])

    assert result[0].shape == (2, 3)
    assert result[1].shape == (2, 3, 2)


# ---------------------------------------------------------------- None outputs


def test_none_outputs_are_reported_with_count_and_first_parameters() -> None:
    def f(x: int, y: int) -> float | None:
        return None if x == 2 else float(x + y)

    expected = r"Output channel 0: 3 of 6 results are None \(first one at \{'x': 2, 'y': 10\}"
    with pytest.warns(RuntimeWarning, match=expected):
        result = _run(f, values=[[1, 2], [10, 20, 30]], names=["x", "y"])

    # Still reshaped, but as an object array
    assert result.shape == (2, 3)
    assert result.dtype == object
    assert result[1, 0] is None


def test_none_outputs_in_second_channel() -> None:
    def f(x: int) -> tuple[int, Any]:
        return x, None if x == 3 else np.zeros(2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(f)

    messages = [str(w.message) for w in caught]
    assert any("Output channel 1: 1 of 3 results are None (first one at {'x': 3})" in m for m in messages)
    # The array/None mix cannot be reshaped either, which is reported separately
    assert any("Could not reshape output channel 1" in m for m in messages)


def test_none_outputs_with_grouped_parameters() -> None:
    def f(a: int, b: int, c: int) -> int | None:
        return None if (a, c) == (2, 300) else a + b + c

    with pytest.warns(RuntimeWarning, match=r"1 of 6 results are None \(first one at \{'a': 2, 'b': 20, 'c': 300\}"):
        _run(f, values=[[[1, 2], [10, 20]], [[100, 200, 300]]], names=[["a", "b"], ["c"]])


# ---------------------------------------------------------------- joblib inside an MPI job


@pytest.fixture
def _no_mpi_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in utils._MPI_SIZE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def _stub_joblib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid starting a real pool with n_workers=-1."""

    def fake(fun: Any, param_names: Any, param_values: Any, indices_iterate: Any, *args: Any, **kwargs: Any) -> Any:
        return [[fun(x=x) for (x,) in zip(*param_values[0], strict=True)]], 1

    monkeypatch.setattr(pu, "_parameterrun_joblib", fake)


@pytest.mark.usefixtures("_no_mpi_env", "_stub_joblib")
@pytest.mark.parametrize("var", utils._MPI_SIZE_ENV_VARS)
def test_joblib_with_all_cores_inside_mpi_job_warns(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    monkeypatch.setenv(var, "4")
    with pytest.warns(RuntimeWarning, match=r"inside an MPI job of 4 ranks.*oversubscribes"):
        result = pu.parameterrun(lambda x: x, "x", [1, 2], n_workers=-1, backend="joblib", pbar_bool=False)
    np.testing.assert_array_equal(result, [1, 2])


@pytest.mark.usefixtures("_no_mpi_env", "_stub_joblib")
@pytest.mark.parametrize(("env_size", "n_workers"), [(None, -1), ("1", -1), ("4", 2), ("4", None)])
def test_joblib_without_oversubscription_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, env_size: str | None, n_workers: int | None
) -> None:
    if env_size is not None:
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", env_size)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pu.parameterrun(lambda x: x, "x", [1, 2], n_workers=n_workers, backend="joblib", pbar_bool=False)


@pytest.mark.usefixtures("_no_mpi_env")
def test_mpi_world_size_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    assert utils._mpi_world_size_from_env() == 1

    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "not-a-number")
    monkeypatch.setenv("PMI_SIZE", "8")
    assert utils._mpi_world_size_from_env() == 8  # Invalid values are skipped


@pytest.mark.usefixtures("_no_mpi_env")
def test_mpi_world_size_from_already_imported_mpi4py(monkeypatch: pytest.MonkeyPatch) -> None:
    comm = types.SimpleNamespace(Get_size=lambda: 6)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", types.SimpleNamespace(COMM_WORLD=comm))
    assert utils._mpi_world_size_from_env() == 6
