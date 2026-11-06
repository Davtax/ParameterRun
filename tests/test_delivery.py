"""Result delivery without MPI (phase 4): result files, the file-backed store, return_on, and the static gather plan."""

import gc
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import parameterrun.parallel_utils as pu
from parameterrun._delivery import _check_result_file, _item_layout, _mapped_file, _write_result_files
from parameterrun._errors import _FailedPoint
from parameterrun._mpi_static import _plan_gather
from parameterrun._store import _channel_paths, _ResultStore


def _run(fun: Any, values: Any = (1, 2, 3), names: Any = "x", **kwargs: Any) -> Any:
    return pu.parameterrun(fun, names, list(values), n_workers=1, pbar_bool=False, backend="joblib", **kwargs)


def _failure(message: str = "boom") -> _FailedPoint:
    return _FailedPoint(summary=f"ValueError: {message}", traceback=f"Traceback ...\nValueError: {message}\n")


# ---------------------------------------------------------------- validation


def test_check_result_file(tmp_path: Path) -> None:
    assert _check_result_file(tmp_path / "result.npy") is None
    assert "must end in .npy" in str(_check_result_file(tmp_path / "result.txt"))
    assert "does not exist" in str(_check_result_file(tmp_path / "missing" / "result.npy"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"return_on": "everyone"}, "return_on must be"),
        ({"return_on": "none"}, "needs a result_file"),
        ({"result_file": "result.txt"}, "must end in .npy"),
    ],
)
def test_invalid_delivery_options_are_rejected(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _run(lambda x: x, **kwargs)


def test_missing_directory_is_reported_before_computing(tmp_path: Path) -> None:
    calls: list[int] = []

    def f(x: int) -> int:
        calls.append(x)
        return x

    with pytest.raises(ValueError, match="does not exist"):
        _run(f, result_file=tmp_path / "missing" / "result.npy")
    assert calls == []


def test_delivery_options_are_not_forwarded_to_fun(tmp_path: Path) -> None:
    def f(x: int, **kwargs: Any) -> int:
        assert not kwargs
        return x

    _run(f, return_on="all", result_file=tmp_path / "r.npy")


# ---------------------------------------------------------------- result files with joblib


def test_result_file_holds_the_reshaped_result(tmp_path: Path) -> None:
    path = tmp_path / "result.npy"
    result = _run(lambda x, y: x * y + 0.5, values=[[1, 2], [3, 4, 5]], names=["x", "y"], result_file=path)

    np.testing.assert_array_equal(np.load(path), result)
    assert np.load(path).shape == (2, 3)


def test_result_file_with_several_outputs_writes_one_file_each(tmp_path: Path) -> None:
    result = _run(lambda x: (x, np.full(2, x * 1j)), result_file=tmp_path / "result.npy")

    assert not (tmp_path / "result.npy").exists()
    np.testing.assert_array_equal(np.load(tmp_path / "result_0.npy"), result[0])
    np.testing.assert_array_equal(np.load(tmp_path / "result_1.npy"), result[1])
    assert np.load(tmp_path / "result_1.npy").dtype == np.complex128


def test_result_file_is_written_even_without_reshape(tmp_path: Path) -> None:
    path = tmp_path / "result.npy"
    result = _run(lambda x, y: x + y, values=[[1, 2], [3, 4, 5]], names=["x", "y"], result_file=path, reshape=False)

    assert result == [[4, 5, 6, 5, 6, 7]]
    np.testing.assert_array_equal(np.load(path), [[4, 5, 6], [5, 6, 7]])


@pytest.mark.parametrize(
    ("fun", "reason"),
    [
        (lambda x: np.ones(x), "inhomogeneous"),  # Ragged
        (lambda x: None if x == 2 else x, "Python objects"),
    ],
)
def test_outputs_that_do_not_form_one_plain_array_are_not_written(tmp_path: Path, fun: Any, reason: str) -> None:
    path = tmp_path / "result.npy"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(fun, result_file=path)

    assert not path.exists()
    messages = [str(w.message) for w in caught]
    assert any(f"Output channel 0 was not written to {path}" in m and reason in m for m in messages), messages


def test_string_outputs_are_written(tmp_path: Path) -> None:
    path = tmp_path / "result.npy"
    _run(lambda x: f"p{x}", result_file=path)
    np.testing.assert_array_equal(np.load(path), ["p1", "p2", "p3"])  # A plain <U array: no pickling needed


def test_return_on_none_returns_nothing_but_writes_the_file(tmp_path: Path) -> None:
    path = tmp_path / "result.npy"
    assert _run(lambda x: 2 * x, return_on="none", result_file=path) is None
    np.testing.assert_array_equal(np.load(path), [2, 4, 6])


def test_return_on_all_under_joblib_is_root() -> None:
    np.testing.assert_array_equal(_run(lambda x: 2 * x, return_on="all"), [2, 4, 6])


def test_result_file_with_on_error_nan(tmp_path: Path) -> None:
    def f(x: int) -> int:
        if x == 2:
            raise ValueError("bad")
        return x

    path = tmp_path / "result.npy"
    with pytest.warns(RuntimeWarning, match="1 of 3 points"):
        _run(f, result_file=path, on_error="nan")
    np.testing.assert_array_equal(np.load(path), [1.0, np.nan, 3.0])


# ---------------------------------------------------------------- file-backed result store


def _store(tmp_path: Path, n_total: int = 4, grid: tuple[int, ...] | None = None) -> _ResultStore:
    return _ResultStore(n_total, grid or (n_total,), tmp_path / "out.npy")


def test_channel_paths(tmp_path: Path) -> None:
    assert _channel_paths(tmp_path / "r.npy", 1) == [tmp_path / "r.npy"]
    assert _channel_paths(tmp_path / "r.npy", 2) == [tmp_path / "r_0.npy", tmp_path / "r_1.npy"]


def test_store_writes_into_a_grid_shaped_file(tmp_path: Path) -> None:
    store = _store(tmp_path, n_total=6, grid=(2, 3))
    store.write([5, 0, 3], [np.array([[5.0, 50.0], [0.0, 0.0], [3.0, 30.0]])])
    store.write([1, 2, 4], [np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])])

    (channel,), _ = store.finalize([["x"]], [[list(range(6))]])
    assert isinstance(channel, np.memmap)
    assert _mapped_file(channel) == (tmp_path / "out.npy").resolve()

    on_disk = np.load(tmp_path / "out.npy")
    assert on_disk.shape == (2, 3, 2)
    np.testing.assert_array_equal(on_disk.reshape(6, 2), channel)

    # Copy-on-write: changing the returned array never changes the file
    channel[0] = -1
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy")[0, 0], [0.0, 0.0])
    del channel, store
    gc.collect()


def test_store_promotes_a_file_backed_channel(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write([0, 1], [np.array([1, 2])])
    store.write([2, 3], [np.array([2.5, 3.5])])  # int64 -> float64: the file is rewritten

    (channel,), _ = store.finalize([["x"]], [[[0, 1, 2, 3]]])
    on_disk = np.load(tmp_path / "out.npy")
    assert on_disk.dtype == np.float64
    np.testing.assert_array_equal(on_disk, [1.0, 2.0, 2.5, 3.5])
    assert not (tmp_path / "out.npy.tmp").exists()
    del channel, store
    gc.collect()


def test_store_fills_nan_in_a_file_backed_int_channel(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_failures([(1, _failure())])
    store.write([0, 2, 3], [np.array([10, 12, 13])])

    with pytest.warns(RuntimeWarning, match="1 of 4 points"):
        (channel,), _ = store.finalize([["x"]], [[[0, 1, 2, 3]]])
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), [10.0, np.nan, 12.0, 13.0])
    del channel, store
    gc.collect()


def test_store_removes_the_file_when_falling_back_to_a_list(tmp_path: Path) -> None:
    store = _store(tmp_path, n_total=3)
    store.write([0], [np.array([[1.0, 2.0]])])
    assert (tmp_path / "out.npy").exists()
    store.write([1, 2], [[np.ones(3), np.ones(4)]])  # Ragged: no single array, so no file

    (channel,), _ = store.finalize([["x"]], [[[0, 1, 2]]])
    assert isinstance(channel, list)
    np.testing.assert_array_equal(channel[0], [1.0, 2.0])
    assert not (tmp_path / "out.npy").exists()


def test_store_allocate_with_list_and_array_channels(tmp_path: Path) -> None:
    store = _store(tmp_path, n_total=2)
    store.allocate([("array", (2,), "<f8"), ("list",)])
    store.write_channel(0, np.array([0, 1]), np.array([[1.0, 1.0], [2.0, 2.0]]))
    store.write_channel(1, np.array([0, 1]), ["a", "b"])
    store.mark_done([0, 1])

    (array, items), n_outputs = store.finalize([["x"]], [[[0, 1]]])
    assert n_outputs == 2
    assert items == ["a", "b"]
    np.testing.assert_array_equal(np.load(tmp_path / "out_0.npy"), array)
    assert not (tmp_path / "out_1.npy").exists()
    del array, store
    gc.collect()


def test_store_keeps_empty_outputs_in_memory(tmp_path: Path) -> None:
    store = _store(tmp_path, n_total=2)
    store.write([0, 1], [np.zeros((2, 0))])  # Zero bytes: a file cannot be mapped
    (channel,), _ = store.finalize([["x"]], [[[0, 1]]])
    assert not isinstance(channel, np.memmap)

    # parameterrun then saves it like any in-memory result
    _write_result_files([channel], [2], tmp_path / "out.npy")
    assert np.load(tmp_path / "out.npy").shape == (2, 0)


def test_store_add_failed_indices_share_one_failure() -> None:
    store = _ResultStore(4)
    store.add_failed_indices(np.array([3, 1]), _failure("first on this rank"))
    store.write([0, 2], [np.array([0.0, 2.0])])
    with pytest.warns(RuntimeWarning, match=r"2 of 4 points.*\{'x': 1\}: ValueError: first on this rank"):
        (channel,), _ = store.finalize([["x"]], [[[0, 1, 2, 3]]])
    np.testing.assert_array_equal(channel, [0.0, np.nan, 2.0, np.nan])


# ---------------------------------------------------------------- static gather plan


def _summary(
    n_outputs: int | None,
    specs: list[tuple[Any, ...]],
    n_points: int = 2,
    error: str | None = None,
    first_failure: tuple[int, _FailedPoint] | None = None,
) -> dict[str, Any]:
    return {
        "n_outputs": n_outputs,
        "n_points": n_points,
        "specs": specs,
        "error": error,
        "failed_indices": np.array([first_failure[0]] if first_failure else [], dtype=np.int64),
        "first_failure": first_failure,
    }


def _plan(summaries: list[dict[str, Any]]) -> tuple[Any, ...]:
    return _plan_gather(summaries, 10, [["x"]], [[list(range(10))]])


def test_plan_gather_arrays_with_promoted_dtype() -> None:
    plan = _plan(
        [
            _summary(1, [("array", (2,), "<i8")]),
            _summary(1, [("array", (2,), "<f8")]),
            _summary(None, [], n_points=0),  # A rank without successful points does not vote
        ]
    )
    assert plan == ("ok", [("array", (2,), "<f8")])


@pytest.mark.parametrize(
    "other",
    [("array", (3,), "<f8"), ("list",)],  # Different shape on another rank, or not stackable there
)
def test_plan_gather_falls_back_to_lists(other: tuple[Any, ...]) -> None:
    plan = _plan(
        [_summary(2, [("array", (2,), "<f8"), ("array", (), "<i8")]), _summary(2, [other, ("array", (), "<i8")])]
    )
    assert plan == ("ok", [("list",), ("array", (), "<i8")])


def test_plan_gather_reports_inconsistent_numbers_of_outputs() -> None:
    plan = _plan([_summary(1, [("list",)]), _summary(2, [("list",), ("list",)])])
    assert plan[:2] == ("error", "ValueError")
    assert "1 outputs for some points but 2 for others" in plan[2]


def test_plan_gather_reports_a_rank_local_error() -> None:
    plan = _plan([_summary(1, [("list",)]), _summary(1, [("list",)], error="fun returned 2 outputs at {'x': 3}")])
    assert plan == ("error", "ValueError", "fun returned 2 outputs at {'x': 3}")


def test_plan_gather_reports_the_first_failure_when_every_point_failed() -> None:
    plan = _plan(
        [
            _summary(None, [], n_points=0, first_failure=(7, _failure("late"))),
            _summary(None, [], n_points=0, first_failure=(2, _failure("early"))),
        ]
    )
    assert plan[:2] == ("error", "RuntimeError")
    assert "All 10 points raised an exception" in plan[2]
    assert "Failure at {'x': 2}" in plan[2] and "early" in plan[2]


def test_item_layout() -> None:
    assert _item_layout(np.zeros((2, 3), dtype=np.complex64)) == ("array", (2, 3), "<c8")
    assert _item_layout(np.array(["a", None], dtype=object)) == ("pickle",)
    assert _item_layout([1, 2]) == ("pickle",)
