"""Dynamic MPI schedule without MPI: result store, worker chunk buffer, chunk sizes and schedule selection (phase 3)."""

import gc
import pickle
import warnings
from collections.abc import Callable, Iterator
from itertools import product
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import parameterrun.parallel_utils as pu
from parameterrun._errors import _evaluate_point, _FailedPoint
from parameterrun._inputs import _get_iteration, _product_element_from_index
from parameterrun._mpi_dynamic import _guided_chunk_size
from parameterrun._pseudo_random import _permute_in_range
from parameterrun._results import _as_flat_lists
from parameterrun._store import _MAX_MESSAGE_BYTES, _ChunkBuffer, _ResultStore, _stack_channel


def _failure(message: str = "boom") -> _FailedPoint:
    return _FailedPoint(summary=f"ValueError: {message}", traceback=f"Traceback ...\nValueError: {message}\n")


# ---------------------------------------------------------------- _stack_channel


def test_stack_channel_numeric_values_become_one_array() -> None:
    stacked = _stack_channel([np.ones(2), np.zeros(2)])
    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == (2, 2)

    scalars = _stack_channel([1, 2.5])
    assert isinstance(scalars, np.ndarray)
    assert scalars.dtype == np.float64


@pytest.mark.parametrize(
    "values",
    [
        [np.ones(2), np.ones(3)],  # Ragged
        ["a", "b"],  # Not numeric
        [1, None],  # Object
        [{"a": 1}],
        [[1, 2], [3, 4]],  # Python lists stay lists
        [2**70],  # Out of the int64 range
    ],
)
def test_stack_channel_keeps_other_values_as_a_list(values: list[Any]) -> None:
    assert _stack_channel(values) is values


# ---------------------------------------------------------------- _ResultStore


def test_store_writes_out_of_order_chunks_into_one_array() -> None:
    store = _ResultStore(5)
    store.write([3, 0], [np.array([[3, 30], [0, 0]])])
    store.write([4, 1, 2], [np.array([[4, 40], [1, 10], [2, 20]])])

    channels, n_outputs = store.finalize([["x"]], [[list(range(5))]])
    assert n_outputs == 1
    np.testing.assert_array_equal(channels[0], [[0, 0], [1, 10], [2, 20], [3, 30], [4, 40]])
    assert store.done.all()


def test_store_promotes_dtypes_like_np_array() -> None:
    store = _ResultStore(3)
    store.write([0], [np.array([1])])  # int64
    store.write([1], [np.array([2.5])])  # float64: must not be truncated to 2
    store.write([2], [np.array([1j])])

    (channel,), _ = store.finalize([["x"]], [[[0, 1, 2]]])
    assert isinstance(channel, np.ndarray)
    assert channel.dtype == np.array([1, 2.5, 1j]).dtype
    np.testing.assert_array_equal(channel, [1, 2.5, 1j])


def test_store_keeps_float32() -> None:
    store = _ResultStore(2)
    store.write([0, 1], [np.array([1.0, 2.0], dtype=np.float32)])
    (channel,), _ = store.finalize([["x"]], [[[0, 1]]])
    assert isinstance(channel, np.ndarray)
    assert channel.dtype == np.float32


def test_store_falls_back_to_a_list_when_the_shape_changes() -> None:
    store = _ResultStore(3)
    store.write([2, 0], [np.array([[2.0, 2.0], [0.0, 0.0]])])
    store.write([1], [[np.ones(3)]])  # Ragged with the previous ones: arrives as a list

    (channel,), _ = store.finalize([["x"]], [[[0, 1, 2]]])
    assert isinstance(channel, list)
    assert [np.shape(value) for value in channel] == [(2,), (3,), (2,)]


def test_store_falls_back_to_a_list_for_non_numeric_outputs() -> None:
    store = _ResultStore(2)
    store.write([1], [["b"]])
    store.write([0], [np.array([7])])

    (channel,), _ = store.finalize([["x"]], [[[0, 1]]])
    assert channel == [7, "b"]


def test_store_rejects_a_different_number_of_outputs() -> None:
    store = _ResultStore(2)
    store.write([0], [np.array([1]), np.array([2])])
    with pytest.raises(ValueError, match="1 outputs for some points but 2 for others"):
        store.write([1], [np.array([1])])


def test_store_ignores_empty_writes() -> None:
    store = _ResultStore(1)
    store.write(np.empty(0, dtype=np.int64), [])
    assert store.n_outputs == 0
    assert not store.done.any()


def test_store_fills_failures_with_nan_and_warns_once() -> None:
    store = _ResultStore(4)
    store.add_failures([(2, _failure("second")), (1, _failure("first"))])  # Before any success: shape unknown yet
    store.write([0, 3], [np.array([10, 13]), np.array([[1j, 1j], [2j, 2j]], dtype=np.complex64)])

    with pytest.warns(RuntimeWarning) as record:
        (ints, matrices), n_outputs = store.finalize([["x"]], [[["a", "b", "c", "d"]]])

    assert n_outputs == 2
    assert isinstance(ints, np.ndarray) and isinstance(matrices, np.ndarray)
    assert ints.dtype == np.float64  # Integers become float64 to hold NaN
    np.testing.assert_array_equal(ints, [10.0, np.nan, np.nan, 13.0])
    assert matrices.dtype == np.complex64  # Complex keeps its dtype
    assert np.isnan(matrices[1:3]).all()

    assert len(record) == 1
    message = str(record[0].message)
    assert message.startswith('2 of 4 points raised an exception and were filled with NaN (on_error="nan").')
    assert "First failure at {'x': 'b'}: ValueError: first" in message


def test_store_fills_failures_in_list_channels() -> None:
    store = _ResultStore(3)
    store.add_failures([(0, _failure())])
    store.write([1, 2], [[np.ones(2), np.ones(3)]])  # Ragged: list channel

    with pytest.warns(RuntimeWarning, match="1 of 3 points"):
        (channel,), _ = store.finalize([["x"]], [[[0, 1, 2]]])

    assert isinstance(channel, list)
    assert np.isnan(channel[0]).all() and np.shape(channel[0]) == (2,)  # Shape of the first successful value


def test_store_raises_when_every_point_failed() -> None:
    store = _ResultStore(2)
    store.add_failures([(1, _failure("late")), (0, _failure("early"))])
    with pytest.raises(RuntimeError, match=r"(?s)All 2 points raised an exception.*Failure at \{'x': 5\}.*early"):
        store.finalize([["x"]], [[[5, 6]]])


def test_store_detects_missing_points() -> None:
    store = _ResultStore(3)
    store.write([0, 2], [np.array([0, 2])])
    with pytest.raises(RuntimeError, match=r"no result for 1 points \(first: 1\)"):
        store.finalize([["x"]], [[[0, 1, 2]]])


# ---------------------------------------------------------------- _ChunkBuffer


def test_chunk_buffer_splits_outputs_into_stacked_channels() -> None:
    buffer = _ChunkBuffer()
    buffer.add(4, (1.0, np.zeros(3)), {"x": 4})
    buffer.add(7, _failure(), {"x": 7})
    buffer.add(2, (2.0, np.ones(3)), {"x": 2})

    assert len(buffer) == 3
    indices, channels, failures = buffer.payload()
    np.testing.assert_array_equal(indices, [4, 2])
    np.testing.assert_array_equal(channels[0], [1.0, 2.0])
    np.testing.assert_array_equal(channels[1], [np.zeros(3), np.ones(3)])
    assert [index for index, _ in failures] == [7]

    # The payload is what travels over MPI
    restored = pickle.loads(pickle.dumps(buffer.payload()))
    np.testing.assert_array_equal(restored[1][1], channels[1])

    buffer.clear()
    assert len(buffer) == 0
    assert buffer.nbytes == 0
    assert buffer.n_outputs == 2  # Kept across chunks for the consistency check


def test_chunk_buffer_rejects_a_different_number_of_outputs() -> None:
    buffer = _ChunkBuffer()
    buffer.add(0, (1, 2), {"x": 0})
    with pytest.raises(ValueError, match=r"fun returned 1 outputs at \{'x': 1\}, but 2 at an earlier point"):
        buffer.add(1, 3, {"x": 1})


def test_chunk_buffer_counts_array_bytes() -> None:
    buffer = _ChunkBuffer()
    buffer.add(0, np.zeros(_MAX_MESSAGE_BYTES // 8), {"x": 0})
    assert buffer.nbytes >= _MAX_MESSAGE_BYTES  # The worker sends this chunk early


def test_chunk_buffer_with_only_failures_has_no_channels() -> None:
    buffer = _ChunkBuffer()
    buffer.add(0, _failure(), {"x": 0})
    indices, channels, failures = buffer.payload()
    assert len(indices) == 0
    assert channels == []
    assert len(failures) == 1


# ---------------------------------------------------------------- chunk sizes and schedule selection


def test_guided_chunk_size_shrinks_to_one() -> None:
    sizes = []
    remaining = 1000
    while remaining:
        k = _guided_chunk_size(remaining, n_workers=3)
        sizes.append(k)
        remaining -= k

    assert sizes[0] == 1000 // 12
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] == 1
    assert sum(sizes) == 1000


@pytest.mark.parametrize(("remaining", "chunk_size", "expected"), [(10, 3, 3), (2, 5, 2), (1, None, 1), (5, None, 1)])
def test_chunk_size_limits(remaining: int, chunk_size: int | None, expected: int) -> None:
    assert _guided_chunk_size(remaining, n_workers=3, chunk_size=chunk_size) == expected


@pytest.mark.parametrize(
    ("schedule", "size", "expected"),
    [
        ("auto", 1, "static"),
        ("auto", 4, "static"),
        ("auto", 7, "static"),
        ("auto", 8, "dynamic"),
        ("auto", 256, "dynamic"),
        ("static", 64, "static"),
        ("dynamic", 2, "dynamic"),
        ("dynamic", 1, "static"),  # Needs a manager and a worker
    ],
)
def test_resolve_mpi_schedule(schedule: str, size: int, expected: str) -> None:
    assert pu._resolve_mpi_schedule(schedule, size) == expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mpi_schedule": "guided"}, "mpi_schedule must be"),
        ({"mpi_chunk_size": 0}, "mpi_chunk_size must be"),
        ({"mpi_chunk_size": 2.5}, "mpi_chunk_size must be"),
        ({"mpi_chunk_size": True}, "mpi_chunk_size must be"),
    ],
)
def test_invalid_mpi_options_are_rejected(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pu.parameterrun(lambda x: x, "x", [1, 2], backend="joblib", n_workers=1, pbar_bool=False, **kwargs)


def test_mpi_options_are_not_forwarded_to_fun() -> None:
    def f(x: int, **kwargs: Any) -> int:
        assert not kwargs
        return x

    pu.parameterrun(
        f, "x", [1, 2], backend="joblib", n_workers=1, pbar_bool=False, mpi_schedule="dynamic", mpi_chunk_size=4
    )


def test_as_flat_lists() -> None:
    flat = _as_flat_lists([np.array([1.5, 2.5]), np.ones((2, 3)), ["a", "b"]])
    assert flat[0] == [1.5, 2.5] and type(flat[0][0]) is float
    assert len(flat[1]) == 2 and flat[1][0].shape == (3,)
    assert flat[2] == ["a", "b"]


# ---------------------------------------------------------------- the dynamic data path against joblib


def _simulated_dynamic_backend(
    chunk_size: int | None, n_workers: int = 3, seed: int = 99, result_file: Path | None = None
) -> Callable[..., Any]:
    """
    Stand-in for `_parameterrun_joblib` that runs the data path of the dynamic MPI backend in one process: permuted
    positions, guided chunks, worker buffers (with pickling, as over MPI) and the manager's result store.
    """

    def backend(
        fun: Callable[..., Any],
        param_names: list[list[str]],
        param_values: list[list[list[Any]]],
        indices_iterate: Iterator[Any],
        desc: str | None,
        on_error: str = "raise",
        **kwargs: Any,
    ) -> tuple[list[Any], int]:
        for option in ("n_workers", "pbar_bool", "pbar_leave"):
            kwargs.pop(option, None)
        n_values = [len(group[0]) for group in param_values]
        indices = [list(range(n)) for n in n_values]
        n_total = prod(n_values)

        store = _ResultStore(n_total, n_values, result_file)
        buffer = _ChunkBuffer()
        position = 0
        while position < n_total:
            stop = position + _guided_chunk_size(n_total - position, n_workers, chunk_size)
            for p in range(position, stop):
                index = _permute_in_range(p, n_total, seed)
                params = _get_iteration(_product_element_from_index(indices, index), param_names, param_values)
                buffer.add(index, _evaluate_point(fun, params, kwargs, on_error), params)
            idx, channels, failures = pickle.loads(pickle.dumps(buffer.payload()))
            store.write(idx, channels)
            store.add_failures(failures)
            buffer.clear()
            position = stop
        return store.finalize(param_names, param_values)

    return backend


def _sweeps() -> Iterator[tuple[str, Callable[..., Any], Any, Any]]:
    yield "scalar", lambda x, y: x * y, ["x", "y"], [range(7), range(5)]
    yield "int-then-float", lambda x: x if x < 5 else x + 0.5, "x", range(12)
    yield "multiple", lambda x: (np.full((2, 2), x, dtype=complex), x, np.float32(x)), "x", range(9)
    yield "grouped", lambda a, b, c: a + b * c, [["a", "b"], ["c"]], [[range(4), range(10, 14)], [range(3)]]
    yield "strings", lambda x: f"p{x}", "x", range(6)
    yield "ragged", lambda x: np.ones(x % 3 + 1), "x", range(7)
    yield "none", lambda x: None if x == 3 else float(x), "x", range(6)
    yield "list-output", lambda x: [x, 2 * x], "x", range(5)


@pytest.mark.parametrize("chunk_size", [None, 1, 4, 1000])
@pytest.mark.parametrize("reshape", [True, False])
@pytest.mark.parametrize(("name", "fun", "names", "values"), list(_sweeps()), ids=[s[0] for s in _sweeps()])
def test_dynamic_data_path_matches_joblib(
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int | None,
    reshape: bool,
    name: str,
    fun: Callable[..., Any],
    names: Any,
    values: Any,
) -> None:
    def run() -> tuple[Any, list[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pu.parameterrun(
                fun, names, values, backend="joblib", n_workers=1, pbar_bool=False, reshape=reshape
            )
        return result, sorted(str(w.message) for w in caught)

    expected, expected_warnings = run()
    monkeypatch.setattr(pu, "_parameterrun_joblib", _simulated_dynamic_backend(chunk_size))
    result, result_warnings = run()

    assert result_warnings == expected_warnings
    _assert_same(result, expected)


@pytest.mark.parametrize("chunk_size", [None, 1, 5])
def test_dynamic_data_path_on_error_nan_matches_joblib(monkeypatch: pytest.MonkeyPatch, chunk_size: int | None) -> None:
    def f(x: int, y: int) -> tuple[int, np.ndarray]:
        if (x * y) % 4 == 3:
            raise ValueError(f"bad {x} {y}")
        return x + y, np.full(2, x * y, dtype=np.complex64)

    def run() -> tuple[Any, list[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pu.parameterrun(
                f, ["x", "y"], [range(5), range(6)], backend="joblib", n_workers=1, pbar_bool=False, on_error="nan"
            )
        return result, [str(w.message) for w in caught]

    expected, expected_warnings = run()
    monkeypatch.setattr(pu, "_parameterrun_joblib", _simulated_dynamic_backend(chunk_size))
    result, result_warnings = run()

    assert len(result_warnings) == 1
    # Same count and first failing point; tracebacks differ only in the frames of the two code paths
    assert result_warnings[0].split("\n")[0] == expected_warnings[0].split("\n")[0]
    _assert_same(result, expected)


def _assert_same(result: Any, expected: Any) -> None:
    """Equal values (NaN == NaN), types and dtypes, recursively."""
    if isinstance(expected, np.ndarray):
        assert isinstance(result, np.ndarray)
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        if expected.dtype == object:
            for r, e in zip(result.ravel(), expected.ravel(), strict=True):
                _assert_same(r, e)
        else:
            np.testing.assert_array_equal(result, expected)
    elif isinstance(expected, list):
        assert isinstance(result, list)
        assert len(result) == len(expected)
        for r, e in zip(result, expected, strict=True):
            _assert_same(r, e)
    else:
        assert result == expected or (np.isnan(result) and np.isnan(expected)), (result, expected)


def test_product_order_is_flat_index_order() -> None:
    """The store relies on flat indices following itertools.product (C order)."""
    indices = [list(range(3)), list(range(4))]
    assert [_product_element_from_index(indices, k) for k in range(12)] == [list(p) for p in product(*indices)]


# ---------------------------------------------------------------- the dynamic data path writing to a file


def test_simulated_dynamic_run_with_result_file_matches_joblib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def f(x: int, y: int) -> tuple[Any, np.ndarray]:
        return (x if x < 2 else x + 0.5), np.full(3, x * y, dtype=np.complex64)

    names, values = ["x", "y"], [range(4), range(5)]
    expected = pu.parameterrun(f, names, values, backend="joblib", n_workers=1, pbar_bool=False)

    path = tmp_path / "dyn.npy"
    monkeypatch.setattr(pu, "_parameterrun_joblib", _simulated_dynamic_backend(chunk_size=3, result_file=path))
    result = pu.parameterrun(f, names, values, backend="joblib", n_workers=1, pbar_bool=False, result_file=path)

    assert isinstance(result, list) and isinstance(expected, list)
    for c in range(2):
        assert result[c].dtype == expected[c].dtype
        np.testing.assert_array_equal(result[c], expected[c])
        np.testing.assert_array_equal(np.load(tmp_path / f"dyn_{c}.npy"), expected[c])
    del result
    gc.collect()
