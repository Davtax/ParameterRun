import gc
import shutil
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

import parameterrun._store as pu_store
import parameterrun.parallel_utils as pu
from parameterrun._mpi_static import _parameterrun_mpi
from parameterrun._pseudo_random import local_values


@pytest.mark.mpi
def test_parameterrun_mpi_numerical_equivalence_and_root_return() -> None:
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD

    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")

    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([10.0, 20.0])

    def f(x: float, y: float, scale: float = 1.0) -> float:
        return scale * (x + y)

    result = pu.parameterrun(
        f,
        param_names=[["x"], ["y"]],
        param_values=[[xs], [ys]],
        scale=2.0,
        backend="mpi",
        pbar_bool=False,
    )

    expected = 2.0 * (xs[:, None] + ys[None, :])

    if comm.Get_rank() == 0:
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected)
    else:
        assert result is None


@pytest.mark.mpi
def test_parameterrun_mpi_multiple_outputs_and_seed_kwarg() -> None:
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD

    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")

    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([10.0, 20.0])

    # `seed` is a user kwarg: it must reach the function and not be captured by the MPI backend
    def f(x: float, y: float, seed: int) -> tuple[float, float]:
        return x + y + seed, x * y

    result = pu.parameterrun(
        f,
        param_names=[["x"], ["y"]],
        param_values=[[xs], [ys]],
        seed=100,
        backend="mpi",
        pbar_bool=False,
    )

    if comm.Get_rank() == 0:
        assert isinstance(result, list)
        assert len(result) == 2
        np.testing.assert_allclose(result[0], xs[:, None] + ys[None, :] + 100)
        np.testing.assert_allclose(result[1], xs[:, None] * ys[None, :])
    else:
        assert result is None


@pytest.mark.mpi
def test_parameterrun_mpi_explicit_shuffle_seed_is_respected() -> None:
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD

    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")

    seed = 1234
    xs = list(range(10))
    rank = comm.Get_rank()

    def f(x: int) -> tuple[int, int]:
        return x, rank  # Record which rank computed each value

    # Only root's seed matters: it is broadcast to the other ranks
    result, n_outputs = _parameterrun_mpi(
        f, [["x"]], [[xs]], desc=None, pbar_bool=False, seed=seed if rank == 0 else None
    )

    if rank == 0:
        assert n_outputs == 2
        assert list(result[0]) == xs  # The backend returns one array per channel
        expected_rank = {v: r for r in range(comm.Get_size()) for v in local_values(len(xs), comm.Get_size(), r, seed)}
        assert list(result[1]) == [expected_rank[x] for x in xs]


@pytest.mark.mpi
def test_parameterrun_mpi_shuffle_seed_reaches_backend_and_seed_kwarg_reaches_function() -> None:
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD

    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")

    shuffle_seed = 98765
    xs = np.arange(12)
    rank = comm.Get_rank()

    def f(x: int, seed: int) -> tuple[int, int]:
        return x + seed, rank  # Record which rank computed each value

    runs = [
        pu.parameterrun(f, "x", xs, seed=1000, shuffle_seed=shuffle_seed, backend="mpi", pbar_bool=False)
        for _ in range(2)
    ]

    if rank == 0:
        size = comm.Get_size()
        rank_of = {v: r for r in range(size) for v in local_values(len(xs), size, r, shuffle_seed)}
        expected_rank = [rank_of[int(x)] for x in xs]
        for result in runs:
            assert isinstance(result, list)
            np.testing.assert_array_equal(result[0], xs + 1000)  # The user's `seed` reached the function
            np.testing.assert_array_equal(result[1], expected_rank)  # `shuffle_seed` fixed the distribution
    else:
        assert runs == [None, None]


@pytest.mark.mpi
def test_parameterrun_mpi_progress_bar_writes_plain_lines(capsys: pytest.CaptureFixture[str]) -> None:
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD

    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")

    result = pu.parameterrun(lambda x: 2 * x, "x", list(range(8)), backend="mpi", pbar_bool=True, desc="mpi-sweep")
    output = capsys.readouterr().out

    if comm.Get_rank() == 0:
        np.testing.assert_array_equal(result, 2 * np.arange(8))
        assert "\r" not in output  # Captured stdout is not a terminal: one plain line per refresh
        assert "mpi-sweep" in output
        assert "Waiting for workers" in output
        assert "3/3" in output  # All three workers reported back
    else:
        assert result is None
        assert output == ""  # Only rank 0 draws a bar


@pytest.mark.mpi
def test_parameterrun_mpi_on_error_nan_fills_failed_points_on_every_rank() -> None:
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD

    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")

    shuffle_seed = 4321
    xs = list(range(20))
    # Make every point of rank 0 fail: it then has no output shape of its own and must learn it from the others
    root_points = {int(v) for v in local_values(len(xs), comm.Get_size(), 0, shuffle_seed)}
    failing = root_points | {7, 13}

    def f(x: int) -> tuple[np.ndarray, int]:
        if x in failing:
            raise ValueError(f"bad {x}")
        return np.full(3, x, dtype=complex), x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pu.parameterrun(f, "x", xs, backend="mpi", pbar_bool=False, on_error="nan", shuffle_seed=shuffle_seed)

    messages = [str(w.message) for w in caught]
    if comm.Get_rank() == 0:
        assert isinstance(result, list)
        vectors, ints = result
        assert vectors.shape == (20, 3)
        assert vectors.dtype == np.complex128
        assert ints.dtype == np.float64
        for x in xs:
            if x in failing:
                assert np.isnan(vectors[x]).all()
                assert np.isnan(ints[x])
            else:
                np.testing.assert_array_equal(vectors[x], np.full(3, x))
                assert ints[x] == x
        first = min(failing)
        assert len(messages) == 1
        assert messages[0].startswith(f"{len(failing)} of 20 points raised an exception and were filled with NaN")
        assert f"First failure at {{'x': {first}}}: ValueError: bad {first}" in messages[0]
    else:
        assert result is None
        assert messages == []  # Only root reports


@pytest.mark.mpi
def test_parameterrun_mpi_on_error_nan_raises_on_every_rank_when_all_points_fail() -> None:
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD

    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")

    def f(x: int) -> int:
        raise ValueError("always")

    # Two points on four ranks: two ranks have no work at all, and must still raise instead of hanging
    with pytest.raises(RuntimeError, match="All 2 points raised an exception"):
        pu.parameterrun(f, "x", [1, 2], backend="mpi", pbar_bool=False, on_error="nan")


# ---------------------------------------------------------------- dynamic schedule (phase 3)


def _comm_of_4() -> Any:
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")
    return comm


def _assert_identical(result: Any, expected: Any) -> None:
    """Same type, dtype, shape and values (NaN == NaN), recursively."""
    if isinstance(expected, np.ndarray):
        assert isinstance(result, np.ndarray)
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        if expected.dtype == object:
            for r, e in zip(result.ravel(), expected.ravel(), strict=True):
                _assert_identical(r, e)
        else:
            np.testing.assert_array_equal(result, expected)
    elif isinstance(expected, list):
        assert isinstance(result, list)
        assert len(result) == len(expected)
        for r, e in zip(result, expected, strict=True):
            _assert_identical(r, e)
    else:
        both_nan = isinstance(expected, float) and np.isnan(expected) and np.isnan(result)
        assert both_nan or result == expected, (result, expected)


def _complex_outputs(x: float, y: float) -> tuple[np.ndarray, float]:
    return np.array([[x, y], [x * y, x - y]]) * (1 + 0.5j), x / (1 + y)


def _python_scalars(x: int) -> float:
    return x if x < 3 else x + 0.25


@pytest.mark.mpi
@pytest.mark.parametrize("chunk_size", [None, 1, 3, 10_000])
@pytest.mark.parametrize(
    ("fun", "names", "values"),
    [
        (_complex_outputs, ["x", "y"], [np.linspace(0, 1, 13), np.arange(7.0)]),
        (_python_scalars, "x", list(range(10))),
        (lambda x: np.ones(x % 3 + 1), "x", list(range(9))),  # Ragged: warning and flat lists
        (lambda x: f"point {x}", "x", list(range(5))),  # Non-numeric
    ],
    ids=["complex-multi", "int-then-float", "ragged", "strings"],
)
def test_parameterrun_mpi_dynamic_is_identical_to_static(
    fun: Callable[..., Any], names: Any, values: Any, chunk_size: int | None
) -> None:
    comm = _comm_of_4()

    def run(schedule: Literal["static", "dynamic"]) -> tuple[Any, list[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pu.parameterrun(
                fun, names, values, backend="mpi", pbar_bool=False, mpi_schedule=schedule, mpi_chunk_size=chunk_size
            )
        return result, sorted(str(w.message) for w in caught)

    static, static_warnings = run("static")
    dynamic, dynamic_warnings = run("dynamic")

    if comm.Get_rank() == 0:
        _assert_identical(dynamic, static)
        assert dynamic_warnings == static_warnings
    else:
        assert static is None and dynamic is None


@pytest.mark.mpi
@pytest.mark.parametrize("n_points", [1, 2, 3])  # Fewer points than (or as many as) the 3 workers
def test_parameterrun_mpi_dynamic_with_few_points(n_points: int) -> None:
    comm = _comm_of_4()
    result = pu.parameterrun(lambda x: 10 * x, "x", list(range(n_points)), backend="mpi", mpi_schedule="dynamic")
    if comm.Get_rank() == 0:
        np.testing.assert_array_equal(result, 10 * np.arange(n_points))
    else:
        assert result is None


@pytest.mark.mpi
def test_parameterrun_mpi_dynamic_balances_uneven_costs_and_forwards_seed_kwarg() -> None:
    comm = _comm_of_4()
    rank = comm.Get_rank()

    # The first points are slow: with a static split, the rank that drew them would finish long after the others
    def f(x: int, seed: int) -> tuple[int, int]:
        time.sleep(0.05 if x < 4 else 0.002)
        return x + seed, rank

    result = pu.parameterrun(
        f, "x", list(range(60)), seed=1000, backend="mpi", pbar_bool=False, mpi_schedule="dynamic", mpi_chunk_size=2
    )

    if rank == 0:
        assert isinstance(result, list)
        values, ranks = result
        np.testing.assert_array_equal(values, np.arange(60) + 1000)  # The user's `seed` reached the function
        counts = {r: int((ranks == r).sum()) for r in range(4)}
        assert counts[0] == 0  # The manager does not compute
        assert all(counts[r] > 0 for r in (1, 2, 3)), counts  # Every worker received work
    else:
        assert result is None


@pytest.mark.mpi
def test_parameterrun_mpi_dynamic_on_error_nan_and_none_outputs() -> None:
    comm = _comm_of_4()

    def f(x: int) -> tuple[float, Any]:
        if x in (3, 11):
            raise ValueError(f"bad {x}")
        return float(x), None if x == 5 else x

    def run(schedule: Literal["static", "dynamic"]) -> tuple[Any, list[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pu.parameterrun(
                f, "x", list(range(16)), backend="mpi", pbar_bool=False, on_error="nan", mpi_schedule=schedule
            )
        # Tracebacks differ in the frames of the two code paths: compare the first lines
        return result, sorted(str(w.message).split("\n")[0] for w in caught)

    static, static_warnings = run("static")
    dynamic, dynamic_warnings = run("dynamic")

    if comm.Get_rank() == 0:
        assert np.isnan(dynamic[0][[3, 11]]).all()
        _assert_identical(dynamic[1], static[1])  # Object array with None and NaN
        np.testing.assert_array_equal(dynamic[0], static[0])
        assert dynamic_warnings == static_warnings
        assert any(w.startswith("2 of 16 points raised an exception") for w in dynamic_warnings)
        assert any("1 of 16 results are None" in w for w in dynamic_warnings)


@pytest.mark.mpi
def test_parameterrun_mpi_dynamic_raises_on_every_rank_when_all_points_fail() -> None:
    _comm_of_4()

    def f(x: int) -> int:
        raise ValueError("always")

    with pytest.raises(RuntimeError, match="All 5 points raised an exception"):
        pu.parameterrun(f, "x", list(range(5)), backend="mpi", pbar_bool=False, on_error="nan", mpi_schedule="dynamic")


@pytest.mark.mpi
def test_parameterrun_mpi_dynamic_progress_bar_counts_all_points(capsys: pytest.CaptureFixture[str]) -> None:
    comm = _comm_of_4()
    result = pu.parameterrun(
        lambda x: 2 * x, "x", list(range(30)), backend="mpi", pbar_bool=True, desc="dyn-sweep", mpi_schedule="dynamic"
    )
    output = capsys.readouterr().out

    if comm.Get_rank() == 0:
        np.testing.assert_array_equal(result, 2 * np.arange(30))
        assert "30/30" in output  # One global bar over every point, not rank 0's share
        assert "Waiting for workers" not in output
    else:
        assert output == ""


@pytest.mark.mpi
def test_parameterrun_mpi_auto_schedule_is_static_on_4_ranks() -> None:
    comm = _comm_of_4()
    rank = comm.Get_rank()
    result = pu.parameterrun(lambda x: rank, "x", list(range(8)), backend="mpi", pbar_bool=False)
    if rank == 0:
        assert 0 in set(np.asarray(result).tolist())  # Rank 0 computes too: static


# ---------------------------------------------------------------- result delivery (phase 4)

SCHEDULES: list[Literal["static", "dynamic"]] = ["static", "dynamic"]


def _expected(fun: Callable[..., Any], names: Any, values: Any, **kwargs: Any) -> Any:
    """The same sweep with joblib on this rank: the reference for both MPI schedules."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pu.parameterrun(fun, names, values, backend="joblib", n_workers=1, pbar_bool=False, **kwargs)


def _shared_tmp_dir(comm: Any) -> Path:
    path = tempfile.mkdtemp(prefix="prun_mpi_") if comm.Get_rank() == 0 else None
    return Path(comm.bcast(path, root=0))


def _vector_and_int(x: int, y: int) -> tuple[np.ndarray, Any]:
    return np.array([x, y, x * y]) * (1 - 1j), (x * y if x < 2 else x * y + 0.5)


@pytest.mark.mpi
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize(
    ("fun", "names", "values"),
    [
        (_vector_and_int, ["x", "y"], [list(range(5)), list(range(3))]),  # Arrays: raw-byte Bcast
        (lambda x: f"p{x}", "x", list(range(7))),  # String array: raw bytes too
        (lambda x: None if x == 3 else x, "x", list(range(6))),  # Object array: pickled
        (lambda x: np.ones(x % 3 + 1), "x", list(range(6))),  # Ragged: flat lists, pickled
    ],
    ids=["arrays", "strings", "objects", "ragged"],
)
def test_parameterrun_mpi_return_on_all_gives_every_rank_the_result(
    schedule: Literal["static", "dynamic"], fun: Callable[..., Any], names: Any, values: Any
) -> None:
    _comm_of_4()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pu.parameterrun(
            fun, names, values, backend="mpi", pbar_bool=False, mpi_schedule=schedule, return_on="all"
        )
    _assert_identical(result, _expected(fun, names, values))


@pytest.mark.mpi
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_parameterrun_mpi_transfers_in_small_blocks(
    monkeypatch: pytest.MonkeyPatch, schedule: Literal["static", "dynamic"]
) -> None:
    """A tiny message limit forces many blocks: static rows, dynamic partial messages and Bcast pieces."""
    _comm_of_4()
    monkeypatch.setattr(pu_store, "_MAX_MESSAGE_BYTES", 40)
    names, values = ["x", "y"], [list(range(9)), list(range(7))]
    result = pu.parameterrun(
        _vector_and_int, names, values, backend="mpi", pbar_bool=False, mpi_schedule=schedule, return_on="all"
    )
    _assert_identical(result, _expected(_vector_and_int, names, values))


@pytest.mark.mpi
def test_parameterrun_mpi_return_on_all_raises_everywhere_when_rank_0_fails() -> None:
    comm = _comm_of_4()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if comm.Get_rank() == 0:
            with pytest.raises(ValueError, match="Could not reshape output channel 0"):
                pu.parameterrun(lambda x: np.ones(x), "x", [1, 2, 3], backend="mpi", reshape="strict", return_on="all")
        else:
            with pytest.raises(RuntimeError, match="parameterrun failed on rank 0: ValueError: Could not reshape"):
                pu.parameterrun(lambda x: np.ones(x), "x", [1, 2, 3], backend="mpi", reshape="strict", return_on="all")


@pytest.mark.mpi
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_parameterrun_mpi_result_file(schedule: Literal["static", "dynamic"]) -> None:
    comm = _comm_of_4()
    tmp = _shared_tmp_dir(comm)

    def f(x: int, y: int) -> tuple[np.ndarray, Any]:
        if (x, y) == (3, 1):
            raise ValueError("bad point")
        return _vector_and_int(x, y)

    names, values = ["x", "y"], [list(range(5)), list(range(4))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pu.parameterrun(
            f,
            names,
            values,
            backend="mpi",
            pbar_bool=False,
            mpi_schedule=schedule,
            on_error="nan",
            return_on="none",
            result_file=tmp / "sweep.npy",
        )
    assert result is None  # On every rank

    comm.Barrier()
    if comm.Get_rank() == 0:
        expected = _expected(f, names, values, on_error="nan")
        for c in range(2):
            on_disk = np.load(tmp / f"sweep_{c}.npy")
            assert on_disk.dtype == expected[c].dtype  # The int channel was promoted to float in the file
            np.testing.assert_array_equal(on_disk, expected[c])
        assert any("1 of 20 points raised an exception" in str(w.message) for w in caught)
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.mpi
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_parameterrun_mpi_result_file_returns_arrays_backed_by_the_file(schedule: Literal["static", "dynamic"]) -> None:
    comm = _comm_of_4()
    tmp = _shared_tmp_dir(comm)
    path = tmp / "squares.npy"
    result = pu.parameterrun(
        lambda x: x * x, "x", list(range(50)), backend="mpi", pbar_bool=False, mpi_schedule=schedule, result_file=path
    )
    comm.Barrier()
    if comm.Get_rank() == 0:
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.arange(50) ** 2)
        result[0] = -1  # Copy-on-write: the file keeps the computed value
        assert np.load(path)[0] == 0
        del result
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        assert result is None


@pytest.mark.mpi
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_parameterrun_mpi_mixed_list_and_array_channels(schedule: Literal["static", "dynamic"]) -> None:
    """Channel 0 is ragged at one point only (on one rank): it becomes a list channel; channel 1 stays an array."""
    comm = _comm_of_4()

    def f(x: int) -> tuple[np.ndarray, int]:
        return np.ones(3 if x == 7 else 2), x

    def run() -> tuple[Any, list[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pu.parameterrun(f, "x", list(range(12)), backend="mpi", pbar_bool=False, mpi_schedule=schedule)
        return result, sorted(str(w.message) for w in caught)

    result, messages = run()
    if comm.Get_rank() == 0:
        expected = _expected(f, "x", list(range(12)))
        _assert_identical(result, expected)
        assert any("Could not reshape output channel 0" in m for m in messages)


@pytest.mark.mpi
def test_parameterrun_mpi_static_inconsistent_number_of_outputs_raises_on_every_rank() -> None:
    _comm_of_4()

    def f(x: int) -> Any:
        return (x, x) if x < 5 else x

    with pytest.raises(ValueError, match="outputs"):
        pu.parameterrun(f, "x", list(range(12)), backend="mpi", pbar_bool=False, mpi_schedule="static")
