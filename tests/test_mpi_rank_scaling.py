"""Real MPI coverage on both sides of the automatic scheduling threshold."""

import shutil
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from parameterrun import parameterrun
from parameterrun._checkpoint import _Checkpoint

pytestmark = pytest.mark.mpi


@pytest.fixture
def comm() -> Any:
    MPI = pytest.importorskip("mpi4py.MPI")
    if MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip("Run with at least two MPI ranks")
    return MPI.COMM_WORLD


@pytest.fixture
def shared_directory(comm: Any) -> Iterator[Path]:
    directory = Path(
        comm.bcast(tempfile.mkdtemp(prefix="prun_rank_scaling_") if comm.Get_rank() == 0 else None, root=0)
    )
    yield directory
    comm.Barrier()
    if comm.Get_rank() == 0:
        shutil.rmtree(directory)


def test_auto_schedule_changes_which_ranks_compute(comm: Any) -> None:
    rank, size = comm.Get_rank(), comm.Get_size()
    calls: list[int] = []

    def fun(x: int) -> tuple[int, int]:
        calls.append(x)
        return x * x, rank

    n_points = 3 * size + 1
    result = parameterrun(fun, "x", range(n_points), backend="mpi", pbar_bool=False, return_on="all")
    assert isinstance(result, list)
    values, ranks = result
    np.testing.assert_array_equal(values, np.arange(n_points) ** 2)
    all_calls = comm.allgather(calls)
    assert sorted(x for items in all_calls for x in items) == list(range(n_points))
    if size >= 8:
        assert all_calls[0] == []  # Dynamic: rank 0 only manages workers.
        assert np.all((ranks >= 1) & (ranks < size))
    else:
        assert all(all_calls)  # Static: every rank computes, including rank 0.
        assert set(ranks.tolist()) == set(range(size))


@pytest.mark.parametrize("schedule", ["auto", "static", "dynamic"])
@pytest.mark.parametrize("chunk_size", [None, 1])
@pytest.mark.parametrize("return_on", ["root", "all", "none"])
def test_fewer_points_than_ranks(
    comm: Any,
    shared_directory: Path,
    schedule: Literal["auto", "static", "dynamic"],
    chunk_size: int | None,
    return_on: Literal["root", "all", "none"],
) -> None:
    calls: list[int] = []

    def fun(x: int) -> tuple[np.ndarray, str]:
        calls.append(x)
        return np.array([x, x * x]), f"point-{x}"

    result = parameterrun(
        fun,
        "x",
        [0, 1],
        backend="mpi",
        mpi_schedule=schedule,
        mpi_chunk_size=chunk_size,
        pbar_bool=False,
        return_on=return_on,
        result_file=shared_directory / "result.npy" if return_on == "none" else None,
    )
    all_calls = comm.allgather(calls)
    assert sorted(x for items in all_calls for x in items) == [0, 1]
    assert sum(bool(items) for items in all_calls) <= 2  # Other ranks must finish without receiving any points.
    if return_on == "none" or (return_on == "root" and comm.Get_rank() != 0):
        assert result is None
        if return_on == "none" and comm.Get_rank() == 0:
            np.testing.assert_array_equal(np.load(shared_directory / "result_0.npy"), [[0, 0], [1, 1]])
            np.testing.assert_array_equal(np.load(shared_directory / "result_1.npy"), ["point-0", "point-1"])
    else:
        assert isinstance(result, list)
        numbers, labels = result
        np.testing.assert_array_equal(numbers, [[0, 0], [1, 1]])
        np.testing.assert_array_equal(labels, ["point-0", "point-1"])


def test_checkpointed_auto_schedule_only_computes_missing_points(comm: Any, shared_directory: Path) -> None:
    rank, size = comm.Get_rank(), comm.Get_size()
    directory = shared_directory
    n_points = size * 3
    calls: list[int] = []

    def fun(x: int) -> int:
        calls.append(x)
        return x * 2

    saved = np.arange(0, n_points, 2)
    if rank == 0:
        snapshot = _Checkpoint.open(directory, fun, [["x"]], [[list(range(n_points))]], {})
        snapshot.add((saved, [saved * 2], []))
        snapshot.commit()
    comm.Barrier()
    result = parameterrun(
        fun,
        "x",
        range(n_points),
        backend="mpi",
        pbar_bool=False,
        return_on="all",
        checkpoint=directory,
        checkpoint_every=3,
    )
    np.testing.assert_array_equal(result, np.arange(n_points) * 2)
    all_calls = comm.allgather(calls)
    assert sorted(x for items in all_calls for x in items) == list(range(1, n_points, 2))
    assert bool(all_calls[0]) == (size < 8)


def test_auto_schedule_fills_failed_points_on_every_rank(comm: Any) -> None:
    def fun(x: int) -> np.ndarray:
        if x in (1, 3):
            raise ValueError("failed point")
        return np.array([x, x * 2], dtype=float)

    n_points = comm.Get_size() + 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = parameterrun(
            fun,
            "x",
            range(n_points),
            backend="mpi",
            pbar_bool=False,
            return_on="all",
            on_error="nan",
        )
    expected = np.array([[x, x * 2] for x in range(n_points)], dtype=float)
    expected[[1, 3]] = np.nan
    np.testing.assert_array_equal(result, expected)
    if comm.Get_rank() == 0:
        assert len(caught) == 1
        assert f"2 of {n_points} points raised" in str(caught[0].message)
    else:
        assert not caught
