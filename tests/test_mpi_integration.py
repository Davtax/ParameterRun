import numpy as np
import pytest

import parameterrun.parallel_utils as pu


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
    result, n_outputs = pu._parameterrun_mpi(
        f, [["x"]], [[xs]], desc=None, pbar_bool=False, seed=seed if rank == 0 else None
    )

    if rank == 0:
        assert n_outputs == 2
        assert result[0] == xs
        expected_rank = {
            v: r for r in range(comm.Get_size()) for v in pu.local_values(len(xs), comm.Get_size(), r, seed)
        }
        assert result[1] == [expected_rank[x] for x in xs]
