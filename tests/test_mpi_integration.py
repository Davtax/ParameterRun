import numpy as np
import pytest

import parameterrun.parallel_utils as pu


@pytest.mark.mpi
def test_parameterrun_mpi_numerical_equivalence_and_root_return():
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD

    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")

    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([10.0, 20.0])

    def f(x, y, scale=1.0):
        return scale * (x + y)

    result = pu.parameterrun(f, param_names=[["x"], ["y"]], param_values=[[xs], [ys]], scale=2.0, backend="mpi",
                             pbar_bool=False, )

    expected = 2.0 * (xs[:, None] + ys[None, :])

    if comm.Get_rank() == 0:
        assert isinstance(result, np.ndarray)

        return np.testing.assert_allclose(result, expected)
    else:
        assert result is None
        return None
