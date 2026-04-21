import numpy as np
from mpi4py import MPI

from parameterrun import parameterrun


def f(x, y, scale=1):
    return scale * (x + y)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

xs = np.linspace(0, 10, 100)
ys = np.linspace(0, 20, 50)

result = parameterrun(f, param_names=[["x"], ["y"]], param_values=[[xs], [ys]], scale=2, backend='mpi')

if rank == 0:
    print(np.all(result == 2 * (xs[:, None] + ys[None, :])))
