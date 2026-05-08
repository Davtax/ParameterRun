import numpy as np

from parameterrun import parameterrun


def f(a, b, c, d, scale=1):
    return scale * (a + b + c + d)


# a_s = np.linspace(0, 10, 100)
# b_s = np.linspace(0, 20, 50)
# c_s = np.linspace(0, 20, 20)
# d_s = np.linspace(0, 20, 200)

a_s = np.linspace(0, 10, 100)
b_s = np.linspace(0, 20, 50)
c_s = np.linspace(0, 20, 20)
d_s = np.linspace(0, 20, 20)

result = parameterrun(f, param_names=[["a"], ["b"], ["c"], ["d"]], param_values=[[a_s], [b_s], [c_s], [d_s]], scale=2,
                      backend='mpi')

# if rank == 0:
#     expected = 2 * (a_s[:, None, None, None] + b_s[None, :, None, None] + c_s[None, None, :, None] + d_s[
#         None, None, None, :])
#     print(np.all(result == expected))
