from time import sleep

import numpy as np

from parameterrun import parameterrun


def f(x, y, scale=1):
    sleep(0.1)  # Simulate a long computation
    return scale * (x + y)


xs = np.linspace(0, 10, 1000)
ys = np.linspace(0, 20, 500)

result = parameterrun(f, param_names=[["x"], ["y"]], param_values=[[xs], [ys]], n_workers=1, scale=2, )
