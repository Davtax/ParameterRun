import numpy as np

from parameterrun import parameterrun


def f(x: float, y: float, scale: float = 1) -> float:
    return scale * (x + y)


xs = np.linspace(0, 10, 100)
ys = np.linspace(0, 20, 50)

result = parameterrun(
    f,
    param_names=[["x"], ["y"]],
    param_values=[[xs], [ys]],
    n_workers=1,
    scale=2,
)
