"""Small helpers shared by the other modules: logging, names, time formatting, MPI detection."""

import contextlib
import os
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any


def _normalized_time(time_seg: float) -> str:
    """
    Normalize the time to a human-readable format
    """
    if time_seg < 60:
        return f"{time_seg:.2f} s"
    elif time_seg < 3600:
        mins, sec = divmod(time_seg, 60)
        return f"{int(mins)} min {sec:.2f} s"
    else:
        hours, time_seg = divmod(time_seg, 3600)
        mins, sec = divmod(time_seg, 60)
        return f"{int(hours)} h {int(mins)} min {sec:.2f} s"


def _callable_name(fun: Callable[..., Any]) -> str:
    """
    Name of a callable, falling back to its repr for objects without __name__ (e.g. functools.partial)
    """
    return getattr(fun, "__name__", repr(fun))


def _log(message: str, verbose: bool | None = False, hostname: str | None = None) -> None:
    if verbose:
        # Compute timestamp
        time_stamp = datetime.now().strftime("%H:%M:%S, %d/%m/%Y")
        msg = f"{time_stamp}: {message}"
        if hostname is not None:
            msg = f"[{hostname}]: {msg}"

        print(msg, flush=True)


# Environment variables set by MPI launchers in every rank. Used to detect an MPI job without importing mpi4py, since
# importing mpi4py.MPI initialises MPI.
_MPI_SIZE_ENV_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE")


def _mpi_world_size_from_env() -> int:
    for var in _MPI_SIZE_ENV_VARS:
        value = os.environ.get(var)
        if value is not None:
            with contextlib.suppress(ValueError):
                return int(value)

    mpi = sys.modules.get("mpi4py.MPI")  # Already initialised by the user: asking costs nothing
    if mpi is not None:
        return int(mpi.COMM_WORLD.Get_size())

    return 1
