"""
Where the result goes after rank 0 has it: ``.npy`` files (``result_file``) and every rank (``return_on="all"``).
"""

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from . import _store
from ._results import _reshape_channel_result
from ._store import _channel_paths

_RETURN_ON = ("root", "all", "none")


def _check_result_file(result_file: str | os.PathLike[str]) -> str | None:
    """Problem with ``result_file`` (checked before computing anything), or None."""
    path = Path(result_file)
    if path.suffix != ".npy":
        return f"result_file must end in .npy, got {str(result_file)!r}"
    parent = path.parent.resolve()
    if not parent.is_dir():
        return f"The directory of result_file does not exist: {parent}"
    if not os.access(parent, os.W_OK):
        return f"The directory of result_file is not writable: {parent}"
    return None


def _mapped_file(channel: Any) -> Path | None:
    """File behind a memory-mapped channel of the result store, if any."""
    filename = getattr(channel, "filename", None) if isinstance(channel, np.memmap) else None
    return Path(filename).resolve() if filename else None


def _write_result_files(
    result: list[Any], n_values: list[int], result_file: str | os.PathLike[str]
) -> None:  # Rank 0 only
    """
    Save each output channel, reshaped to the grid, to its ``.npy`` file. Channels the MPI result store already keeps
    in that file are skipped. Channels that cannot form one plain array (ragged outputs, None, other Python objects)
    are not written, with a warning.
    """
    for c, (channel, path) in enumerate(zip(result, _channel_paths(result_file, len(result)), strict=True)):
        if _mapped_file(channel) == path.resolve():
            continue
        try:
            array = _reshape_channel_result(channel, n_values)
        except ValueError as err:
            reason = str(err)
        else:
            if not array.dtype.hasobject:
                np.save(path, array)
                continue
            reason = "the outputs are Python objects (an object array would need pickling)"
        warnings.warn(
            f"Output channel {c} was not written to {path}: {reason}. A .npy file holds one array.",
            RuntimeWarning,
            stacklevel=3,
        )


def _item_layout(item: Any) -> tuple[Any, ...]:
    if isinstance(item, np.ndarray) and not item.dtype.hasobject:
        return ("array", item.shape, item.dtype)
    return ("pickle",)


def _broadcast_result(comm: Any, result: Any, root_error: BaseException | None) -> Any:  # pragma: no cover
    """
    Give every rank rank 0's final result. Numeric arrays travel as raw bytes with ``Bcast`` (in blocks of at most
    ``_MAX_MESSAGE_BYTES``), after their shape and dtype; anything else is pickled. If rank 0 failed while finishing
    the result, it re-raises and the other ranks raise a RuntimeError, instead of waiting forever.
    """
    rank = comm.Get_rank()
    layout: tuple[Any, ...] | None = None
    if rank == 0:
        if root_error is not None:
            layout = ("error", f"{type(root_error).__name__}: {root_error}")
        elif isinstance(result, np.ndarray):
            layout = ("single", _item_layout(result))
        elif isinstance(result, list) and any(isinstance(item, np.ndarray) for item in result):
            layout = ("list", [_item_layout(item) for item in result])
        else:
            layout = ("pickle",)
    layout = comm.bcast(layout, root=0)

    if layout[0] == "error":
        if root_error is not None:
            raise root_error
        raise RuntimeError(f"parameterrun failed on rank 0: {layout[1]}")
    if layout[0] == "pickle":
        return comm.bcast(result, root=0)
    if layout[0] == "single":
        return _broadcast_item(comm, result if rank == 0 else None, layout[1])
    return [
        _broadcast_item(comm, result[i] if rank == 0 else None, item_layout) for i, item_layout in enumerate(layout[1])
    ]


def _broadcast_item(comm: Any, item: Any, layout: tuple[Any, ...]) -> Any:  # pragma: no cover
    from mpi4py import MPI

    if layout[0] == "pickle":
        return comm.bcast(item, root=0)

    if comm.Get_rank() == 0:
        array = np.ascontiguousarray(item)
    else:
        array = np.empty(layout[1], dtype=np.dtype(layout[2]))
    data = array.reshape(-1).view(np.uint8)
    step = _store._MAX_MESSAGE_BYTES
    for start in range(0, len(data), step):
        comm.Bcast([data[start : start + step], MPI.BYTE], root=0)
    return item if comm.Get_rank() == 0 else array
