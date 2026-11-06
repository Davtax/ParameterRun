"""Bounded transfers of serialized results, including a single oversized output."""

import pickle
import tempfile
from typing import Any

import numpy as np

from . import _store


def _send_payload(comm: Any, payload: Any, dest: int, tag: int) -> None:
    from mpi4py import MPI

    # Spill large serialized payloads to disk rather than duplicating them in memory.
    with tempfile.SpooledTemporaryFile(max_size=_store._MAX_MESSAGE_BYTES) as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        comm.send(file.tell(), dest=dest, tag=tag)
        file.seek(0)
        while block := file.read(_store._MAX_MESSAGE_BYTES):
            comm.Send([np.frombuffer(block, dtype=np.uint8), MPI.BYTE], dest=dest, tag=tag)


def _recv_payload(comm: Any, nbytes: int, source: int, tag: int) -> Any:
    from mpi4py import MPI

    with tempfile.SpooledTemporaryFile(max_size=_store._MAX_MESSAGE_BYTES) as file:
        while nbytes:
            block = np.empty(min(nbytes, _store._MAX_MESSAGE_BYTES), dtype=np.uint8)
            comm.Recv([block, MPI.BYTE], source=source, tag=tag)
            file.write(block.tobytes())
            nbytes -= len(block)
        file.seek(0)
        return pickle.load(file)


def _finish_store(comm: Any, store: Any, param_names: Any, param_values: Any) -> Any:
    """Finalize on root, then release every rank with either the result or an error."""
    result: Any = ([], 0)
    error = None
    if comm.Get_rank() == 0:
        try:
            result = store.finalize(param_names, param_values)
        except Exception as err:
            error = err
    message = comm.bcast(f"{type(error).__name__}: {error}" if error is not None else None, root=0)
    if error is not None:
        raise error
    if message is not None:
        raise RuntimeError(f"parameterrun failed on rank 0: {message}")
    return result
