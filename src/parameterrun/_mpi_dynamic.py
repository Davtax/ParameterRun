"""
MPI backend, dynamic schedule: rank 0 is a manager that hands out chunks of points on demand; the other ranks compute.

Fast ranks simply ask for more work, so differences in speed between nodes and cores, and uneven task costs, no longer
leave the whole job waiting for the slowest rank.

Protocol (pickled point-to-point messages, all small):

* worker -> manager, ``_TAG_RESULT``: results of the last chunk (``None`` at start-up), and a request for more work.
* worker -> manager, ``_TAG_PARTIAL``: part of a chunk's results, sent early when they exceed ``_MAX_MESSAGE_BYTES``.
* worker -> manager, ``_TAG_ERROR``: text to print; the manager prints it once and aborts the job.
* manager -> worker, ``_TAG_WORK``: ``(start, stop)``, positions in the seeded permutation of the flat indices. Workers
  map positions to flat indices themselves, so the order is never materialized.
* manager -> worker, ``_TAG_STOP``: no more work.

At the end the manager broadcasts ``None``, or the message of the error that every rank must raise (all points failed).
"""

import sys
import traceback
from collections.abc import Callable
from math import prod
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from . import _store
from ._errors import _abort_header, _evaluate_point, _FailedPoint, _print_and_abort
from ._inputs import _get_iteration, _product_element_from_index
from ._mpi_static import _shared_seed
from ._progress import _pbar_options
from ._pseudo_random import _permute_in_range
from ._store import Channel, _ResultStore, _stack_channel
from ._utils import _log

_TAG_RESULT = 11
_TAG_PARTIAL = 12
_TAG_ERROR = 13
_TAG_WORK = 14
_TAG_STOP = 15


# Rough size of a non-array output, only used to decide when to send early
_OBJECT_NBYTES = 64

Payload = tuple[np.ndarray, list[Channel], list[tuple[int, _FailedPoint]]]


def _guided_chunk_size(remaining: int, n_workers: int, chunk_size: int | None = None) -> int:
    """
    Number of points in the next chunk. Guided by default: a quarter of each worker's share of what is left, so chunks
    are large at the start (few messages) and shrink to one point at the end (no tail).
    """
    if chunk_size is not None:
        return max(1, min(chunk_size, remaining))
    return max(1, remaining // (4 * n_workers))


class _ChunkBuffer:
    """Worker-side results of the current chunk, split into output channels before sending."""

    def __init__(self) -> None:
        self.n_outputs: int | None = None  # Fixed by the first successful point of this worker
        self.clear()

    def clear(self) -> None:
        self.indices: list[int] = []
        self.values: list[tuple[Any, ...]] = []
        self.failures: list[tuple[int, _FailedPoint]] = []
        self.nbytes = 0

    def __len__(self) -> int:
        return len(self.indices) + len(self.failures)

    def add(self, index: int, value: Any, params: dict[str, Any]) -> None:
        if isinstance(value, _FailedPoint):
            self.failures.append((index, value))
            self.nbytes += len(value.traceback)
            return

        outputs = value if isinstance(value, tuple) else (value,)
        if self.n_outputs is None:
            self.n_outputs = len(outputs)
        elif len(outputs) != self.n_outputs:
            raise ValueError(
                f"fun returned {len(outputs)} outputs at {params}, but {self.n_outputs} at an earlier point. Return "
                f"the same number of outputs (a tuple of the same length) for every point."
            )
        self.indices.append(index)
        self.values.append(outputs)
        self.nbytes += sum(output.nbytes if isinstance(output, np.ndarray) else _OBJECT_NBYTES for output in outputs)

    def payload(self) -> Payload:
        channels: list[Channel] = []
        if self.values:
            assert self.n_outputs is not None  # For type checker
            channels = [_stack_channel([outputs[c] for outputs in self.values]) for c in range(self.n_outputs)]
        return np.asarray(self.indices, dtype=np.int64), channels, self.failures


def _manager(
    comm: Any,
    n_total: int,
    store: _ResultStore,
    pbar: Any,
    chunk_size: int | None,
) -> None:  # pragma: no cover
    from mpi4py import MPI

    n_workers = comm.Get_size() - 1
    status = MPI.Status()
    next_position = 0
    active = n_workers

    while active:
        payload = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source, tag = status.Get_source(), status.Get_tag()

        if tag == _TAG_ERROR:
            _print_and_abort(comm, payload)

        if payload is not None:
            indices, channels, failures = payload
            store.write(indices, channels)
            store.add_failures(failures)
            pbar.update(len(indices) + len(failures))

        if tag == _TAG_PARTIAL:  # The worker is still on its chunk
            continue

        if next_position < n_total:
            stop = next_position + _guided_chunk_size(n_total - next_position, n_workers, chunk_size)
            comm.send((next_position, stop), dest=source, tag=_TAG_WORK)
            next_position = stop
        else:
            comm.send(None, dest=source, tag=_TAG_STOP)
            active -= 1


def _worker(
    comm: Any,
    fun: Callable[..., Any],
    param_names: list[list[str]],
    param_values: list[list[list[Any]]],
    fun_kwargs: dict[str, Any],
    on_error: str,
    seed: int,
) -> int:  # pragma: no cover
    from mpi4py import MPI

    n_values = [len(group[0]) for group in param_values]
    indices = [list(range(n_value)) for n_value in n_values]
    n_total = prod(n_values)

    status = MPI.Status()
    buffer = _ChunkBuffer()
    n_computed = 0

    comm.send(None, dest=0, tag=_TAG_RESULT)  # Ready
    while True:
        work = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == _TAG_STOP:
            return n_computed

        start, stop = work
        for position in range(start, stop):
            index = _permute_in_range(position, n_total, seed)
            params = _get_iteration(_product_element_from_index(indices, index), param_names, param_values)
            try:
                value = _evaluate_point(fun, params, fun_kwargs, on_error)
            except Exception as err:  # Only with on_error="raise"; the note on err holds the parameters
                header = _abort_header(comm.Get_rank(), comm.Get_size(), MPI.Get_processor_name())
                comm.send(header + "".join(traceback.format_exception(err)), dest=0, tag=_TAG_ERROR)
                comm.recv(source=0, tag=_TAG_STOP)  # Never arrives: the manager aborts the job
                raise  # Unreachable unless the abort fails

            buffer.add(index, value, params)
            # Send early, so messages stay far below the 2 GB limit of pickled MPI messages even for large chunks
            if buffer.nbytes >= _store._MAX_MESSAGE_BYTES:
                n_computed += len(buffer)
                comm.send(buffer.payload(), dest=0, tag=_TAG_PARTIAL)
                buffer.clear()

        n_computed += len(buffer)
        comm.send(buffer.payload(), dest=0, tag=_TAG_RESULT)
        buffer.clear()


def _parameterrun_mpi_dynamic(
    fun: Callable[..., Any],
    param_names: list[list[str]],
    param_values: list[list[list[Any]]],
    desc: str | None,
    pbar_bool: bool = True,
    verbose: bool | None = False,
    pbar_leave: bool | None = True,
    seed: int | None = None,
    fun_kwargs: dict[str, Any] | None = None,
    on_error: str = "raise",
    chunk_size: int | None = None,
    result_file: Any = None,
) -> tuple[list[Channel], int]:  # pragma: no cover
    """
    Rank 0 returns the channels in flat (C-order) index order: numeric outputs of a fixed shape as one array of shape
    ``(n_total, *output_shape)``, anything else as a flat list. The other ranks return ``([], 0)``. Needs >= 2 ranks.
    """
    # fun_kwargs is an explicit dict (not **kwargs) so user kwargs such as `seed` cannot collide with our parameters
    fun_kwargs = fun_kwargs or {}

    from mpi4py import MPI

    # A private communicator per call: messages of a call that failed half-way can never reach a later call
    comm = MPI.COMM_WORLD.Dup()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise ValueError("The dynamic MPI schedule needs at least 2 ranks (1 manager + workers)")

    n_total = prod(len(group[0]) for group in param_values)
    seed = _shared_seed(comm, seed)
    hostname = MPI.Get_processor_name()

    if rank == 0:
        _log(
            f"Dynamic schedule: 1 manager + {size - 1} workers, "
            f"{'guided chunks' if chunk_size is None else f'chunks of {chunk_size} points'}",
            verbose,
            hostname=hostname,
        )
        store = _ResultStore(n_total, [len(group[0]) for group in param_values], result_file)
        # Output goes through the MPI launcher, which forwards it line by line
        pbar = tqdm(total=n_total, desc=desc, leave=pbar_leave, **_pbar_options(pbar_bool, sys.stdout, forwarded=True))
        try:
            _manager(comm, n_total, store, pbar, chunk_size)
        except Exception as err:  # A bug or inconsistent outputs: don't leave the workers waiting
            _print_and_abort(
                comm, f"parameterrun: the manager (rank 0) failed. Aborting the MPI job.\n{traceback.format_exc()}"
            )
            raise err
        pbar.close()

        try:
            result = store.finalize(param_names, param_values)
        except RuntimeError as err:  # Every point failed: every rank raises
            comm.bcast(str(err), root=0)
            raise
        comm.bcast(None, root=0)
        comm.Free()
        return result

    try:
        n_computed = _worker(comm, fun, param_names, param_values, fun_kwargs, on_error, seed)
    except Exception:  # Not from fun (that is handled inside): a bug or inconsistent outputs
        _print_and_abort(
            comm,
            f"parameterrun: rank {rank} of {size} on {hostname} failed. Aborting the MPI job.\n"
            f"{traceback.format_exc()}",
        )
        raise
    _log(f"Rank {rank} finished ({n_computed} points)", verbose, hostname=hostname)

    error = comm.bcast(None, root=0)
    if error is not None:
        raise RuntimeError(error)
    comm.Free()
    return [], 0
