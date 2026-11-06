"""
MPI backend, dynamic schedule: rank 0 is a manager that hands out chunks of points on demand; the other ranks compute.

Fast ranks simply ask for more work, so differences in speed between nodes and cores, and uneven task costs, no longer
leave the whole job waiting for the slowest rank.

Protocol (small control messages and serialized payloads transferred in bounded blocks):

* worker -> manager, ``_TAG_RESULT``: results of the last chunk (``None`` at start-up), and a request for more work.
* worker -> manager, ``_TAG_PARTIAL``: part of a chunk's results, sent early when they exceed ``_MAX_MESSAGE_BYTES``.
* worker -> manager, ``_TAG_ERROR``: text to print; the manager prints it once and aborts the job.
* worker -> manager, ``_TAG_SIGNAL`` (checkpointed runs): the worker got SIGUSR1 (``False``: save now) or SIGTERM
  (``True``: save, then end the job).
* manager -> worker, ``_TAG_WORK``: ``(start, stop)``, positions in the seeded permutation of the flat indices. Workers
  map positions to flat indices themselves, so the order is never materialized.
* manager -> worker, ``_TAG_STOP``: no more work.

At the end the manager broadcasts ``None``, or the message of the error that every rank must raise (all points failed).
"""

import contextlib
import sys
import traceback
from collections.abc import Callable
from math import prod
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from . import _store
from ._checkpoint import _SaveSignals
from ._errors import _abort_header, _evaluate_point, _print_and_abort
from ._inputs import _get_iteration, _product_element_from_index
from ._mpi_static import _share_remaining, _shared_seed
from ._mpi_transfer import _finish_store, _recv_payload, _send_payload
from ._progress import _pbar_options
from ._pseudo_random import _permute_in_range
from ._store import Channel, _ChunkBuffer, _ResultStore
from ._utils import _log

_TAG_RESULT = 11
_TAG_PARTIAL = 12
_TAG_ERROR = 13
_TAG_WORK = 14
_TAG_STOP = 15
_TAG_SIGNAL = 16


def _guided_chunk_size(remaining: int, n_workers: int, chunk_size: int | None = None) -> int:
    """
    Number of points in the next chunk. Guided by default: a quarter of each worker's share of what is left, so chunks
    are large at the start (few messages) and shrink to one point at the end (no tail).
    """
    if chunk_size is not None:
        return max(1, min(chunk_size, remaining))
    return max(1, remaining // (4 * n_workers))


def _manager(
    comm: Any,
    n_total: int,
    store: _ResultStore,
    pbar: Any,
    chunk_size: int | None,
    checkpoint: Any = None,
    log: Callable[[str], None] | None = None,
) -> None:  # pragma: no cover
    """
    Hand out the positions ``0 ... n_total`` (of the points still to compute) and store the results. With a
    ``checkpoint`` (whose store is ``store``), the results go through it and are saved whenever it is due.
    """
    from mpi4py import MPI

    n_workers = comm.Get_size() - 1
    status = MPI.Status()
    next_position = 0
    active = n_workers

    while active:
        payload = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source, tag = status.Get_source(), status.Get_tag()

        if tag == _TAG_ERROR:
            if checkpoint is not None:
                checkpoint.commit_quietly()
            _print_and_abort(comm, payload)

        if tag == _TAG_SIGNAL:  # Forwarded by a worker: save as if the manager got the signal
            if checkpoint is not None:
                checkpoint.signals.save = True
                checkpoint.signals.stop = checkpoint.signals.stop or bool(payload)
            payload = None

        if payload is not None:
            payload = _recv_payload(comm, payload, source, tag)
            indices, channels, failures = payload
            if checkpoint is not None:
                checkpoint.add(payload)
            else:
                store.write(indices, channels)
                store.add_failures(failures)
            pbar.update(len(indices) + len(failures))

        if checkpoint is not None and checkpoint.due():
            stop = checkpoint.signals.stop
            if checkpoint.commit() and log is not None:
                log(f"Checkpoint saved: {checkpoint.n_saved} of {store.n_total} points")
            if stop:  # SIGTERM: the snapshot is saved, end the whole job
                _print_and_abort(comm, "parameterrun: terminated by SIGTERM after saving the checkpoint.\n")

        if tag in (_TAG_PARTIAL, _TAG_SIGNAL):  # The worker is still on its chunk
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
    remaining: np.ndarray | None = None,
    signals: _SaveSignals | None = None,
) -> int:  # pragma: no cover
    """Compute the chunks the manager hands out. Positions index ``remaining`` (the points left) when resuming."""
    from mpi4py import MPI

    n_values = [len(group[0]) for group in param_values]
    indices = [list(range(n_value)) for n_value in n_values]
    n_total = prod(n_values) if remaining is None else len(remaining)

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
            if remaining is not None:
                index = int(remaining[index])
            params = _get_iteration(_product_element_from_index(indices, index), param_names, param_values)
            try:
                value = _evaluate_point(fun, params, fun_kwargs, on_error)
            except Exception as err:  # Only with on_error="raise"; the note on err holds the parameters
                header = _abort_header(comm.Get_rank(), comm.Get_size(), MPI.Get_processor_name())
                comm.send(header + "".join(traceback.format_exception(err)), dest=0, tag=_TAG_ERROR)
                comm.recv(source=0, tag=_TAG_STOP)  # Never arrives: the manager aborts the job
                raise  # Unreachable unless the abort fails

            buffer.add(index, value, params)
            if signals is not None and signals.save:  # Tell the manager, which saves (and ends the job on SIGTERM)
                comm.send(signals.stop, dest=0, tag=_TAG_SIGNAL)
                signals.save = False
            # Send early, so messages stay far below the 2 GB limit of pickled MPI messages even for large chunks
            if buffer.nbytes >= _store._MAX_MESSAGE_BYTES:
                n_computed += len(buffer)
                _send_payload(comm, buffer.payload(), dest=0, tag=_TAG_PARTIAL)
                buffer.clear()

        n_computed += len(buffer)
        _send_payload(comm, buffer.payload(), dest=0, tag=_TAG_RESULT)
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
    checkpoint: Any = None,
    checkpointed: bool = False,
) -> tuple[list[Channel], int]:  # pragma: no cover
    """
    Rank 0 returns the channels in flat (C-order) index order: numeric outputs of a fixed shape as one array of shape
    ``(n_total, *output_shape)``, anything else as a flat list. The other ranks return ``([], 0)``. Needs >= 2 ranks.

    With a ``checkpoint`` (a ``_Checkpoint``, given on rank 0 only), the manager saves the results whenever it is due,
    without stopping the workers, and only the points it does not hold yet are handed out. ``checkpointed`` (every
    rank) installs the signal handlers of ``_SaveSignals``: workers forward SIGUSR1 and SIGTERM to the manager.
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
    remaining = _share_remaining(comm, checkpoint, n_total)  # None unless resuming
    n_left = n_total if remaining is None else len(remaining)

    if rank == 0:
        _log(
            f"Dynamic schedule: 1 manager + {size - 1} workers, "
            f"{'guided chunks' if chunk_size is None else f'chunks of {chunk_size} points'}",
            verbose,
            hostname=hostname,
        )
        if checkpoint is not None:
            store = checkpoint.store
            if remaining is not None:
                _log(
                    f"Resuming from checkpoint: {n_total - n_left} of {n_total} points done", verbose, hostname=hostname
                )
        else:
            store = _ResultStore(n_total, [len(group[0]) for group in param_values], result_file)
        # Output goes through the MPI launcher, which forwards it line by line
        pbar_options = _pbar_options(pbar_bool, sys.stdout, forwarded=True)
        pbar = tqdm(total=n_total, initial=n_total - n_left, desc=desc, leave=pbar_leave, **pbar_options)

        def log(message: str) -> None:
            _log(message, verbose, hostname=hostname)

        try:
            with checkpoint.signals if checkpoint is not None else contextlib.nullcontext():
                _manager(comm, n_left, store, pbar, chunk_size, checkpoint, log)
            if checkpoint is not None and checkpoint.commit():
                log(f"Checkpoint saved: {checkpoint.n_saved} of {n_total} points")
        except Exception as err:  # A bug or inconsistent outputs: don't leave the workers waiting
            if checkpoint is not None:
                checkpoint.commit_quietly()
            _print_and_abort(
                comm, f"parameterrun: the manager (rank 0) failed. Aborting the MPI job.\n{traceback.format_exc()}"
            )
            raise err
        except BaseException:  # KeyboardInterrupt, SystemExit: keep what arrived
            if checkpoint is not None:
                checkpoint.commit_quietly()
            raise
        pbar.close()

        try:
            return _finish_store(comm, store, param_names, param_values)
        finally:
            comm.Free()

    try:
        with _SaveSignals() if checkpointed else contextlib.nullcontext() as signals:
            n_computed = _worker(comm, fun, param_names, param_values, fun_kwargs, on_error, seed, remaining, signals)
    except Exception:  # Not from fun (that is handled inside): a bug or inconsistent outputs
        _print_and_abort(
            comm,
            f"parameterrun: rank {rank} of {size} on {hostname} failed. Aborting the MPI job.\n"
            f"{traceback.format_exc()}",
        )
        raise
    _log(f"Rank {rank} finished ({n_computed} points)", verbose, hostname=hostname)

    try:
        return _finish_store(comm, None, param_names, param_values)
    finally:
        comm.Free()
