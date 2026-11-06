"""MPI backend, static schedule: each rank derives its share of the points from a seeded pseudo-random permutation."""

import contextlib
import sys
from collections.abc import Callable, Iterator
from math import prod
from time import time
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from . import _store
from ._checkpoint import _SaveSignals
from ._errors import _abort_on_error, _all_points_failed_error, _evaluate_point, _FailedPoint
from ._inputs import _get_iteration, _params_at, _product_element_from_index
from ._mpi_transfer import _finish_store, _recv_payload, _send_payload
from ._progress import _pbar_options
from ._pseudo_random import local_values
from ._store import Channel, ChannelSpec, Payload, _channel_spec, _ChunkBuffer, _ResultStore, _rows_per_block
from ._utils import _log

# Tags of the result blocks sent to rank 0, one per output channel: _TAG_CHANNEL + c (tag 0 is the done signal)
_TAG_CHANNEL = 100


def _shared_seed(comm: Any, seed: int | None) -> int:  # pragma: no cover
    """
    Seed of the permutation of the points, identical on every rank: root's seed wins, a random one if it is None.
    """
    if comm.Get_rank() == 0 and seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    shared = comm.bcast(seed, root=0)
    assert isinstance(shared, int)  # For type checker
    return shared


def _parameterrun_mpi(
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
    result_file: Any = None,
) -> tuple[list[Channel], int]:  # pragma: no cover
    """
    Rank 0 returns the channels in flat (C-order) index order, from a ``_ResultStore``: numeric outputs of a fixed
    shape as one array of shape ``(n_total, *output_shape)`` (in ``result_file`` if given), anything else as a flat
    list. The other ranks return ``([], n_outputs)``.

    Each rank stacks its results per channel. Rank 0 gathers a short summary of every rank (``_plan_gather``) and
    decides per channel: arrays travel as raw bytes in blocks of at most ``_MAX_MESSAGE_BYTES`` straight into the
    store (no pickling, no 2 GB limit, one copy on rank 0); other outputs use bounded serialized transfers.
    """
    # fun_kwargs is an explicit dict (not **kwargs) so user kwargs such as `seed` cannot collide with our parameters
    fun_kwargs = fun_kwargs or {}

    from mpi4py import MPI

    # A private communicator per call: messages of a call that failed half-way can never reach a later call
    comm = MPI.COMM_WORLD.Dup()
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_values = [len(group[0]) for group in param_values]
    indices = [list(range(n_value)) for n_value in n_values]
    n_total = prod(n_values)  # Compute the total number of iterations

    seed = _shared_seed(comm, seed)
    count = n_total // size + (1 if rank < (n_total % size) else 0)

    # Use a generator to avoid storing all groups in memory and all indices
    def groups_compute_generator() -> Iterator[tuple[int, list[int]]]:
        for idx in local_values(n_total, size, rank, seed):
            yield int(idx), _product_element_from_index(indices, int(idx))

    groups_compute = groups_compute_generator()

    # Output goes through the MPI launcher, which forwards it line by line
    pbar_options = _pbar_options(pbar_bool and rank == 0, sys.stdout, forwarded=True)
    pbar = tqdm(groups_compute, desc=desc, leave=pbar_leave, total=count, **pbar_options)

    buffer = _ChunkBuffer()  # This rank's results, split into output channels
    local_error: str | None = None  # Inconsistent outputs on this rank: reported to every rank after the loop

    for orig_idx, index in pbar:
        params = _get_iteration(index, param_names, param_values)
        try:
            value = _evaluate_point(fun, params, fun_kwargs, on_error)
        except Exception as err:  # Only with on_error="raise"
            if size == 1:
                raise
            _abort_on_error(comm, err)
        try:
            buffer.add(orig_idx, value, params)
        except ValueError as err:
            local_error = str(err)
            break
        except Exception as err:
            if size > 1:
                _abort_on_error(comm, err)
            raise

    hostname = MPI.Get_processor_name()
    _log(f"Rank {rank} finished", verbose, hostname=hostname)

    # Notify root when each worker is done, and show a progress bar
    if rank == 0:
        pbar = tqdm(total=size - 1, desc="Waiting for workers", leave=pbar_leave, **pbar_options)

        for _ in range(size - 1):
            # Blocking receive: rank 0 sleeps inside MPI instead of spinning on Iprobe
            comm.recv(source=MPI.ANY_SOURCE, tag=0)
            pbar.update(1)

        pbar.close()
    else:
        comm.send(True, dest=0, tag=0)  # Notify root that the work is done

    local_indices, local_channels, failures = buffer.payload()
    originals = buffer.values  # As returned by fun: what list channels send, so values keep their Python types
    failures.sort(key=lambda item: item[0])
    summary = {
        "n_outputs": buffer.n_outputs,
        "n_points": len(local_indices),
        "specs": [_channel_spec(channel) for channel in local_channels],
        "error": local_error,
        "failed_indices": np.asarray([index for index, _ in failures], dtype=np.int64),
        "first_failure": failures[0] if failures else None,
    }
    del failures
    summaries = comm.gather(summary, root=0)
    plan = _plan_gather(summaries, n_total, param_names, param_values) if summaries is not None else None
    plan = comm.bcast(plan, root=0)
    if plan[0] == "error":  # Every rank knows it, so every rank raises
        raise (RuntimeError if plan[1] == "RuntimeError" else ValueError)(plan[2])

    specs: list[ChannelSpec] = plan[1]
    if all(spec[0] == "array" for spec in specs):
        buffer.values = originals = []  # Not needed: free them before the transfer
    _log("Gathering results", verbose and rank == 0, hostname=hostname)

    store = None
    allocation_error = None
    if rank == 0:
        try:
            store = _ResultStore(n_total, n_values, result_file)
            store.allocate(specs)
            assert summaries is not None
            for item in summaries:
                if item["first_failure"] is not None:
                    store.add_failed_indices(item["failed_indices"], item["first_failure"][1])
        except Exception as err:
            allocation_error = err
    message = comm.bcast(str(allocation_error) if allocation_error is not None else None, root=0)
    if message is not None:
        comm.Free()
        if allocation_error is not None:
            raise allocation_error
        raise RuntimeError(f"parameterrun result allocation failed on rank 0: {message}")

    all_indices = comm.gather(local_indices, root=0)
    try:
        if rank != 0:
            for c, spec in enumerate(specs):
                if spec[0] == "array" and len(local_indices):
                    _send_rows(comm, _TAG_CHANNEL + c, np.ascontiguousarray(local_channels[c], dtype=np.dtype(spec[2])))
            del local_channels
            for c, spec in enumerate(specs):
                if spec[0] == "list":
                    _send_payload(comm, [outputs[c] for outputs in originals], dest=0, tag=_TAG_CHANNEL + c)
        else:
            assert store is not None and all_indices is not None
            # Root's own points, then each rank's array channels block by block, then the list channels
            for c, spec in enumerate(specs if len(local_indices) else []):  # Root may have no successful point
                data = local_channels[c] if spec[0] == "array" else [outputs[c] for outputs in originals]
                store.write_channel(c, local_indices.astype(np.intp), data)
            del local_channels, originals
            for source in range(1, size):
                source_indices = np.asarray(all_indices[source], dtype=np.intp)
                for c, spec in enumerate(specs):
                    if spec[0] == "array":
                        for start, block in _recv_rows(comm, source, _TAG_CHANNEL + c, len(source_indices), spec):
                            store.write_channel(c, source_indices[start : start + len(block)], block)
            for c, spec in enumerate(specs):
                if spec[0] == "list":
                    for source in range(1, size):
                        nbytes = comm.recv(source=source, tag=_TAG_CHANNEL + c)
                        data = _recv_payload(comm, nbytes, source, _TAG_CHANNEL + c)
                        store.write_channel(c, np.asarray(all_indices[source], dtype=np.intp), data)
            for source_indices in all_indices:
                store.mark_done(source_indices)
    except Exception as err:
        # Some ranks may already be inside a blocking transfer; a collective is no longer safe.
        if size > 1:
            _abort_on_error(comm, err)
        raise

    _log("Results gathered", verbose, hostname=hostname)
    try:
        return _finish_store(comm, store, param_names, param_values)
    finally:
        comm.Free()


def _plan_gather(
    summaries: list[dict[str, Any]],
    n_total: int,
    param_names: list[list[str]],
    param_values: list[list[list[Any]]],
) -> tuple[Any, ...]:
    """
    Decide on rank 0, from every rank's summary, how each output channel is gathered: ``("ok", specs)`` with one
    ``("array", output_shape, dtype)`` or ``("list",)`` per channel, or ``("error", exception name, message)``.

    A channel travels as an array only if every rank that computed points could stack it, with the same output shape;
    its dtype is promoted over the ranks like ``np.array`` would. Otherwise it is gathered as lists.
    """
    errors = [item["error"] for item in summaries if item["error"]]
    if errors:
        return ("error", "ValueError", errors[0])

    counts = sorted({item["n_outputs"] for item in summaries if item["n_outputs"] is not None})
    if not counts:  # Every point failed
        index, failure = min(
            (item["first_failure"] for item in summaries if item["first_failure"] is not None), key=lambda t: t[0]
        )
        assert isinstance(failure, _FailedPoint)  # For type checker
        error = _all_points_failed_error(n_total, failure, _params_at(int(index), param_names, param_values))
        return ("error", "RuntimeError", str(error))
    if len(counts) > 1:
        return (
            "error",
            "ValueError",
            f"fun returned {counts[0]} outputs for some points but {counts[-1]} for others. Return the same number "
            f"of outputs (a tuple of the same length) for every point.",
        )

    specs: list[ChannelSpec] = []
    for c in range(counts[0]):
        rank_specs = [item["specs"][c] for item in summaries if item["n_points"] > 0]
        if all(spec[0] == "array" for spec in rank_specs) and len({tuple(spec[1]) for spec in rank_specs}) == 1:
            dtype = np.result_type(*[np.dtype(spec[2]) for spec in rank_specs])
            specs.append(("array", tuple(rank_specs[0][1]), dtype.str))
        else:
            specs.append(("list",))
    return ("ok", specs)


def _send_rows(comm: Any, tag: int, array: np.ndarray) -> None:  # pragma: no cover
    """Send a contiguous array to rank 0 as raw bytes, in blocks of whole rows (see ``_recv_rows``)."""
    from mpi4py import MPI

    step = _rows_per_block(array[0].nbytes if len(array) else 0)
    for start in range(0, len(array), step):
        block = np.ascontiguousarray(array[start : start + step])
        comm.Send([block.reshape(-1).view(np.uint8), MPI.BYTE], dest=0, tag=tag)


def _recv_rows(
    comm: Any, source: int, tag: int, n_rows: int, spec: ChannelSpec
) -> Iterator[tuple[int, np.ndarray]]:  # pragma: no cover
    """Receive what ``_send_rows`` sent, as (first row, block) pairs. Only one block is in memory at a time."""
    from mpi4py import MPI

    output_shape, dtype = tuple(spec[1]), np.dtype(spec[2])
    row_nbytes = int(np.prod(output_shape)) * dtype.itemsize
    step = _rows_per_block(row_nbytes)
    for start in range(0, n_rows, step):
        block = np.empty((min(step, n_rows - start), *output_shape), dtype=dtype)
        comm.Recv([block.reshape(-1).view(np.uint8), MPI.BYTE], source=source, tag=tag)
        yield start, block


# Messages of the checkpointed static schedule (rounds), worker -> rank 0
_TAG_ROUND_PART = 21  # A Payload of this round
_TAG_ROUND_END = 22  # (this rank has no work left, error message or None): the last message of the round


def _share_remaining(comm: Any, checkpoint: Any, n_total: int) -> np.ndarray | None:  # pragma: no cover
    """
    Flat indices still to compute, identical on every rank: None if nothing was resumed (all points), otherwise
    rank 0's done mask is broadcast as bits. Collective: every rank calls it, ``checkpoint`` is only given on rank 0.
    """
    packed = None
    if comm.Get_rank() == 0 and checkpoint is not None and checkpoint.n_resumed:
        packed = np.packbits(checkpoint.store.done)
    packed = comm.bcast(packed, root=0)
    if packed is None:
        return None
    return np.flatnonzero(~np.unpackbits(packed, count=n_total).astype(bool))


def _round_is_over(n_round: int, round_start: float, every: int | None, interval: float | None) -> bool:
    return (every is not None and n_round >= every) or (interval is not None and time() - round_start >= interval)


def _parameterrun_mpi_rounds(
    fun: Callable[..., Any],
    param_names: list[list[str]],
    param_values: list[list[list[Any]]],
    desc: str | None,
    checkpoint: Any,
    checkpoint_every: int | None,
    checkpoint_interval: float | None,
    pbar_bool: bool = True,
    verbose: bool | None = False,
    pbar_leave: bool | None = True,
    seed: int | None = None,
    fun_kwargs: dict[str, Any] | None = None,
    on_error: str = "raise",
) -> tuple[list[Channel], int]:  # pragma: no cover
    """
    Static schedule with checkpoints: the same pseudo-random split of the points as ``_parameterrun_mpi``, computed in
    synchronised rounds. A round ends on every rank after ``checkpoint_every / size`` points, or ``checkpoint_interval``
    seconds (each rank's own clock, from the common start of the round); then every rank sends its results of the
    round to rank 0, which saves them, and rank 0 tells every rank whether to go on. Fast ranks wait for the slowest at
    the end of each round: the price of a static schedule with snapshots.

    ``checkpoint`` (a ``_Checkpoint``) is only given on rank 0. When resuming, only the points it does not hold are
    split among the ranks, so the number of ranks may differ from the interrupted run.
    """
    fun_kwargs = fun_kwargs or {}

    from mpi4py import MPI

    # A private communicator per call: messages of a call that failed half-way can never reach a later call
    comm = MPI.COMM_WORLD.Dup()
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = MPI.Get_processor_name()

    n_values = [len(group[0]) for group in param_values]
    axes = [list(range(n_value)) for n_value in n_values]
    n_total = prod(n_values)

    seed = _shared_seed(comm, seed)
    remaining = _share_remaining(comm, checkpoint, n_total)
    n_left = n_total if remaining is None else len(remaining)
    count = n_left // size + (1 if rank < (n_left % size) else 0)
    if rank == 0 and remaining is not None:
        _log(f"Resuming from checkpoint: {n_total - n_left} of {n_total} points done", verbose, hostname=hostname)

    def points() -> Iterator[int]:
        for position in local_values(n_left, size, rank, seed):
            yield int(position) if remaining is None else int(remaining[position])

    todo = points()
    every = None if checkpoint_every is None else -(-checkpoint_every // size)  # Points per rank and round
    pbar_options = _pbar_options(pbar_bool and rank == 0, sys.stdout, forwarded=True)
    pbar = tqdm(total=count, desc=desc, leave=pbar_leave, **pbar_options)

    buffer = _ChunkBuffer()
    exhausted = False
    local_error: str | None = None
    result: tuple[list[Channel], int] = ([], 0)

    # SIGUSR1: save at the end of this round; SIGTERM: save, then every rank terminates
    signals = checkpoint.signals if checkpoint is not None else _SaveSignals()
    with signals:
        while True:
            pending: list[Payload] = []  # This rank's results of the round
            n_round = 0
            round_start = time()
            while not (
                exhausted
                or local_error
                or signals.save  # SIGUSR1 or SIGTERM: end the round now, so rank 0 saves
                or _round_is_over(n_round, round_start, every, checkpoint_interval)
            ):
                index = next(todo, None)
                if index is None:
                    exhausted = True
                    break
                params = _get_iteration(_product_element_from_index(axes, index), param_names, param_values)
                try:
                    value = _evaluate_point(fun, params, fun_kwargs, on_error)
                except Exception as err:  # Only with on_error="raise"
                    if checkpoint is not None:  # Rank 0 can at least keep its own points of this round
                        with contextlib.suppress(Exception):
                            for payload in [*pending, buffer.payload()]:
                                checkpoint.add(payload)
                        checkpoint.commit_quietly()
                    if size == 1:
                        raise
                    _abort_on_error(comm, err)
                try:
                    buffer.add(index, value, params)
                except ValueError as err:
                    local_error = str(err)
                    break
                except Exception as err:
                    if size > 1:
                        _abort_on_error(comm, err)
                    raise
                n_round += 1
                pbar.update(1)
                if buffer.nbytes >= _store._MAX_MESSAGE_BYTES:  # Keep each message far below the 2 GB pickle limit
                    pending.append(buffer.payload())
                    buffer.clear()
            if len(buffer):
                pending.append(buffer.payload())
                buffer.clear()
            state = (exhausted or local_error is not None, local_error, signals.stop)
            signals.save = False

            decision: tuple[Any, ...]
            root_error: Exception | None = None
            if rank != 0:
                try:
                    for payload in pending:
                        _send_payload(comm, payload, dest=0, tag=_TAG_ROUND_PART)
                    comm.send(state, dest=0, tag=_TAG_ROUND_END)
                except Exception as err:
                    _abort_on_error(comm, err)
                    raise
                del pending
                decision = comm.bcast(None, root=0)
            else:
                try:
                    decision = _collect_round(comm, checkpoint, pending, state)
                except Exception as err:  # A receive failed: senders may still be blocked mid-transfer
                    if size > 1:
                        _abort_on_error(comm, err)
                    raise
                del pending
                try:
                    if decision[0] != "error" and checkpoint.commit():
                        _log(f"Checkpoint saved: {checkpoint.n_saved} of {n_total} points", verbose, hostname=hostname)
                    if decision[0] == "stop":
                        result = checkpoint.store.finalize(param_names, param_values)
                except Exception as err:
                    root_error = err
                    decision = ("error", type(err).__name__, str(err))
                comm.bcast(decision, root=0)

            if decision[0] == "error":  # Every rank knows it, so every rank raises
                pbar.close()
                comm.Free()
                if root_error is not None:
                    raise root_error
                raise (ValueError if decision[1] == "ValueError" else RuntimeError)(decision[2])
            if decision[0] == "terminate":  # SIGTERM on some rank, and the round is saved: every rank ends
                pbar.close()
                comm.Free()
                signals.terminate()
            if decision[0] == "stop":
                break

    pbar.close()
    _log(f"Rank {rank} finished", verbose, hostname=hostname)
    comm.Free()
    return result


def _collect_round(
    comm: Any, checkpoint: Any, own: list[Payload], own_state: tuple[bool, str | None, bool]
) -> tuple[Any, ...]:  # pragma: no cover
    """
    Rank 0: put every rank's results of the round into the checkpoint, and decide ``("continue",)``, ``("stop",)``
    (no rank has work left), ``("terminate",)`` (a rank got SIGTERM) or ``("error", exception name, message)``.
    Always receives every message of the round, even after an error, so nothing is left in flight.
    """
    from mpi4py import MPI

    status = MPI.Status()
    states = [own_state]
    store_error: Exception | None = None

    def absorb(payload: Payload) -> None:
        nonlocal store_error
        if store_error is None:
            try:
                checkpoint.add(payload)
            except Exception as err:  # Drain all senders even after a storage failure
                store_error = err

    for payload in own:
        absorb(payload)
    for source in range(1, comm.Get_size()):
        while True:
            message = comm.recv(source=source, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == _TAG_ROUND_END:
                states.append(message)
                break
            absorb(_recv_payload(comm, message, source, _TAG_ROUND_PART))

    if store_error is not None:
        checkpoint.commit_quietly()
        return ("error", type(store_error).__name__, str(store_error))
    errors = [error for _, error, _ in states if error]
    if errors:
        checkpoint.commit_quietly()  # The points stored before the error are consistent
        return ("error", "ValueError", errors[0])
    if all(finished for finished, _, _ in states):
        return ("stop",)
    return ("terminate",) if any(stop for _, _, stop in states) else ("continue",)
