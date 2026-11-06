"""MPI backend, static schedule: each rank derives its share of the points from a seeded pseudo-random permutation."""

import sys
from collections.abc import Callable, Iterator
from math import prod
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ._errors import _abort_on_error, _all_points_failed_error, _evaluate_point, _FailedPoint
from ._inputs import _get_iteration, _params_at, _product_element_from_index
from ._progress import _pbar_options
from ._pseudo_random import local_values
from ._store import Channel, ChannelSpec, _channel_spec, _ResultStore, _rows_per_block
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
    store (no pickling, no 2 GB limit, one copy on rank 0); other outputs are gathered as pickled lists.
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

    # Lazy import: _mpi_dynamic imports _shared_seed from this module
    from ._mpi_dynamic import _ChunkBuffer

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
    n_outputs = len(specs)
    if all(spec[0] == "array" for spec in specs):
        buffer.values = originals = []  # Not needed: free them before the transfer
    _log("Gathering results", verbose and rank == 0, hostname=hostname)

    all_indices = comm.gather(local_indices, root=0)

    if rank != 0:
        for c, spec in enumerate(specs):
            if spec[0] == "array" and len(local_indices):
                _send_rows(comm, _TAG_CHANNEL + c, np.ascontiguousarray(local_channels[c], dtype=np.dtype(spec[2])))
        del local_channels
        for c, spec in enumerate(specs):
            if spec[0] == "list":
                comm.gather([outputs[c] for outputs in originals], root=0)
        comm.Free()
        return [], n_outputs

    assert summaries is not None and all_indices is not None  # For type checker
    store = _ResultStore(n_total, n_values, result_file)
    store.allocate(specs)
    for item in summaries:
        if item["first_failure"] is not None:
            store.add_failed_indices(item["failed_indices"], item["first_failure"][1])

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
            gathered = comm.gather([], root=0)  # Root's own list was written above
            assert gathered is not None  # For type checker
            for source in range(1, size):
                store.write_channel(c, np.asarray(all_indices[source], dtype=np.intp), gathered[source])
    for source_indices in all_indices:
        store.mark_done(source_indices)

    _log("Results gathered", verbose, hostname=hostname)
    comm.Free()
    return store.finalize(param_names, param_values)


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
