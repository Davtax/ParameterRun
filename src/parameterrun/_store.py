"""
Result store on rank 0 of the MPI backends: one preallocated array per output channel, written by flat index.

Results arrive in blocks, in any order. Numeric outputs of a fixed shape go straight into a preallocated array, so
rank 0 holds one copy of the result instead of lists of per-point objects. Anything else (non-numeric or ragged
outputs) falls back to a flat list for that channel, and the usual post-processing in ``parameterrun`` then gives the
same result, dtypes and warnings as the joblib backend.

With a ``result_file``, array channels live in ``.npy`` files (``np.lib.format.open_memmap``) of the grid shape, so the
size of the result is not limited by rank 0's memory. ``finalize`` reopens them copy-on-write: the returned arrays read
from the file, and changing them in memory never changes the file.

``write`` is append-only by flat index and ``done`` marks the completed points, which is what checkpointing will build
on.
"""

import contextlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._errors import _all_points_failed_error, _FailedPoint, _nan_like, _warn_failed_points
from ._inputs import _params_at

_NUMERIC_KINDS = "biufc"

# Upper bound for one MPI message and for one block copied between files. Read at call time (tests lower it)
_MAX_MESSAGE_BYTES = 64 * 2**20

# One output channel: a preallocated array of shape (n_total, *output_shape), or a flat list of per-point values
Channel = np.ndarray | list[Any]

# How rank 0 allocates a channel: ("array", output_shape, dtype string), or ("list",)
ChannelSpec = tuple[Any, ...]


def _stack_channel(values: list[Any]) -> Channel:
    """
    Stack the values of one output channel into a contiguous array when they are numeric NumPy arrays or scalars with
    one shape (cheap to send and to store). Otherwise return the list unchanged. Python lists and tuples are not
    stacked, so that ``reshape=False`` still returns them as the function did.
    """
    if any(isinstance(value, (list, tuple)) for value in values):
        return values
    try:
        array = np.asarray(values)
    except (ValueError, TypeError, OverflowError):  # Ragged shapes, or integers out of range
        return values
    if array.dtype.kind in _NUMERIC_KINDS and array.ndim >= 1 and len(array) == len(values):
        return array
    return values


def _channel_spec(data: Channel) -> ChannelSpec:
    if isinstance(data, np.ndarray) and data.dtype.kind in _NUMERIC_KINDS:
        return ("array", tuple(data.shape[1:]), data.dtype.str)
    return ("list",)


def _channel_paths(result_file: str | os.PathLike[str], n_outputs: int) -> list[Path]:
    """``result.npy`` for one output channel; ``result_0.npy``, ``result_1.npy``, ... for several."""
    path = Path(result_file)
    if n_outputs == 1:
        return [path]
    return [path.with_name(f"{path.stem}_{c}{path.suffix}") for c in range(n_outputs)]


def _rows_per_block(row_nbytes: int) -> int:
    return max(1, _MAX_MESSAGE_BYTES // max(row_nbytes, 1))


class _ResultStore:
    def __init__(
        self,
        n_total: int,
        grid_shape: Sequence[int] | None = None,
        result_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self.n_total = n_total
        self.grid_shape = tuple(grid_shape) if grid_shape is not None else (n_total,)
        self.result_file = result_file
        self.done = np.zeros(n_total, dtype=bool)
        self.failures: dict[int, _FailedPoint] = {}
        self._channels: list[Channel | None] | None = None  # Allocated on the first successful result
        self._paths: list[Path | None] = []  # File of each array channel that lives in a file

    @property
    def n_outputs(self) -> int:
        return 0 if self._channels is None else len(self._channels)

    # ------------------------------------------------------------------ writing

    def allocate(self, specs: Sequence[ChannelSpec]) -> None:
        """Allocate the channels explicitly (the static backend knows them before any data arrives)."""
        paths = _channel_paths(self.result_file, len(specs)) if self.result_file is not None else []
        self._channels = []
        self._paths = []
        for c, spec in enumerate(specs):
            if spec[0] == "array":
                path = paths[c] if paths else None
                self._channels.append(self._new_array(tuple(spec[1]), np.dtype(spec[2]), path))
                self._paths.append(path if isinstance(self._channels[c], np.memmap) else None)
            else:
                self._channels.append([None] * self.n_total)
                self._paths.append(None)

    def write(self, indices: Sequence[int] | np.ndarray, channels: Sequence[Channel]) -> None:
        """Store the successful points ``indices``; ``channels[c][j]`` is output ``c`` of point ``indices[j]``."""
        flat = np.asarray(indices, dtype=np.intp)
        if len(flat) == 0:
            return

        if self._channels is None:
            self.allocate([_channel_spec(data) for data in channels])
        elif len(channels) != len(self._channels):
            raise ValueError(
                f"fun returned {len(channels)} outputs for some points but {len(self._channels)} for others. Return "
                f"the same number of outputs (a tuple of the same length) for every point."
            )

        for c, data in enumerate(channels):
            self.write_channel(c, flat, data)
        self.mark_done(flat)

    def write_channel(self, c: int, indices: np.ndarray, data: Channel) -> None:
        """Write one channel only; call ``mark_done`` once every channel of these points is written."""
        assert self._channels is not None  # For type checker
        current = self._channels[c]
        if isinstance(current, np.ndarray):
            fits = (
                isinstance(data, np.ndarray)
                and data.dtype.kind in _NUMERIC_KINDS
                and data.shape[1:] == current.shape[1:]
            )
            if fits:
                assert isinstance(data, np.ndarray)  # For type checker
                # Promote like np.array(list) would (e.g. int then float -> float64), never truncate
                dtype = np.result_type(current.dtype, data.dtype)
                del current  # The file of a memory-mapped channel can only be replaced once nothing maps it
                if dtype != self._array(c).dtype:
                    self._promote(c, dtype)
                self._array(c)[indices] = data
                return
            del current
            self._to_list(c)  # Different shape or a non-numeric value: fall back to a list

        items = self._channels[c]
        assert isinstance(items, list)  # For type checker
        for index, value in zip(indices.tolist(), data, strict=True):
            items[index] = value

    def mark_done(self, indices: Sequence[int] | np.ndarray) -> None:
        self.done[np.asarray(indices, dtype=np.intp)] = True

    def add_failures(self, failures: Sequence[tuple[int, _FailedPoint]]) -> None:
        for index, failure in failures:
            self.failures[int(index)] = failure
            self.done[int(index)] = True

    def add_failed_indices(self, indices: Sequence[int] | np.ndarray, failure: _FailedPoint) -> None:
        """Failed points that share one representative failure (only the first one is ever reported in full)."""
        for index in np.asarray(indices, dtype=np.intp).tolist():
            self.failures[index] = failure
            self.done[index] = True

    # ------------------------------------------------------------------ finishing

    def finalize(self, param_names: list[list[str]], param_values: list[list[list[Any]]]) -> tuple[list[Channel], int]:
        """
        Fill the failed points with NaN (and warn once), and return the channels in flat (C-order) index order.
        Raises RuntimeError if every point failed.
        """
        if not self.done.all():
            missing = int(np.flatnonzero(~self.done)[0])
            raise RuntimeError(f"Internal error: no result for {int((~self.done).sum())} points (first: {missing})")

        if self._channels is None:
            first = min(self.failures)
            raise _all_points_failed_error(
                self.n_total, self.failures[first], _params_at(first, param_names, param_values)
            )

        if self.failures:
            failed = np.fromiter(sorted(self.failures), dtype=np.intp, count=len(self.failures))
            for c in range(len(self._channels)):
                self._fill_nan(c, failed)
            first = int(failed[0])
            _warn_failed_points(len(failed), self.n_total, first, self.failures[first], param_names, param_values)

        for c, path in enumerate(self._paths):
            if path is not None:
                self._reopen_copy_on_write(c, path)

        channels = [channel for channel in self._channels if channel is not None]
        return channels, len(channels)

    # ------------------------------------------------------------------ channels

    def _array(self, c: int) -> np.ndarray:
        assert self._channels is not None  # For type checker
        array = self._channels[c]
        assert isinstance(array, np.ndarray)  # For type checker
        return array

    def _new_array(self, output_shape: tuple[int, ...], dtype: np.dtype[Any], path: Path | None) -> np.ndarray:
        flat_shape = (self.n_total, *output_shape)
        nbytes = int(np.prod(flat_shape)) * dtype.itemsize
        if path is None or nbytes == 0:  # An empty file cannot be mapped: keep it in memory, saved at the end
            return np.zeros(flat_shape, dtype=dtype)
        mapped = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(*self.grid_shape, *output_shape))
        return mapped.reshape(flat_shape)

    def _promote(self, c: int, dtype: np.dtype[Any]) -> None:
        assert self._channels is not None  # For type checker
        path = self._paths[c]
        if path is None:
            self._channels[c] = self._array(c).astype(dtype)
            return

        # Copy into a new file block by block, close both mappings, then move the new file into place
        tmp = path.with_name(f"{path.name}.tmp")
        old = self._array(c)
        flat_shape = old.shape
        new = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=(*self.grid_shape, *old.shape[1:]))
        new_flat = new.reshape(old.shape)
        step = _rows_per_block(int(np.prod(old.shape[1:])) * max(old.itemsize, dtype.itemsize))
        for start in range(0, self.n_total, step):
            new_flat[start : start + step] = old[start : start + step]
        new.flush()
        self._channels[c] = None
        del old, new, new_flat
        os.replace(tmp, path)
        self._channels[c] = np.lib.format.open_memmap(path, mode="r+").reshape(flat_shape)

    def _to_list(self, c: int) -> None:
        assert self._channels is not None  # For type checker
        array = self._array(c)
        items: list[Any] = [None] * self.n_total
        for index in np.flatnonzero(self.done).tolist():
            if index not in self.failures:
                # Copy rows: views would keep the file mapped
                items[index] = array[index].copy() if array.ndim > 1 else array[index]
        self._channels[c] = items
        del array
        path = self._paths[c]
        if path is not None:
            self._paths[c] = None
            with contextlib.suppress(OSError):  # The file is incomplete now; parameterrun reports it was not written
                path.unlink()

    def _fill_nan(self, c: int, failed: np.ndarray) -> None:
        assert self._channels is not None  # For type checker
        channel = self._channels[c]
        if isinstance(channel, np.ndarray):
            kind = channel.dtype.kind
            del channel
            if kind not in "fc":  # Integer and boolean outputs become float64 to hold NaN
                self._promote(c, np.dtype(np.float64))
            self._array(c)[failed] = np.nan
            return

        assert isinstance(channel, list)  # For type checker
        template_index = next(i for i in range(self.n_total) if i not in self.failures)
        fill = _nan_like(channel[template_index])
        for index in failed.tolist():
            channel[index] = fill

    def _reopen_copy_on_write(self, c: int, path: Path) -> None:
        assert self._channels is not None  # For type checker
        array = self._array(c)
        flat_shape = array.shape
        if isinstance(array, np.memmap):
            array.flush()
        self._channels[c] = None
        del array
        self._channels[c] = np.load(path, mmap_mode="c").reshape(flat_shape)
