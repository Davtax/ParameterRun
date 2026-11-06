"""
Checkpoints: save the finished points of a long run to a directory, and resume from it after a crash or a walltime kill.

Layout of the checkpoint directory (append-only, every file written atomically: ``.tmp`` then ``os.replace``):

* ``manifest.json``: format version, run fingerprint, code fingerprint of ``fun``, and the list of committed chunks.
* ``chunk_000000.npz``, ``chunk_000001.npz``, ...: the ``Payload``s (flat indices, channels, failed points) of one
  save. Numeric outputs are plain arrays (loaded with ``allow_pickle=False``); only non-numeric outputs are pickled.
  Channels that live in a memory-mapped ``result_file`` are not copied: the file is flushed before the manifest is
  written, and reopened on resume. Only chunks listed in the manifest count; a chunk written just before a crash is
  ignored.

A snapshot records *which* flat indices are done, not a position in some order, so it does not depend on the backend,
the MPI schedule, the shuffle seed or the number of ranks: a run can be resumed under any of them. On resume, the
chunks are replayed into a fresh ``_ResultStore`` and only the points that are not done yet are computed. Failed points
(``on_error="nan"``) count as done, so a resumed run gives the same result as an uninterrupted one.

The run fingerprint (parameter names and values, ``fun_kwargs``) must match, otherwise resuming raises. The code of
``fun`` is fingerprinted too (bytecode, constants, defaults, closures, and helper functions of the same module that it
calls), but changes to other modules or libraries cannot be detected.
"""

import contextlib
import functools
import hashlib
import inspect
import json
import os
import pickle
import re
import signal
import sys
import threading
import types
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from time import monotonic
from typing import Any, BinaryIO, Literal

import joblib
import numpy as np

from ._errors import _FailedPoint
from ._store import _NUMERIC_KINDS, Payload, _channel_paths, _ResultStore

_FORMAT_VERSION = 1
_MANIFEST = "manifest.json"
_CHUNK_PATTERN = "chunk_*.npz"
_RESUME_MODES = ("auto", "force", "restart")

# Default seconds between saves when neither checkpoint_every nor checkpoint_interval is given
_DEFAULT_INTERVAL = 600.0

CheckpointResume = Literal["auto", "force", "restart"]


# ---------------------------------------------------------------------------------------------- fingerprints


def _digest(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8", "backslashreplace")).hexdigest()


def _value_digest(value: Any) -> str:
    """Hash of a value by content (``joblib.hash`` pickles it), or of its ``repr`` if it cannot be pickled."""
    try:
        return str(joblib.hash(value))
    except Exception:  # Unpicklable objects (locks, open files, ...)
        return "repr:" + _digest(value)


def _const_part(const: Any) -> Any:
    if isinstance(const, types.CodeType):
        return _code_object_part(const)
    if isinstance(const, frozenset):  # Iteration order depends on the string hash seed
        return ("frozenset", sorted(repr(item) for item in const))
    if isinstance(const, tuple):
        return tuple(_const_part(item) for item in const)
    return repr(const)


def _code_object_part(code: types.CodeType) -> tuple[Any, ...]:
    """Bytecode, constants and names, without line numbers: moving a function or adding comments changes nothing."""
    return (
        code.co_code,
        tuple(_const_part(const) for const in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_exceptiontable,
    )


def _global_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_names(const)
    return names


def _code_part(obj: Any, seen: set[int]) -> Any:
    if id(obj) in seen:  # Recursion between helpers
        return ("seen", getattr(obj, "__qualname__", type(obj).__qualname__))
    seen.add(id(obj))

    if isinstance(obj, functools.partial):
        return ("partial", _code_part(obj.func, seen), _value_digest(obj.args), _value_digest(obj.keywords))
    if inspect.ismethod(obj):
        return ("method", _code_part(obj.__func__, seen), _value_digest(getattr(obj.__self__, "__dict__", None)))
    if isinstance(obj, types.FunctionType):
        closure = []
        for cell in obj.__closure__ or ():
            try:
                content = cell.cell_contents
            except ValueError:  # Empty cell
                closure.append("empty")
                continue
            closure.append(_code_part(content, seen) if callable(content) else _value_digest(content))
        helpers = []
        for name in sorted(_global_names(obj.__code__)):
            helper = obj.__globals__.get(name)
            # Functions defined next to fun (the user's own helpers); library functions are not followed
            if isinstance(helper, types.FunctionType) and helper.__module__ == obj.__module__:
                helpers.append((name, _code_part(helper, seen)))
        return (
            "function",
            _code_object_part(obj.__code__),
            _value_digest(obj.__defaults__),
            _value_digest(obj.__kwdefaults__),
            tuple(closure),
            tuple(helpers),
        )
    if callable(obj) and not isinstance(obj, type) and hasattr(obj, "__dict__"):  # Instance with __call__
        return ("callable", _code_part(type(obj).__call__, seen), _value_digest(vars(obj)))
    # Builtins, ufuncs, C extensions, classes: only the name can be checked
    return ("other", getattr(obj, "__module__", None), getattr(obj, "__qualname__", getattr(obj, "__name__", None)))


def _source_digest(fun: Any) -> str | None:
    target = fun.func if isinstance(fun, functools.partial) else fun
    try:
        return _digest(inspect.getsource(target))
    except (OSError, TypeError):  # Lambdas typed in a REPL, builtins, C functions
        return None


def _code_fingerprint(fun: Any) -> dict[str, Any]:
    """
    Fingerprint of ``fun``. Across Python versions source differences prove a change, but source equality alone
    cannot validate bound state or helpers, so the comparison remains unverifiable.
    """
    return {
        "name": _stable_callable_name(fun),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "bytecode": _digest(_code_part(fun, set())),
        "source": _source_digest(fun),
    }


def _stable_callable_name(fun: Any) -> str:
    if isinstance(fun, functools.partial):
        return f"partial({_stable_callable_name(fun.func)})"
    return str(getattr(fun, "__name__", type(fun).__qualname__))


def _code_matches(saved: dict[str, Any], current: dict[str, Any]) -> bool | None:
    """True or False, or None when another Python version prevents complete validation."""
    if saved.get("name") != current["name"]:
        return False
    if saved.get("python") == current["python"]:
        return bool(saved.get("bytecode") == current["bytecode"])
    if saved.get("source") is not None and current["source"] is not None:
        if saved["source"] != current["source"]:
            return False
        # Source equality says nothing about defaults, closures, bound state or helper code.
        # Bytecode cannot validate those across interpreter versions: report this as unverifiable.
    return None


def _run_fingerprint(
    param_names: list[list[str]], param_values: list[list[list[Any]]], fun_kwargs: dict[str, Any]
) -> dict[str, Any]:
    return {
        "param_names": param_names,
        "n_values": [len(group[0]) for group in param_values],
        "param_values": _value_digest(param_values),
        "fun_kwargs": _value_digest(fun_kwargs),
    }


# ---------------------------------------------------------------------------------------------- files


def _write_atomic(path: Path, write: Callable[[BinaryIO], Any]) -> None:
    """Write a file so that ``path`` is either the old file or the complete new one, even if the job is killed."""
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "wb") as file:
        write(file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def _read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / _MANIFEST
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"The checkpoint manifest {path} cannot be read: {err}") from err
    if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT_VERSION:
        raise ValueError(
            f"{path} is not a parameterrun checkpoint of format {_FORMAT_VERSION}. Delete it or pass another "
            f"checkpoint directory."
        )
    return manifest


def _clear_directory(directory: Path) -> None:
    """Remove the files of a checkpoint (only those: the directory may hold other files of the user)."""
    for path in [
        directory / _MANIFEST,
        directory / f"{_MANIFEST}.tmp",
        *_chunk_files(directory),
        *_chunk_files(directory, temporary=True),
    ]:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _chunk_files(directory: Path, *, temporary: bool = False) -> list[Path]:
    suffix = ".tmp" if temporary else ""
    return [
        path
        for path in directory.glob(f"{_CHUNK_PATTERN}{suffix}")
        if re.fullmatch(r"chunk_[0-9]{6,}\.npz" + re.escape(suffix), path.name)
    ]


def _check_checkpoint_options(
    checkpoint_every: int | None, checkpoint_interval: float | None, checkpoint_resume: str
) -> None:
    if checkpoint_every is not None and (
        isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 1
    ):
        raise ValueError(f"checkpoint_every must be a positive integer or None, got {checkpoint_every!r}")
    if checkpoint_interval is not None and (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, (int, float))
        or not checkpoint_interval > 0
    ):
        raise ValueError(
            f"checkpoint_interval must be a positive number of seconds or None, got {checkpoint_interval!r}"
        )
    if checkpoint_resume not in _RESUME_MODES:
        raise ValueError(f'checkpoint_resume must be "auto", "force" or "restart", got {checkpoint_resume!r}')


def _save_triggers(checkpoint_every: int | None, checkpoint_interval: float | None) -> tuple[int | None, float | None]:
    """When to save: the given triggers, or every ``_DEFAULT_INTERVAL`` seconds if none is given."""
    if checkpoint_every is None and checkpoint_interval is None:
        return None, _DEFAULT_INTERVAL
    return checkpoint_every, None if checkpoint_interval is None else float(checkpoint_interval)


# ---------------------------------------------------------------------------------------------- chunks


def _write_chunk(file: BinaryIO, payloads: Sequence[Payload]) -> None:
    """
    Write payloads as one ``.npz`` file. Numeric channels, indices and failures are plain arrays (loaded with
    ``allow_pickle=False``); only non-numeric outputs (lists of Python objects) are pickled, as bytes. Channels that are
    None (their data lives in the ``result_file``) are not written.
    """
    arrays: dict[str, np.ndarray] = {"n_payloads": np.array(len(payloads))}
    for k, (indices, channels, failures) in enumerate(payloads):
        arrays[f"{k}_indices"] = np.asarray(indices, dtype=np.int64)
        arrays[f"{k}_n_channels"] = np.array(len(channels))
        for c, data in enumerate(channels if len(indices) else []):
            if data is None:
                continue
            if isinstance(data, np.ndarray) and data.dtype.kind in _NUMERIC_KINDS:
                arrays[f"{k}_array_{c}"] = data
            else:
                pickled = pickle.dumps(list(data), protocol=pickle.HIGHEST_PROTOCOL)
                arrays[f"{k}_pickle_{c}"] = np.frombuffer(pickled, dtype=np.uint8)
        if failures:
            arrays[f"{k}_failed"] = np.array([index for index, _ in failures], dtype=np.int64)
            arrays[f"{k}_failed_summary"] = np.array([failure.summary for _, failure in failures], dtype=str)
            arrays[f"{k}_failed_traceback"] = np.array([failure.traceback for _, failure in failures], dtype=str)
    np.savez(file, **arrays)  # type: ignore[arg-type]


def _read_chunk(path: Path) -> list[tuple[np.ndarray, list[Any], list[tuple[int, _FailedPoint]]]]:
    payloads = []
    with np.load(path, allow_pickle=False) as data:
        for k in range(int(data["n_payloads"])):
            indices = data[f"{k}_indices"]
            channels: list[Any] = []
            for c in range(int(data[f"{k}_n_channels"])):
                if f"{k}_array_{c}" in data.files:
                    channels.append(data[f"{k}_array_{c}"])
                elif f"{k}_pickle_{c}" in data.files:
                    channels.append(pickle.loads(data[f"{k}_pickle_{c}"].tobytes()))
                else:
                    channels.append(None)  # In the result_file, or no successful point in this payload
            failures: list[tuple[int, _FailedPoint]] = []
            if f"{k}_failed" in data.files:
                failures = [
                    (int(index), _FailedPoint(summary=str(summary), traceback=str(trace)))
                    for index, summary, trace in zip(
                        data[f"{k}_failed"], data[f"{k}_failed_summary"], data[f"{k}_failed_traceback"], strict=True
                    )
                ]
            payloads.append((indices, channels, failures))
    return payloads


# ---------------------------------------------------------------------------------------------- signals


class _SaveSignals:
    """
    Signals that ask a checkpointed run to save now, installed (in the main thread) for the duration of the run:

    * ``SIGUSR1`` (POSIX): save, and go on. With Slurm, ``#SBATCH --signal=USR1@600`` sends it 10 minutes before the
      walltime limit.
    * ``SIGTERM`` (sent by Slurm at the walltime limit or on ``scancel``, some 30 s before ``SIGKILL``): save, then
      terminate as the signal would have (``terminate``). Not installed if ``SIGTERM`` was ignored.

    The handlers only set flags; the backends look at them between points (and MPI ranks forward them to rank 0).
    """

    def __init__(self) -> None:
        self.save = False
        self.stop = False
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> "_SaveSignals":
        if threading.current_thread() is not threading.main_thread():
            return self  # Handlers can only be set in the main thread
        for name in ("SIGTERM", "SIGUSR1"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                if signal.getsignal(signum) is signal.SIG_IGN:
                    continue
                self._previous[signum] = signal.signal(signum, self._handle)
            except (ValueError, OSError):  # Not supported here
                continue
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._restore()

    def _handle(self, signum: int, frame: Any) -> None:
        self.save = True
        if signum == signal.SIGTERM:
            self.stop = True

    def _restore(self) -> None:
        for signum, previous in self._previous.items():
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
        self._previous = {}

    def terminate(self) -> None:
        """
        After the save: restore the previous SIGTERM handler and raise the signal again, so the process ends as it would
        have without us (or the user's own handler runs). Raises SystemExit if that handler returns.
        """
        self._restore()
        signal.raise_signal(signal.SIGTERM)
        raise SystemExit(128 + int(signal.SIGTERM))


# ---------------------------------------------------------------------------------------------- checkpoint


class _Checkpoint:
    """
    Rank 0's side of a checkpoint: the ``_ResultStore`` of the run (with the points of earlier runs already in it),
    and the points added since the last save. ``add`` stores a payload, ``due`` tells whether it is time to save, and
    ``commit`` writes the new points as one chunk and then the manifest.

    Channels that live in the ``result_file`` (memory-mapped) are not copied into the chunks: ``commit`` flushes the
    file instead, and resuming reopens it. If such a channel stops living in its file (e.g. a later point returns
    another shape), the next ``commit`` writes one chunk with every finished point and drops the older chunks.
    """

    def __init__(
        self,
        directory: Path,
        manifest: dict[str, Any],
        store: _ResultStore,
        every: int | None,
        interval: float | None,
    ) -> None:
        self.directory = directory
        self.manifest = manifest
        self.store = store
        self.every = every
        self.interval = interval
        self.signals = _SaveSignals()
        self.n_resumed = int(store.done.sum())  # Points loaded from earlier runs
        self._pending: list[Payload] = []
        self._pending_points = 0
        self._stripped: set[int] = {c for c, mapped in enumerate(manifest.get("mapped", [])) if mapped}
        self._last_save = monotonic()

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        fun: Any,
        param_names: list[list[str]],
        param_values: list[list[list[Any]]],
        fun_kwargs: dict[str, Any],
        result_file: str | os.PathLike[str] | None = None,
        every: int | None = None,
        interval: float | None = None,
        resume: CheckpointResume = "auto",
    ) -> "_Checkpoint":
        """
        Open the checkpoint directory: resume from it if it holds a snapshot of the same run, start a new one if it is
        empty or missing. Raises ValueError if it belongs to another run (unless ``resume="restart"``).
        """
        directory = Path(path)
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"checkpoint must be a directory, but {directory} is a file")
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise ValueError(f"The checkpoint directory is not writable: {directory}")

        n_values = [len(group[0]) for group in param_values]
        n_total = int(np.prod(n_values))
        run = _run_fingerprint(param_names, param_values, fun_kwargs)
        code = _code_fingerprint(fun)
        store = _ResultStore(n_total, n_values, result_file, preserve_files=True)

        manifest = None if resume == "restart" else _read_manifest(directory)
        if manifest is None:
            _clear_directory(directory)
            manifest = {"format": _FORMAT_VERSION, "run": run, "code": code, "chunks": [], "n_done": 0}
            _write_manifest(directory, manifest)
        else:
            _check_same_run(directory, manifest, run, code, resume)
            manifest["code"] = code  # After a forced resume, later checks compare against the new code
            if manifest.get("specs"):
                store.allocate(manifest["specs"], _reopen_result_files(directory, manifest, result_file, n_values))
            _replay_chunks(directory, manifest["chunks"], store)
            # Chunks written after the last manifest (a crash in between) are not part of the snapshot
            listed = set(manifest["chunks"])
            for stray in _chunk_files(directory):
                if stray.name not in listed:
                    stray.unlink()

        every, interval = _save_triggers(every, interval)
        return cls(directory, manifest, store, every, interval)

    @property
    def n_saved(self) -> int:
        return int(self.manifest["n_done"])

    def add(self, payload: Payload) -> None:
        indices, channels, failures = payload
        self.store.write(indices, channels)  # Raises before anything is kept if the outputs are inconsistent
        self.store.add_failures(failures)
        n_points = len(indices) + len(failures)
        if n_points:
            # Channels in the result_file are saved by flushing it: keep only the other ones until the next commit
            mapped = self.store.mapped()
            kept: list[Any] = []
            for c, data in enumerate(channels):
                if c < len(mapped) and mapped[c]:
                    self._stripped.add(c)
                    kept.append(None)
                else:
                    kept.append(data)
            self._pending.append((indices, kept, failures))
            self._pending_points += n_points

    def due(self, unflushed: int = 0) -> bool:
        """
        Time to save? ``unflushed``: points finished but not ``add``-ed yet (still in the caller's buffer). A signal
        (SIGUSR1, SIGTERM) makes it due at once.
        """
        if self.signals.save:
            return True
        if self.every is not None and self._pending_points + unflushed >= self.every:
            return True
        return self.interval is not None and monotonic() - self._last_save >= self.interval

    def commit(self) -> bool:
        """Save the points added since the last save. Returns whether anything was written."""
        self._last_save = monotonic()
        self.signals.save = False
        mapped = self.store.mapped()
        # A channel whose data was left to its file no longer lives there: the old chunks lack it, so save everything
        compact = any(c >= len(mapped) or not mapped[c] for c in self._stripped)
        if not self._pending and not compact:
            return False

        self.store.flush()  # The result_file holds its channels before the manifest says their points are saved
        payloads: Sequence[Payload] = [self.store.full_payload()] if compact else self._pending  # type: ignore[list-item]
        old_chunks = list(self.manifest["chunks"])
        name = f"chunk_{self.manifest.get('next_chunk', len(old_chunks)):06d}.npz"
        _write_atomic(self.directory / name, lambda file: _write_chunk(file, payloads))
        manifest = {
            **self.manifest,
            "chunks": [name] if compact else [*old_chunks, name],
            "next_chunk": self.manifest.get("next_chunk", len(old_chunks)) + 1,
            "n_done": int(self.store.done.sum()),
            "specs": self.store.specs(),
            "mapped": mapped,
            "result_file": str(Path(self.store.result_file).resolve()) if self.store.result_file is not None else None,
        }
        _write_manifest(self.directory, manifest)
        self.manifest = manifest
        self.store.remove_retired_files()
        if compact:
            for old in old_chunks:
                with contextlib.suppress(OSError):
                    (self.directory / old).unlink()
            self._stripped = {c for c, is_mapped in enumerate(mapped) if is_mapped}
        self._pending = []
        self._pending_points = 0
        return True

    def commit_quietly(self) -> None:
        """Save what can be saved while another exception is on its way up (never hides that exception)."""
        with contextlib.suppress(Exception):
            self.commit()

    def remove(self) -> None:
        """Delete the snapshot (after success, with ``keep_checkpoint=False``). Removes the directory if it is empty."""
        _clear_directory(self.directory)
        with contextlib.suppress(OSError):
            self.directory.rmdir()

    def remaining(self) -> np.ndarray | None:
        """Flat indices still to compute, or None if nothing was resumed (all of them, never materialized)."""
        return None if self.n_resumed == 0 else np.flatnonzero(~self.store.done)


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    data = json.dumps(manifest, indent=1).encode("utf-8")
    _write_atomic(directory / _MANIFEST, lambda file: file.write(data))


def _reopen_result_files(
    directory: Path,
    manifest: dict[str, Any],
    result_file: str | os.PathLike[str] | None,
    n_values: list[int],
) -> dict[int, np.ndarray]:
    """
    Memory-map the ``result_file`` channels of a resumed snapshot (``r+``): their data is in the files, not in the
    chunks. Raises ValueError if a file is missing or no longer matches.
    """
    mapped = [c for c, is_mapped in enumerate(manifest.get("mapped", [])) if is_mapped]
    if not mapped:
        return {}
    saved_file = manifest.get("result_file")
    if result_file is None or str(Path(result_file).resolve()) != saved_file:
        raise ValueError(
            f"The checkpoint in {directory} keeps its numeric outputs in result_file={saved_file!r}, but this run has "
            f"result_file={None if result_file is None else str(result_file)!r}. Pass the same result_file to resume, "
            f'or checkpoint_resume="restart" to start over.'
        )

    specs = manifest["specs"]
    paths = _channel_paths(result_file, len(specs))
    existing: dict[int, np.ndarray] = {}
    for c in mapped:
        output_shape = tuple(specs[c][1])
        try:
            array = np.load(paths[c], mmap_mode="r+")
        except (OSError, ValueError) as err:
            raise ValueError(
                f"The checkpoint in {directory} needs {paths[c]}, which cannot be opened ({err}). Pass "
                f'checkpoint_resume="restart" to start over.'
            ) from err
        saved_dtype = np.dtype(specs[c][2])
        # The file may have been promoted since (e.g. to float64 for NaN, when the run finished): never narrower
        if array.shape != (*n_values, *output_shape) or np.result_type(saved_dtype, array.dtype) != array.dtype:
            problem = (
                f"{paths[c]} no longer matches the checkpoint in {directory} (shape {array.shape}, dtype "
                f"{array.dtype}; expected {(*n_values, *output_shape)}, {saved_dtype}). Pass "
                f'checkpoint_resume="restart" to start over.'
            )
            del array  # Unmap before raising: the traceback would keep the file mapped (and locked on Windows)
            existing.clear()
            raise ValueError(problem)
        specs[c] = ["array", list(output_shape), array.dtype.str]
        existing[c] = array.reshape(int(np.prod(n_values)), *output_shape)
    return existing


def _check_same_run(
    directory: Path, manifest: dict[str, Any], run: dict[str, Any], code: dict[str, Any], resume: str
) -> None:
    saved_run = manifest.get("run", {})
    differences = [
        key for key in ("param_names", "n_values", "param_values", "fun_kwargs") if saved_run.get(key) != run[key]
    ]
    if differences:
        raise ValueError(
            f"The checkpoint in {directory} belongs to another run: {', '.join(differences)} differ. Pass another "
            f'checkpoint directory, or checkpoint_resume="restart" to discard it.'
        )

    matches = _code_matches(manifest.get("code", {}), code)
    if matches is None:
        warnings.warn(
            f"Resuming the checkpoint in {directory}, written under Python {manifest['code'].get('python')}: whether "
            f"{code['name']} changed cannot be checked under Python {code['python']} "
            "(source equality alone cannot validate captured state or helpers).",
            RuntimeWarning,
            stacklevel=4,
        )
    elif not matches:
        if resume != "force":
            raise ValueError(
                f"The code of {code['name']} changed since the checkpoint in {directory} was written (or a helper "
                f"function of the same module, a default value or a closure variable did). Its {manifest['n_done']} "
                f'saved points could differ from new ones. Pass checkpoint_resume="force" to resume anyway (e.g. '
                f'after a harmless edit), or checkpoint_resume="restart" to discard it.'
            )
        warnings.warn(
            f"The code of {code['name']} changed since the checkpoint in {directory} was written; resuming anyway "
            f'(checkpoint_resume="force").',
            RuntimeWarning,
            stacklevel=4,
        )


def _replay_chunks(directory: Path, chunks: Sequence[str], store: _ResultStore) -> None:
    for name in chunks:
        try:
            payloads = _read_chunk(directory / name)
        except Exception as err:
            raise ValueError(
                f"The checkpoint chunk {directory / name} is missing or damaged ({type(err).__name__}: {err}). "
                f'Pass checkpoint_resume="restart" to start over.'
            ) from err
        for indices, channels, failures in payloads:
            store.write(indices, channels)
            store.add_failures(failures)
        del payloads
