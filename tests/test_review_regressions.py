"""Regression coverage for checkpoint integrity and MPI failure/transfer protocols."""

import builtins
import functools
import gc
import os
import shutil
import subprocess
import sys
import textwrap
import types
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import parameterrun._checkpoint as checkpoints
import parameterrun.parallel_utils as pu
from parameterrun._checkpoint import _Checkpoint, _code_fingerprint, _code_matches
from parameterrun._delivery import _item_layout
from parameterrun._errors import _nan_like
from parameterrun._mpi_transfer import _recv_payload, _send_payload
from parameterrun._store import _ChunkBuffer
from parameterrun._utils import _mpi_world_size_from_env


def _identity(x: Any) -> Any:
    return x


class _Scale:
    def __init__(self, scale: int) -> None:
        self.scale = scale

    def __call__(self, x: int) -> int:
        return self.scale * x


def test_fingerprint_detects_swapped_keyword_bindings() -> None:
    namespace: dict[str, Any] = {}
    exec("def fun(x, y): return x - y", namespace)
    before = _code_fingerprint(namespace["fun"])
    exec("def fun(y, x): return y - x", namespace)
    assert _code_matches(before, _code_fingerprint(namespace["fun"])) is False


@pytest.mark.parametrize(
    "replacement",
    [
        {"co_flags": _identity.__code__.co_flags | 0x80},
        {"co_posonlyargcount": 1},
        {"co_exceptiontable": b"\x80\x01\x00\x00"},
        {"co_freevars": ("changed",)},
        {"co_cellvars": ("changed",)},
    ],
)
def test_fingerprint_includes_execution_metadata(replacement: dict[str, Any]) -> None:
    # Compare metadata only; these synthetic functions must never be executed.
    code = _identity.__code__.replace(**replacement)
    closure = tuple(types.CellType(1) for _ in code.co_freevars) or None
    changed = types.FunctionType(code, globals(), closure=closure)
    assert _code_matches(_code_fingerprint(_identity), _code_fingerprint(changed)) is False


@pytest.mark.parametrize("change", ["closure", "default", "helper"])
def test_equal_source_across_interpreters_never_validates_changed_state(change: str) -> None:
    def make(scale: int) -> Any:
        def helper(x: int, offset: int = scale) -> int:
            return x + offset

        if change == "closure":

            def fun(x: int) -> int:
                return scale * x
        elif change == "default":

            def with_default(x: int, factor: int = scale) -> int:
                return factor * x

            return with_default
        else:

            def fun(x: int) -> int:
                return helper(x)

        return fun

    saved = {**_code_fingerprint(make(1)), "python": "other"}
    current = _code_fingerprint(make(2))
    assert saved["source"] == current["source"]
    assert _code_matches(saved, current) is None


@pytest.mark.parametrize("kind", ["partial", "instance"])
def test_recreated_callables_resume_and_reject_changed_state(tmp_path: Path, kind: str) -> None:
    def make(scale: int) -> Any:
        return functools.partial(pow, exp=scale) if kind == "partial" else _Scale(scale)

    names = "base" if kind == "partial" else "x"
    options: dict[str, Any] = dict(backend="joblib", n_workers=1, pbar_bool=False, checkpoint=tmp_path)
    first = pu.parameterrun(make(2), names, [2, 3], **options)
    np.testing.assert_array_equal(pu.parameterrun(make(2), names, [2, 3], **options), first)
    with pytest.raises(ValueError, match="code of .* changed"):
        pu.parameterrun(make(3), names, [2, 3], **options)


@pytest.mark.parametrize("action", ["open", "restart", "remove"])
def test_checkpoint_cleanup_preserves_unrelated_temporary_files(tmp_path: Path, action: str) -> None:
    if action == "restart":
        _Checkpoint.open(tmp_path, _identity, [["x"]], [[[0]]], {})
    unrelated = tmp_path / "simulation.tmp"
    unrelated.write_bytes(b"user data")
    (tmp_path / "chunk_notes.npz.tmp").write_bytes(b"notes")
    (tmp_path / "chunk_notes.npz").write_bytes(b"notes")
    (tmp_path / "manifest.json.tmp").write_bytes(b"interrupted manifest")
    (tmp_path / "chunk_000012.npz.tmp").write_bytes(b"interrupted chunk")
    checkpoint = _Checkpoint.open(
        tmp_path, _identity, [["x"]], [[[0]]], {}, resume="restart" if action == "restart" else "auto"
    )
    if action == "remove":
        checkpoint.remove()
    assert unrelated.read_bytes() == b"user data"
    assert (tmp_path / "chunk_notes.npz.tmp").read_bytes() == b"notes"
    assert (tmp_path / "chunk_notes.npz").read_bytes() == b"notes"
    assert not (tmp_path / "manifest.json.tmp").exists()
    assert not (tmp_path / "chunk_000012.npz.tmp").exists()


@pytest.mark.parametrize("failure", ["chunk", "manifest"])
def test_failed_compaction_keeps_the_previous_snapshot_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    directory, target = tmp_path / "checkpoint", tmp_path / "out.npy"
    checkpoint = _Checkpoint.open(directory, _identity, [["x"]], [[[0, 1]]], {}, result_file=target)
    checkpoint.add((np.array([0]), [np.array([[1, 2]])], []))
    checkpoint.commit()
    old_manifest = (directory / "manifest.json").read_bytes()
    checkpoint.add((np.array([1]), [[np.array([3, 4, 5])]], []))
    assert target.exists()  # The committed snapshot still needs this mapped channel.

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected save failure")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_write_chunk" if failure == "chunk" else "_write_manifest", fail)
        with pytest.raises(OSError, match="injected"):
            checkpoint.commit()
    assert (directory / "manifest.json").read_bytes() == old_manifest
    restored = _Checkpoint.open(directory, _identity, [["x"]], [[[0, 1]]], {}, result_file=target)
    assert restored.store.done.tolist() == [True, False]
    np.testing.assert_array_equal(restored.store.full_payload()[0], [0])
    np.testing.assert_array_equal(np.load(target)[0], [1, 2])
    del restored
    gc.collect()  # Release Windows file mappings before the successful compaction removes the file.

    checkpoint.commit()
    assert not target.exists()
    restored = _Checkpoint.open(directory, _identity, [["x"]], [[[0, 1]]], {}, result_file=target)
    assert restored.store.done.all()
    channels, _ = restored.store.finalize([["x"]], [[[0, 1]]])
    np.testing.assert_array_equal(channels[0][0], [1, 2])
    np.testing.assert_array_equal(channels[0][1], [3, 4, 5])


def test_layout_preserves_nested_structured_dtype_and_offsets() -> None:
    dtype = np.dtype(
        {"names": ["position", "value"], "formats": [("f8", (2,)), "i4"], "offsets": [0, 24], "itemsize": 32}
    )
    array = np.zeros(3, dtype=dtype)
    assert np.dtype(_item_layout(array)[2]) == dtype


def test_ragged_nan_fallback_in_joblib_and_checkpoint(tmp_path: Path) -> None:
    def fun(x: int) -> list[list[int]]:
        if x == 1:
            raise ValueError("bad point")
        return [[1], [2, 3]]

    assert _nan_like([[1], [2, 3]]) is None
    for checkpoint in (None, tmp_path):
        with pytest.warns(RuntimeWarning, match="1 of 2 points raised"):
            result = pu.parameterrun(
                fun,
                "x",
                [0, 1],
                backend="joblib",
                n_workers=1,
                pbar_bool=False,
                on_error="nan",
                reshape=False,
                checkpoint=checkpoint,
            )
        assert result == [[[[1], [2, 3]], None]]


@pytest.mark.parametrize(
    "output",
    ["x" * 2**20, [np.ones(2**17)], {"data": b"x" * 2**20}, np.array(["x" * 2**20], dtype=object)],
    ids=["string", "nested-array", "dict", "object-array"],
)
def test_buffer_accounts_for_variable_length_serialized_outputs(output: Any) -> None:
    buffer = _ChunkBuffer()
    buffer.add(0, output, {})
    assert buffer.nbytes >= 2**20


def test_serialized_payloads_split_even_one_oversized_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import parameterrun._store as stores

    # The transport is simulated: loading a native MPI runtime is unnecessary here.
    mpi4py = types.ModuleType("mpi4py")
    mpi4py.__dict__["MPI"] = types.SimpleNamespace(BYTE=object())
    monkeypatch.setitem(sys.modules, "mpi4py", mpi4py)
    monkeypatch.setattr(stores, "_MAX_MESSAGE_BYTES", 1024)

    class Transport:
        def __init__(self) -> None:
            self.nbytes = 0
            self.blocks: list[bytes] = []
            self.received = 0

        def send(self, nbytes: int, **kwargs: Any) -> None:
            self.nbytes = nbytes

        def Send(self, buffer: Any, **kwargs: Any) -> None:
            block = buffer[0].tobytes()
            assert len(block) <= 1024
            self.blocks.append(block)

        def Recv(self, buffer: Any, **kwargs: Any) -> None:
            buffer[0][:] = np.frombuffer(self.blocks[self.received], dtype=np.uint8)
            self.received += 1

    comm = Transport()
    payload = ("x" * 5000, [np.arange(5000)], {"values": b"y" * 5000})
    _send_payload(comm, payload, dest=0, tag=11)
    assert len(comm.blocks) > 1
    assert comm.nbytes == sum(map(len, comm.blocks))
    restored = _recv_payload(comm, comm.nbytes, source=1, tag=11)
    assert restored[0] == payload[0]
    np.testing.assert_array_equal(restored[1][0], payload[1][0])
    assert restored[2] == payload[2]
    assert comm.received == len(comm.blocks)


def test_subprocess_runner_does_not_import_mpi_in_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def without_mpi(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise AssertionError("The pytest parent must not initialize MPI")
        return original_import(name, *args, **kwargs)

    def completed(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0, "ALL RANKS COMPLETED", "")

    monkeypatch.setattr(builtins, "__import__", without_mpi)
    monkeypatch.setattr(shutil, "which", lambda name: "mpiexec")
    monkeypatch.setattr(sys.modules[__name__], "find_spec", lambda name: object())
    monkeypatch.setattr(subprocess, "run", completed)
    test_mpi_regressions_finish_without_hanging(tmp_path, "warning", "static", False, False)


_MPI_SCRIPT = textwrap.dedent("""
    import faulthandler
    import sys
    import time
    import warnings
    from pathlib import Path
    import numpy as np
    from mpi4py import MPI
    import parameterrun.parallel_utils as pu
    import parameterrun._store as stores
    import parameterrun._mpi_static as static
    from parameterrun._checkpoint import _Checkpoint
    from parameterrun._delivery import _broadcast_result

    # Dump the blocked MPI calls before the parent's timeout, so a CI hang has useful diagnostics.
    faulthandler.dump_traceback_later(30, exit=True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mode, schedule, checkpointed, directory = sys.argv[1:]
    directory = Path(directory)
    options = dict(backend="mpi", mpi_schedule=schedule, pbar_bool=False, return_on="all")
    if checkpointed == "yes":
        options.update(checkpoint=directory / "checkpoint", checkpoint_every=4)

    if mode == "structured":
        dtype = np.dtype({"names": ["xy", "value"], "formats": [("f8", (2,)), "i4"],
                          "offsets": [0, 24], "itemsize": 32})
        expected = np.zeros(5, dtype=dtype)
        expected["xy"] = np.arange(10).reshape(5, 2)
        expected["value"] = np.arange(5)
        result = _broadcast_result(comm, expected if rank == 0 else None, None)
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, expected)
    elif mode == "receiver_ready":
        original_broadcast = pu._broadcast_result
        class TracedComm:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.sequence = 0
            def __getattr__(self, name):
                return getattr(self.wrapped, name)
            def Bcast(self, buffer, root):
                print(f"rank {rank}: Bcast {self.sequence} start, bytes={buffer[0].nbytes}", flush=True)
                self.wrapped.Bcast(buffer, root=root)
                print(f"rank {rank}: Bcast {self.sequence} end", flush=True)
                self.sequence += 1
        def traced_broadcast(comm, result, error):
            return original_broadcast(TracedComm(comm), result, error)
        pu._broadcast_result = traced_broadcast
        original_write = stores._ResultStore.write_channel
        original_send = static._send_rows
        marker = directory / "receiver_ready"
        def delayed_write(self, *args, **kwargs):
            original_write(self, *args, **kwargs)
            if not marker.exists():
                time.sleep(0.5)
                marker.touch()
        def checked_send(comm, tag, array):
            assert marker.exists(), "worker sent before root was ready"
            original_send(comm, tag, array)
        stores._ResultStore.write_channel = delayed_write
        static._send_rows = checked_send
        def fun(x):
            return np.full(5000, x, dtype=float), str(x), np.full(5000, x * 1j)
        numbers, labels, complex_numbers = pu.parameterrun(fun, "x", range(20), **options)
        for x in range(20):
            np.testing.assert_array_equal(numbers[x], np.full(5000, x))
            np.testing.assert_array_equal(complex_numbers[x], np.full(5000, x * 1j))
        assert list(labels) == [str(x) for x in range(20)]
    elif mode == "large":
        stores._MAX_MESSAGE_BYTES = 1024
        def fun(x):
            return str(x) * 5000, [np.full(800, x)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pu.parameterrun(fun, "x", range(12), **options)
        assert list(result[0]) == [str(x) * 5000 for x in range(12)]
        for x, value in enumerate(result[1]):
            np.testing.assert_array_equal(value, [np.full(800, x)])
    else:
        original_finalize = stores._ResultStore.finalize
        original_write = stores._ResultStore.write_channel
        original_commit = _Checkpoint.commit
        if mode in ("finalize_io", "late_sender"):
            def fail(self, *args, **kwargs):
                if mode == "late_sender":
                    assert (directory / "sender_finished").exists(), "root finalized before transfers completed"
                raise OSError("injected finalization failure")
            stores._ResultStore.finalize = fail
        if mode == "late_sender":
            original_send_rows = static._send_rows
            def delayed_send(comm, tag, array):
                original_send_rows(comm, tag, array)
                if rank == 1:
                    time.sleep(0.5)
                    (directory / "sender_finished").touch()
            static._send_rows = delayed_send
        if mode in ("transfer", "round_store"):
            def fail(self, *args, **kwargs):
                raise OSError("injected storage failure")
            stores._ResultStore.write_channel = fail
        if mode == "commit":
            def fail(self, *args, **kwargs):
                raise OSError("injected commit failure")
            _Checkpoint.commit = fail
        if mode == "allocation":
            target = directory / "out.npy"
            if rank == 0:
                target.mkdir()
            comm.Barrier()
            options["result_file"] = target
        def fun(x):
            if mode == "warning" and x == 1:
                raise ValueError("bad point")
            return np.full(5000, x, dtype=float)
        caught = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                pu.parameterrun(fun, "x", range(20), on_error="nan", **options)
        except Exception as err:
            if rank == 0 and mode in ("warning", "finalize_io", "late_sender"):
                assert isinstance(err, RuntimeWarning if mode == "warning" else OSError), repr(err)
            assert "injected" in str(err) or "points raised" in str(err) or mode == "allocation", str(err)
            caught = True
        assert caught, (rank, mode)
        assert all(comm.allgather(caught))
        stores._ResultStore.finalize = original_finalize
        stores._ResultStore.write_channel = original_write
        _Checkpoint.commit = original_commit
        if mode == "late_sender":
            static._send_rows = original_send_rows
        # A failed private communicator must not contaminate a later sweep.
        result = pu.parameterrun(lambda x: x, "x", range(8), backend="mpi", pbar_bool=False, return_on="all")
        np.testing.assert_array_equal(result, np.arange(8))
    comm.Barrier()
    faulthandler.cancel_dump_traceback_later()
    if rank == 0:
        print("ALL RANKS COMPLETED", flush=True)
""")


@pytest.mark.mpi_subprocess
@pytest.mark.parametrize(
    ("mode", "schedule", "checkpointed", "aborts"),
    [
        (mode, schedule, checkpointed, False)
        for mode in ("warning", "finalize_io")
        for schedule in ("static", "dynamic")
        for checkpointed in (False, True)
    ]
    + [
        ("allocation", "static", False, False),
        ("round_store", "static", True, False),
        ("commit", "static", True, False),
        ("transfer", "static", False, True),
        ("late_sender", "static", False, False),
        ("receiver_ready", "static", False, False),
        ("structured", "static", False, False),
        ("large", "static", False, False),
        ("large", "static", True, False),
        ("large", "dynamic", False, False),
        ("large", "dynamic", True, False),
    ],
)
def test_mpi_regressions_finish_without_hanging(
    tmp_path: Path, mode: str, schedule: str, checkpointed: bool, aborts: bool
) -> None:
    launcher = shutil.which("mpiexec")
    if launcher is None or find_spec("mpi4py") is None:
        pytest.skip("Needs an MPI launcher and mpi4py")
    if _mpi_world_size_from_env() > 1:
        pytest.skip("Do not nest MPI jobs")
    script = tmp_path / "regression.py"
    script.write_text(_MPI_SCRIPT, encoding="utf-8")
    env = {
        **os.environ,
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
    }
    result = subprocess.run(
        [
            launcher,
            "-n",
            "4",
            sys.executable,
            str(script),
            mode,
            schedule,
            "yes" if checkpointed else "no",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
        check=False,
    )
    output = result.stdout + result.stderr
    if aborts:
        assert result.returncode != 0, output
        assert "injected storage failure" in output
    else:
        assert result.returncode == 0, output
        assert "ALL RANKS COMPLETED" in output
