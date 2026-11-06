"""Checkpoints: saving, resuming, validation of the run and of the code of fun (joblib backend, and the MPI paths)."""

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

import parameterrun.parallel_utils as pu
from parameterrun._checkpoint import _Checkpoint, _code_fingerprint, _code_matches, _read_chunk, _SaveSignals
from parameterrun._utils import _mpi_world_size_from_env

# Points evaluated by the serial tests, and the point at which `flaky` raises (None: never)
CALLS: list[int] = []
FAIL_AT: dict[str, int | None] = {"x": None}


def flaky(x: int, y: int) -> tuple[float, str]:
    CALLS.append(x * 10 + y)
    if FAIL_AT["x"] is not None and x * 10 + y == FAIL_AT["x"]:
        raise RuntimeError("node died")
    return float(x * y), f"{x}-{y}"


XS = list(range(4))
YS = list(range(5))


def _run(directory: Path, **kwargs: Any) -> Any:
    options: dict[str, Any] = {"n_workers": 1, "pbar_bool": False, "backend": "joblib", "checkpoint": directory}
    return pu.parameterrun(flaky, ["x", "y"], [XS, YS], **{**options, **kwargs})


@pytest.fixture(autouse=True)
def _reset() -> None:
    CALLS.clear()
    FAIL_AT["x"] = None


def _expected() -> list[np.ndarray]:
    numbers = np.array([[float(x * y) for y in YS] for x in XS])
    labels = np.array([[f"{x}-{y}" for y in YS] for x in XS])
    return [numbers, labels]


def _manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    return manifest


# ---------------------------------------------------------------- resume


def test_interrupted_run_resumes_and_computes_only_the_missing_points(tmp_path: Path) -> None:
    FAIL_AT["x"] = 23  # The 14th point
    with pytest.raises(RuntimeError, match="node died"):
        _run(tmp_path, checkpoint_every=4)
    assert _manifest(tmp_path)["n_done"] == 13  # Everything before the failure was saved

    FAIL_AT["x"] = None
    CALLS.clear()
    result = _run(tmp_path, checkpoint_every=4)
    assert len(CALLS) == 20 - 13
    for got, expected in zip(result, _expected(), strict=True):
        np.testing.assert_array_equal(got, expected)


def test_finished_run_returns_from_the_snapshot_without_computing(tmp_path: Path) -> None:
    first = _run(tmp_path)
    CALLS.clear()
    second = _run(tmp_path)
    assert CALLS == []
    for got, expected in zip(second, first, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_parallel_joblib_saves_before_raising_and_resumes(tmp_path: Path) -> None:
    marker = tmp_path / "fail"  # Exists: the last point fails. Seen by the joblib worker processes too
    marker.touch()

    def fails_at_the_end(x: int) -> int:
        if x == 39 and marker.exists():
            time.sleep(1.0)  # Let the consumer read the earlier results (joblib raises as soon as a task fails)
            raise ValueError("late failure")
        return x * x

    kwargs: dict[str, Any] = {"n_workers": 2, "pbar_bool": False, "backend": "joblib", "checkpoint": tmp_path / "c"}
    with pytest.raises(ValueError, match="late failure"):
        pu.parameterrun(fails_at_the_end, "x", range(40), **kwargs)
    # joblib may raise before it yields every result finished earlier, so not all 39 points need to be saved
    assert 0 < _manifest(tmp_path / "c")["n_done"] <= 39

    marker.unlink()
    np.testing.assert_array_equal(pu.parameterrun(fails_at_the_end, "x", range(40), **kwargs), np.arange(40) ** 2)


def test_checkpoint_every_writes_one_chunk_per_save(tmp_path: Path) -> None:
    _run(tmp_path, checkpoint_every=6)
    assert _manifest(tmp_path)["chunks"] == [f"chunk_{k:06d}.npz" for k in range(4)]  # 6 + 6 + 6 + 2 points
    assert _manifest(tmp_path)["n_done"] == 20


def test_keep_checkpoint_false_deletes_the_snapshot(tmp_path: Path) -> None:
    directory = tmp_path / "snap"
    _run(directory, keep_checkpoint=False)
    assert not directory.exists()


def test_failed_points_are_saved_and_count_as_done(tmp_path: Path) -> None:
    FAIL_AT["x"] = 12
    with pytest.warns(RuntimeWarning, match="1 of 20 points raised"):
        _run(tmp_path, on_error="nan")
    CALLS.clear()
    with pytest.warns(RuntimeWarning, match="1 of 20 points raised"):
        numbers, labels = _run(tmp_path, on_error="nan")
    assert CALLS == []
    assert np.isnan(numbers[1, 2])
    assert labels[1, 2] is None


def test_result_file_channels_are_not_copied_into_the_chunks(tmp_path: Path) -> None:
    """Numeric outputs live in the memory-mapped result_file: chunks only record which points are done."""
    target, directory = tmp_path / "result.npy", tmp_path / "c"
    FAIL_AT["x"] = 23
    with pytest.raises(RuntimeError, match="node died"):
        _run(directory, checkpoint_every=4, result_file=target)
    manifest = _manifest(directory)
    assert manifest["mapped"] == [True, False]  # Numbers in result_0.npy; labels (strings) in the chunks
    for name in manifest["chunks"]:
        with np.load(directory / name, allow_pickle=False) as data:
            assert not [key for key in data.files if "_array_" in key]

    FAIL_AT["x"] = None
    CALLS.clear()
    result = _run(directory, checkpoint_every=4, result_file=target)
    assert len(CALLS) == 20 - 13
    for got, expected in zip(result, _expected(), strict=True):
        np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(np.load(tmp_path / "result_0.npy"), _expected()[0])


def test_resuming_needs_the_same_intact_result_file(tmp_path: Path) -> None:
    target, directory = tmp_path / "result.npy", tmp_path / "c"
    _run(directory, result_file=target)
    with pytest.raises(ValueError, match="Pass the same result_file"):
        _run(directory)
    with pytest.raises(ValueError, match="Pass the same result_file"):
        _run(directory, result_file=tmp_path / "other.npy")
    np.save(tmp_path / "result_0.npy", np.zeros(3))
    with pytest.raises(ValueError, match="no longer matches"):
        _run(directory, result_file=target)
    (tmp_path / "result_0.npy").unlink()
    with pytest.raises(ValueError, match="cannot be opened"):
        _run(directory, result_file=target)
    result = _run(directory, result_file=target, checkpoint_resume="restart")
    np.testing.assert_array_equal(result[0], _expected()[0])


def test_resume_after_the_result_file_was_promoted_to_float(tmp_path: Path) -> None:
    """A finished run fills failed points with NaN, which turns an integer result_file into float64."""

    def maybe(x: int) -> int:
        if x == 2:
            raise ValueError("no")
        return x

    target = tmp_path / "result.npy"
    kwargs: dict[str, Any] = {"n_workers": 1, "pbar_bool": False, "backend": "joblib", "on_error": "nan"}
    with pytest.warns(RuntimeWarning, match="1 of 4 points"):
        pu.parameterrun(maybe, "x", range(4), result_file=target, checkpoint=tmp_path / "c", **kwargs)
    assert np.load(target).dtype == np.float64
    with pytest.warns(RuntimeWarning, match="1 of 4 points"):
        again = pu.parameterrun(maybe, "x", range(4), result_file=target, checkpoint=tmp_path / "c", **kwargs)
    np.testing.assert_array_equal(again, [0, 1, np.nan, 3])


def test_channel_leaving_its_result_file_compacts_the_snapshot(tmp_path: Path) -> None:
    """A later output of another shape moves the channel out of its file: the next save writes every point."""

    def growing(x: int) -> np.ndarray:
        return np.full(2 if x < 5 else 3, float(x))

    kwargs: dict[str, Any] = {
        "n_workers": 1,
        "pbar_bool": False,
        "backend": "joblib",
        "checkpoint": tmp_path / "c",
        "checkpoint_every": 2,
        "result_file": tmp_path / "result.npy",
    }
    with pytest.warns(RuntimeWarning):  # Cannot reshape, and the result_file cannot be written
        first = pu.parameterrun(growing, "x", range(8), **kwargs)
    manifest = _manifest(tmp_path / "c")
    assert manifest["mapped"] == [False]
    assert manifest["chunks"][0] != "chunk_000000.npz"  # The early chunks without data were dropped
    assert sorted(path.name for path in (tmp_path / "c").glob("chunk_*")) == sorted(manifest["chunks"])

    with pytest.warns(RuntimeWarning):
        second = pu.parameterrun(growing, "x", range(8), **kwargs)
    assert isinstance(first, list) and isinstance(second, list)
    (first_values,), (second_values,) = first, second  # One channel, not reshaped: a flat list of ragged arrays
    for x, (got, expected) in enumerate(zip(second_values, first_values, strict=True)):
        np.testing.assert_array_equal(got, growing(x))
        np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------- validation


def test_other_parameter_values_raise(tmp_path: Path) -> None:
    _run(tmp_path)
    with pytest.raises(ValueError, match="belongs to another run: param_values"):
        pu.parameterrun(flaky, ["x", "y"], [XS, [0, 1, 2, 3, 9]], n_workers=1, pbar_bool=False, checkpoint=tmp_path)


def test_other_fun_kwargs_raise(tmp_path: Path) -> None:
    def scaled(x: int, scale: float) -> float:
        return x * scale

    kwargs: dict[str, Any] = {"n_workers": 1, "pbar_bool": False, "backend": "joblib", "checkpoint": tmp_path}
    pu.parameterrun(scaled, "x", range(3), scale=1.0, **kwargs)
    with pytest.raises(ValueError, match="fun_kwargs"):
        pu.parameterrun(scaled, "x", range(3), scale=2.0, **kwargs)


def _make_scaled(factor: float) -> Callable[[int], float]:
    def scaled(x: int) -> float:
        return x * factor

    return scaled


def test_changed_code_raises_force_resumes_and_restart_recomputes(tmp_path: Path) -> None:
    kwargs: dict[str, Any] = {"n_workers": 1, "pbar_bool": False, "backend": "joblib", "checkpoint": tmp_path}
    pu.parameterrun(_make_scaled(1.0), "x", range(3), **kwargs)

    with pytest.raises(ValueError, match="code of scaled changed"):
        pu.parameterrun(_make_scaled(2.0), "x", range(3), **kwargs)

    with pytest.warns(RuntimeWarning, match="resuming anyway"):
        forced = pu.parameterrun(_make_scaled(2.0), "x", range(3), checkpoint_resume="force", **kwargs)
    np.testing.assert_array_equal(forced, [0.0, 1.0, 2.0])  # The saved points

    restarted = pu.parameterrun(_make_scaled(2.0), "x", range(3), checkpoint_resume="restart", **kwargs)
    np.testing.assert_array_equal(restarted, [0.0, 2.0, 4.0])


def test_damaged_chunk_raises_and_stray_chunks_are_ignored(tmp_path: Path) -> None:
    _run(tmp_path, checkpoint_every=10)
    (tmp_path / "chunk_000005.npz").write_bytes(b"left over by a crash")
    _run(tmp_path)  # Not in the manifest: ignored and removed
    assert not (tmp_path / "chunk_000005.npz").exists()

    (tmp_path / "chunk_000001.npz").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="missing or damaged"):
        _run(tmp_path)


def test_other_directory_contents_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"something": "else"}', encoding="utf-8")
    with pytest.raises(ValueError, match="is not a parameterrun checkpoint"):
        _run(tmp_path)
    afile = tmp_path / "file"
    afile.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        _run(afile)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"checkpoint_every": 0}, "checkpoint_every"),
        ({"checkpoint_every": 2.5}, "checkpoint_every"),
        ({"checkpoint_interval": -1}, "checkpoint_interval"),
        ({"checkpoint_resume": "maybe"}, "checkpoint_resume"),
    ],
)
def test_invalid_options_raise(tmp_path: Path, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, **kwargs)


def test_snapshot_chunks_hold_payloads(tmp_path: Path) -> None:
    _run(tmp_path, checkpoint_every=20)
    payloads = _read_chunk(tmp_path / "chunk_000000.npz")
    indices = np.concatenate([payload[0] for payload in payloads])
    np.testing.assert_array_equal(np.sort(indices), np.arange(20))
    with np.load(tmp_path / "chunk_000000.npz", allow_pickle=False) as data:
        assert any("_array_0" in key for key in data.files)  # Numbers: a plain array
        assert any("_pickle_1" in key for key in data.files)  # Strings: Python objects


def test_numeric_snapshots_contain_no_pickle(tmp_path: Path) -> None:
    def pair(x: int) -> tuple[float, np.ndarray]:
        if x == 3:
            raise ValueError("bad point")
        return float(x), np.arange(x, x + 2)

    kwargs: dict[str, Any] = {"n_workers": 1, "pbar_bool": False, "backend": "joblib", "on_error": "nan"}
    with pytest.warns(RuntimeWarning, match="1 of 6 points"):
        pu.parameterrun(pair, "x", range(6), checkpoint=tmp_path, **kwargs)
    with np.load(tmp_path / "chunk_000000.npz", allow_pickle=False) as data:
        assert not [key for key in data.files if "pickle" in key]
        assert list(data["0_failed_summary"]) == ["ValueError: bad point"]
    with pytest.warns(RuntimeWarning, match="1 of 6 points"):
        result = pu.parameterrun(pair, "x", range(6), checkpoint=tmp_path, **kwargs)
    assert isinstance(result, list)
    numbers, arrays = result
    assert np.isnan(numbers[3]) and np.isnan(arrays[3]).all()


class _Stop(Exception):
    pass


def test_sigterm_saves_then_terminates(tmp_path: Path) -> None:
    """SIGTERM: save at once, then run the previous handler (here one that raises, instead of killing pytest)."""

    def previous(signum: int, frame: Any) -> None:
        raise _Stop

    def terminated_at_five(x: int) -> int:
        if x == 5 and not (tmp_path / "resumed").exists():
            signal.raise_signal(signal.SIGTERM)
        return x

    kwargs: dict[str, Any] = {"n_workers": 1, "pbar_bool": False, "backend": "joblib", "checkpoint": tmp_path / "c"}
    original = signal.signal(signal.SIGTERM, previous)
    try:
        with pytest.raises(_Stop):
            pu.parameterrun(terminated_at_five, "x", range(10), checkpoint_interval=1e6, **kwargs)
        assert signal.getsignal(signal.SIGTERM) is previous  # Restored
        assert _manifest(tmp_path / "c")["n_done"] == 6  # Saved right after the point that got the signal
    finally:
        signal.signal(signal.SIGTERM, original)

    (tmp_path / "resumed").touch()
    np.testing.assert_array_equal(pu.parameterrun(terminated_at_five, "x", range(10), **kwargs), np.arange(10))


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is POSIX only")
def test_sigusr1_saves_and_goes_on(tmp_path: Path) -> None:
    def signal_at_five(x: int) -> int:
        if x == 5:
            signal.raise_signal(getattr(signal, "SIGUSR1"))  # noqa: B009 (absent on Windows)
        return x

    result = pu.parameterrun(
        signal_at_five, "x", range(10), n_workers=1, pbar_bool=False, checkpoint=tmp_path, checkpoint_interval=1e6
    )
    np.testing.assert_array_equal(result, np.arange(10))
    assert len(_manifest(tmp_path)["chunks"]) == 2  # One at the signal, one at the end


def test_save_signals_outside_the_main_thread_do_nothing() -> None:
    installed: list[int] = []

    def enter() -> None:
        with _SaveSignals() as signals:
            installed.extend(signals._previous)

    thread = threading.Thread(target=enter)
    thread.start()
    thread.join()
    assert installed == []


def test_ignored_sigterm_stays_ignored() -> None:
    original = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        with _SaveSignals():
            assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, original)


def test_interval_trigger(tmp_path: Path) -> None:
    checkpoint = _Checkpoint.open(tmp_path, flaky, [["x"]], [[[1, 2]]], {}, interval=1e-9)
    assert checkpoint.every is None
    checkpoint.add((np.array([0]), [np.array([1.0])], []))
    assert checkpoint.due()
    assert checkpoint.commit()
    assert not checkpoint.commit()  # Nothing new


def test_default_trigger_is_an_interval(tmp_path: Path) -> None:
    checkpoint = _Checkpoint.open(tmp_path, flaky, [["x"]], [[[1, 2]]], {})
    assert checkpoint.every is None and checkpoint.interval == 600.0


# ---------------------------------------------------------------- code fingerprint


def _module_function(source: str, name: str = "fun", module: str = "user_script") -> Callable[..., Any]:
    namespace = types.ModuleType(module)
    exec(compile(source, f"<{module}>", "exec"), namespace.__dict__)
    function = namespace.__dict__[name]
    assert callable(function)
    return function


def _same(a: Callable[..., Any], b: Callable[..., Any]) -> bool | None:
    return _code_matches(_code_fingerprint(a), _code_fingerprint(b))


def test_fingerprint_ignores_comments_and_line_numbers() -> None:
    a = _module_function("def fun(x):\n    return x + 1\n")
    b = _module_function("\n\n# A comment\ndef fun(x):\n    # Another one\n    return x + 1\n")
    assert _same(a, b)


@pytest.mark.parametrize(
    "changed",
    [
        "def fun(x):\n    return x + 2\n",  # Constant
        "def fun(x):\n    return x * 1\n",  # Operation
        "def fun(x, y=3):\n    return x + 1\n",  # Default value
        "def fun(x):\n    return abs(x) + 1\n",  # Called name
    ],
)
def test_fingerprint_detects_code_changes(changed: str) -> None:
    assert _same(_module_function("def fun(x):\n    return x + 1\n"), _module_function(changed)) is False


def test_fingerprint_follows_helpers_of_the_same_module() -> None:
    a = _module_function("def helper(x):\n    return x\n\ndef fun(x):\n    return helper(x)\n")
    b = _module_function("def helper(x):\n    return -x\n\ndef fun(x):\n    return helper(x)\n")
    assert _same(a, b) is False


def test_fingerprint_handles_recursion_partials_methods_and_builtins() -> None:
    import functools

    recursive = _module_function("def fun(n):\n    return 1 if n < 2 else n * fun(n - 1)\n")
    assert _same(recursive, recursive)

    def power(x: float, p: int) -> float:
        return float(x**p)

    assert _same(functools.partial(power, p=2), functools.partial(power, p=3)) is False

    class Model:
        def __init__(self, k: float) -> None:
            self.k = k

        def __call__(self, x: float) -> float:
            return self.k * x

        def run(self, x: float) -> float:
            return self.k * x

    assert _same(Model(1.0), Model(2.0)) is False
    assert _same(Model(1.0).run, Model(1.0).run)
    assert _same(np.sin, np.sin)
    assert _same(np.sin, np.cos) is False


def test_fingerprint_across_python_versions_uses_the_source() -> None:
    saved = {**_code_fingerprint(flaky), "python": "2.7", "bytecode": "other"}
    assert _code_matches(saved, _code_fingerprint(flaky))
    assert _code_matches({**saved, "source": None}, _code_fingerprint(flaky)) is None


def test_unknown_code_change_warns(tmp_path: Path) -> None:
    _run(tmp_path)
    manifest = _manifest(tmp_path)
    manifest["code"] = {**manifest["code"], "python": "2.7", "source": None}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="cannot be checked"):
        _run(tmp_path)


# ---------------------------------------------------------------- MPI


def _mpi_comm() -> Any:
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 4:
        pytest.skip("Run this test with mpirun -n 4")
    return comm


def _shared_temporary_directory(comm: Any) -> Path:
    import tempfile

    return Path(comm.bcast(tempfile.mkdtemp() if comm.Get_rank() == 0 else None, root=0))


def _mpi_product(x: int, y: int) -> tuple[float, str]:
    return float(x * y), f"{x}-{y}"


@pytest.mark.mpi
@pytest.mark.parametrize("schedule", ["static", "dynamic"])
def test_mpi_checkpoint_resumes_a_scattered_snapshot(schedule: Literal["static", "dynamic"]) -> None:
    """A snapshot holding every other point (not a prefix of any permutation) is completed by 4 ranks."""
    comm = _mpi_comm()
    rank = comm.Get_rank()
    directory = _shared_temporary_directory(comm)
    xs, ys = list(range(7)), list(range(5))

    if rank == 0:
        checkpoint = _Checkpoint.open(directory, _mpi_product, [["x"], ["y"]], [[xs], [ys]], {})
        saved = np.arange(0, 35, 2)
        checkpoint.add((saved, [np.full(len(saved), -1.0), [f"saved-{i}" for i in saved]], []))
        checkpoint.commit()
    comm.Barrier()

    result = pu.parameterrun(
        _mpi_product,
        ["x", "y"],
        [xs, ys],
        backend="mpi",
        pbar_bool=False,
        mpi_schedule=schedule,
        checkpoint=directory,
        checkpoint_every=3,
    )
    if rank != 0:
        assert result is None
        return

    assert isinstance(result, list)
    numbers, labels = (np.asarray(channel).reshape(-1) for channel in result)
    for i in range(35):
        x, y = divmod(i, 5)
        if i % 2 == 0:  # From the snapshot
            assert numbers[i] == -1 and labels[i] == f"saved-{i}"
        else:
            assert numbers[i] == x * y and labels[i] == f"{x}-{y}"
    assert _manifest(directory)["n_done"] == 35


@pytest.mark.mpi
@pytest.mark.parametrize("schedule", ["static", "dynamic"])
def test_mpi_checkpoint_full_run_and_rerun(schedule: Literal["static", "dynamic"]) -> None:
    comm = _mpi_comm()
    directory = _shared_temporary_directory(comm)
    kwargs: dict[str, Any] = {"backend": "mpi", "pbar_bool": False, "mpi_schedule": schedule, "checkpoint": directory}

    def value(x: int) -> np.ndarray:
        return np.array([x, x * x], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = pu.parameterrun(value, "x", range(50), checkpoint_every=7, return_on="all", **kwargs)
        second = pu.parameterrun(value, "x", range(50), return_on="all", **kwargs)
    expected = np.array([[x, x * x] for x in range(50)], dtype=float)
    np.testing.assert_array_equal(np.asarray(first), expected)
    np.testing.assert_array_equal(np.asarray(second), expected)
    if comm.Get_rank() == 0:
        assert len(_manifest(directory)["chunks"]) > 1


@pytest.mark.mpi
def test_mpi_checkpoint_mismatch_raises_on_every_rank() -> None:
    comm = _mpi_comm()
    directory = _shared_temporary_directory(comm)
    if comm.Get_rank() == 0:
        _Checkpoint.open(directory, _mpi_product, [["x"]], [[[1, 2, 3]]], {})
    comm.Barrier()
    with pytest.raises(ValueError, match="belongs to another run"):
        pu.parameterrun(_mpi_product, "x", [1, 2], y=1, backend="mpi", pbar_bool=False, checkpoint=directory)


_ABORTING_MPI_SCRIPT = textwrap.dedent(
    """
    import os
    import signal
    import sys
    import time
    from pathlib import Path

    import numpy as np
    from parameterrun import parameterrun

    directory, schedule, logs, how = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    marker = directory.parent / "fail"
    calls = []

    def f(x):
        calls.append(x)
        with open(logs / f"{os.getpid()}.txt", "a") as file:  # One file per process: appends never interleave
            file.write(f"{x} ")
        if len(calls) == 7 and marker.exists():  # After 3 rounds of 2 points per rank (static schedule)
            if how == "sigterm":  # As Slurm at the walltime limit: save, then end the job
                signal.raise_signal(signal.SIGTERM)
            else:
                raise ValueError("node failure")
        time.sleep(0.005)
        return float(x * x)

    result = parameterrun(f, "x", range(40), backend="mpi", pbar_bool=False, mpi_schedule=schedule,
                          checkpoint=directory, checkpoint_every=8)
    if result is not None:
        assert np.array_equal(result, np.arange(40.0) ** 2), result
        print("RESUMED OK", flush=True)
    """
)


@pytest.mark.mpi_subprocess
@pytest.mark.parametrize("how", ["error", "sigterm"])
@pytest.mark.parametrize("schedule", ["static", "dynamic"])
def test_mpi_job_aborted_resumes_from_its_snapshot(tmp_path: Path, schedule: str, how: str) -> None:
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None or find_spec("mpi4py") is None:
        pytest.skip("Needs mpiexec on PATH and mpi4py")
    if _mpi_world_size_from_env() > 1:
        pytest.skip("Already inside an MPI job: don't nest mpiexec")

    script = tmp_path / "sweep.py"
    script.write_text(_ABORTING_MPI_SCRIPT, encoding="utf-8")
    directory, first_logs, second_logs = tmp_path / "checkpoint", tmp_path / "logs1", tmp_path / "logs2"
    first_logs.mkdir()
    second_logs.mkdir()
    (tmp_path / "fail").touch()
    src_dir = str(Path(pu.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src_dir, os.environ.get("PYTHONPATH")]))}

    def run(logs: Path) -> subprocess.CompletedProcess[str]:
        command = [mpiexec, "-n", "4", sys.executable, str(script), str(directory), schedule, str(logs), how]
        return subprocess.run(command, capture_output=True, text=True, timeout=120, env=env, check=False)

    first = run(first_logs)
    assert first.returncode != 0, first.stdout + first.stderr
    if how == "error":
        assert "ValueError: node failure" in first.stdout + first.stderr
    elif schedule == "dynamic":
        assert "terminated by SIGTERM after saving the checkpoint" in first.stdout + first.stderr
    n_saved = _manifest(directory)["n_done"]
    assert 0 < n_saved < 40

    (tmp_path / "fail").unlink()
    second = run(second_logs)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "RESUMED OK" in second.stdout
    computed = [int(x) for log in second_logs.iterdir() for x in log.read_text().split()]
    assert len(computed) == 40 - n_saved  # Only the missing points were computed
    assert len(set(computed)) == len(computed)
