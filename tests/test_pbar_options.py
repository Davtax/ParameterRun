"""Progress-bar defaults and non-terminal output (phase 1.4)."""

import io
import sys
import types
from typing import Any

import pytest

import parameterrun._progress as progress
import parameterrun.parallel_utils as pu


class _FakeTerminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _clean_tqdm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    monkeypatch.delenv("TQDM_MININTERVAL", raising=False)
    monkeypatch.delitem(sys.modules, "IPython", raising=False)


def _fake_ipython(monkeypatch: pytest.MonkeyPatch, shell_class_name: str) -> None:
    shell = type(shell_class_name, (), {})()
    monkeypatch.setitem(sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: shell))


# ---------------------------------------------------------------- terminal / notebook detection


def test_stream_is_terminal() -> None:
    assert progress._stream_is_terminal(_FakeTerminal())
    assert not progress._stream_is_terminal(io.StringIO())
    assert not progress._stream_is_terminal(object())  # No isatty at all

    closed = io.StringIO()
    closed.close()
    assert not progress._stream_is_terminal(closed)  # isatty raises ValueError


def test_in_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not progress._in_notebook()

    _fake_ipython(monkeypatch, "TerminalInteractiveShell")
    assert not progress._in_notebook()

    _fake_ipython(monkeypatch, "ZMQInteractiveShell")
    assert progress._in_notebook()


# ---------------------------------------------------------------- default of pbar_bool


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_pbar_bool_wins(monkeypatch: pytest.MonkeyPatch, explicit: bool) -> None:
    monkeypatch.setenv("TQDM_DISABLE", "0" if not explicit else "1")
    assert progress._resolve_pbar_bool(explicit, io.StringIO()) is explicit
    assert progress._resolve_pbar_bool(explicit, _FakeTerminal()) is explicit


def test_default_pbar_follows_terminal() -> None:
    assert progress._resolve_pbar_bool(None, _FakeTerminal()) is True
    assert progress._resolve_pbar_bool(None, io.StringIO()) is False


def test_default_pbar_on_in_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ipython(monkeypatch, "ZMQInteractiveShell")
    assert progress._resolve_pbar_bool(None, io.StringIO()) is True


@pytest.mark.parametrize(
    ("value", "expected"), [("1", False), ("true", False), ("ON", False), ("0", True), ("false", True), ("", True)]
)
def test_default_pbar_honours_tqdm_disable(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("TQDM_DISABLE", value)
    assert progress._resolve_pbar_bool(None, _FakeTerminal()) is expected
    assert progress._resolve_pbar_bool(None, io.StringIO()) is expected


# ---------------------------------------------------------------- line-based output


def test_line_stream_for_logs_writes_plain_lines() -> None:
    target = io.StringIO()
    stream = progress._LineStream(target, terminal=False)

    assert stream.write("\rdesc:  33%|###  | 1/3") == len("\rdesc:  33%|###  | 1/3")
    stream.write("\r          \r")  # tqdm clearing the line: nothing to log
    stream.write("\x1b[A")
    stream.write("\rdesc: 100%|#####| 3/3\n")
    stream.flush()

    assert target.getvalue() == "desc:  33%|###  | 1/3\ndesc: 100%|#####| 3/3\n"


def test_line_stream_for_forwarded_terminal_ends_each_refresh_with_newline() -> None:
    target = io.StringIO()
    stream = progress._LineStream(target, terminal=True)

    stream.write("\rdesc: 1/3")
    stream.write("\r     \r")

    # Newline so the MPI launcher forwards it, then cursor up so the next refresh overwrites it
    assert target.getvalue() == "\rdesc: 1/3\n\x1b[F"


def test_line_stream_exposes_encoding() -> None:
    target = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    assert progress._LineStream(target, terminal=False).encoding == "utf-8"


def test_pbar_options_disabled() -> None:
    target = io.StringIO()
    assert progress._pbar_options(False, target) == {"disable": True, "file": target}


def test_pbar_options_terminal() -> None:
    target = _FakeTerminal()
    assert progress._pbar_options(True, target) == {"disable": False, "file": target}

    forwarded = progress._pbar_options(True, target, forwarded=True)
    assert isinstance(forwarded["file"], progress._LineStream)
    assert forwarded["file"]._terminal
    assert "mininterval" not in forwarded


def test_pbar_options_not_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    options = progress._pbar_options(True, io.StringIO())
    assert isinstance(options["file"], progress._LineStream)
    assert not options["file"]._terminal
    assert options["mininterval"] == progress._LOG_MININTERVAL

    monkeypatch.setenv("TQDM_MININTERVAL", "5")  # tqdm reads it itself, as long as we do not pass mininterval
    assert "mininterval" not in progress._pbar_options(True, io.StringIO())


def test_pbar_options_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ipython(monkeypatch, "ZMQInteractiveShell")
    target = io.StringIO()
    assert progress._pbar_options(True, target) == {"disable": False, "file": target}


# ---------------------------------------------------------------- through parameterrun


def _run_capturing_stderr(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> str:
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured)
    result = pu.parameterrun(lambda x: x, "x", [1, 2, 3], n_workers=1, backend="joblib", **kwargs)
    assert list(result) == [1, 2, 3]  # type: ignore[arg-type]
    return captured.getvalue()


def test_parameterrun_default_hides_bar_when_not_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_capturing_stderr(monkeypatch) == ""


def test_parameterrun_default_shows_bar_on_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _FakeTerminal()
    monkeypatch.setattr(sys, "stderr", captured)
    pu.parameterrun(lambda x: x, "x", [1, 2, 3], n_workers=1, backend="joblib")
    assert "3/3" in captured.getvalue()
    assert "\r" in captured.getvalue()  # Normal redraws on a terminal


def test_parameterrun_forced_bar_writes_plain_lines_when_not_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _run_capturing_stderr(monkeypatch, pbar_bool=True, desc="sweep")
    assert "\r" not in output
    lines = output.splitlines()
    assert lines, "expected at least one progress line"
    assert all(line.startswith("sweep") for line in lines)
    assert "3/3" in lines[-1]
