"""Progress bars: when to show them, and how to write them to terminals, logs and MPI launchers."""

import contextlib
import os
import sys
from collections.abc import Generator
from typing import Any, TextIO

import joblib
from tqdm.auto import tqdm

# Seconds between progress-bar lines when the output is not a terminal (log files, pipes). Overridden by the
# TQDM_MININTERVAL environment variable.
_LOG_MININTERVAL = 30.0


def _stream_is_terminal(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # No isatty, or the stream is closed
        return False


def _in_notebook() -> bool:
    ipython = sys.modules.get("IPython")
    get_ipython = getattr(ipython, "get_ipython", None)
    if get_ipython is None:
        return False
    return type(get_ipython()).__name__ == "ZMQInteractiveShell"


def _resolve_pbar_bool(pbar_bool: bool | None, stream: Any) -> bool:
    """
    Explicit values win. Otherwise honour TQDM_DISABLE, and show the bar only on a terminal or in a notebook.
    """
    if pbar_bool is not None:
        return bool(pbar_bool)

    env_disable = os.environ.get("TQDM_DISABLE")
    if env_disable is not None:
        return env_disable.strip().lower() not in {"1", "true", "yes", "on"}

    return _stream_is_terminal(stream) or _in_notebook()


class _LineStream:
    """
    File-like wrapper that makes every progress-bar refresh visible as soon as it is written.

    * ``terminal=False`` (log files, pipes): each refresh becomes one plain line, without carriage returns.
    * ``terminal=True`` (MPI rank whose stdout is a terminal): MPI launchers forward output line by line, so a bar
      redrawn with ``\\r`` only would never show up. End each refresh with a newline and move the cursor back up.
    """

    def __init__(self, stream: TextIO, terminal: bool) -> None:
        self._stream = stream
        self._terminal = terminal

    @property
    def encoding(self) -> str | None:
        return getattr(self._stream, "encoding", None)

    def write(self, text: str) -> int:
        content = text.replace("\r", "").replace("\x1b[A", "").strip("\n")
        if content.strip():
            if self._terminal:
                self._stream.write(text.rstrip("\n") + "\n\x1b[F")
            else:
                self._stream.write(content + "\n")
        return len(text)

    def flush(self) -> None:
        self._stream.flush()


def _pbar_options(pbar_bool: bool, stream: TextIO, forwarded: bool = False) -> dict[str, Any]:
    """
    Keyword arguments for tqdm. ``forwarded`` is True when the output goes through an MPI launcher.
    """
    if not pbar_bool:
        return {"disable": True, "file": stream}

    if _in_notebook():
        return {"disable": False, "file": stream}

    if _stream_is_terminal(stream):
        file: Any = _LineStream(stream, terminal=True) if forwarded else stream
        return {"disable": False, "file": file}

    options: dict[str, Any] = {"disable": False, "file": _LineStream(stream, terminal=False)}
    if "TQDM_MININTERVAL" not in os.environ:
        options["mininterval"] = _LOG_MININTERVAL
    return options


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> Generator[tqdm, None, None]:
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback  # ty: ignore[invalid-assignment]
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
