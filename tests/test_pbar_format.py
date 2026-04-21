"""Test progress bar format for joblib backend."""
import re

import numpy as np

from parameterrun import parameterrun


def capture_pbar_output(n_workers: int, n_x: int = 10, n_y: int = 50) -> str:
    """
    Run parameterrun and capture progress bar output.

    Parameters
    ----------
    n_workers : int
        Number of workers to use
    n_x : int
        Number of x values
    n_y : int
        Number of y values

    Returns
    -------
    str
        Captured progress bar output
    """

    def dummy_func(x, y):
        """Simple dummy function with two grouped parameters."""
        return x + y

    xs = np.linspace(0, 10, n_x)
    ys = np.linspace(0, 20, n_y)

    # Use StringIO to capture stderr (where tqdm writes progress bars)
    import sys
    from io import StringIO
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        parameterrun(dummy_func, param_names=[["x"], ["y"]], param_values=[[xs], [ys]], n_workers=n_workers,
                     pbar_bool=True, backend="joblib", pbar_leave=True,  # Keep the progress bar output for inspection
                     )
        output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    return output


def test_pbar_format_single_worker():
    """Test progress bar format with n_workers=1."""
    output = capture_pbar_output(n_workers=1, n_x=10, n_y=50)

    # The format should contain:
    # - Total count with progress: X/500 (e.g., "350/500" meaning 350/500 iterations)
    # - Elapsed time: [MM:SS or HH:MM:SS]
    # - Remaining time: <MM:SS or <HH:MM:SS
    # - Rate: XX.XXit/s or YY.YYit/s

    # Expected format: f: [['x'], ['y']]:   0%|          | 350/500 [00:17<7:05:24, 19.58it/s]

    # Check for presence of total count pattern (e.g., "500/500" at completion)
    print(output)
    assert re.search(r'\d+/500', output), f"Progress bar missing total count pattern. Output: {output}"

    # Check for remaining time pattern (e.g., "<MM:SS" or "<HH:MM:SS")
    assert re.search(r'<\d+:\d{2}:\d{2}|<\d+:\d{2}',
                     output), f"Progress bar missing remaining time pattern. Output: {output}"

    # Check for rate pattern (XX.XXit/s or YY.YYit/s)
    assert re.search(r'\d+\.\d+it/s|\d+\.\d+s/it', output), f"Progress bar missing rate pattern. Output: {output}"


def test_pbar_format_multiple_workers():
    """Test progress bar format with n_workers=2."""
    output = capture_pbar_output(n_workers=2, n_x=10, n_y=50)

    # The format should be the same for multi-worker case
    # Expected format: f: [['x'], ['y']]:   0%|          | 350/500 [00:17<7:05:24, 19.58it/s]

    # Check for presence of total count pattern (e.g., "500/500" at completion)
    assert re.search(r'\d+/500', output), f"Progress bar missing total count pattern. Output: {output}"

    # Check for remaining time pattern (e.g., "<MM:SS" or "<HH:MM:SS")
    assert re.search(r'<\d+:\d{2}:\d{2}|<\d+:\d{2}',
                     output), f"Progress bar missing remaining time pattern. Output: {output}"

    # Check for rate pattern (XX.XXit/s or YY.YYit/s)
    assert re.search(r'\d+\.\d+it/s|\d+\.\d+s/it', output), f"Progress bar missing rate pattern. Output: {output}"


def test_pbar_does_not_have_incorrect_format():
    """Test that progress bar does NOT use the incorrect format."""
    output = capture_pbar_output(n_workers=1, n_x=5, n_y=10)

    # Incorrect format would look like: "35it [00:03,  9.75it/s]"
    # which has no total count or remaining time

    # Check that we DON'T have the incorrect pattern
    # The incorrect pattern is: digit(s) + "it [" with no "/" before
    incorrect_pattern = r'(?<![/\d])\d+it \['

    assert not re.search(incorrect_pattern,
                         output), f"Progress bar has incorrect format (missing total count). Output: {output}"
