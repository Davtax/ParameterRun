from parameterrun.parallel_utils import _normalized_time


def test_normalized_time_seconds():
    assert _normalized_time(12.3) == "12.30 s"


def test_normalized_time_minutes():
    assert _normalized_time(125.0) == "2 min 5.00 s"


def test_normalized_time_hours():
    assert _normalized_time(3723.0) == "1 h 2 min 3.00 s"
