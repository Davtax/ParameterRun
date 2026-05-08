import pytest

from parameterrun._pseudo_random import _permute_in_range, local_values


def _collect_assignments(total: int, workers: int, seed: int) -> list[list[int]]:
    return [list(local_values(total, workers, rank, seed)) for rank in range(workers)]


@pytest.mark.parametrize(
    ("total", "workers", "seed"),
    [
        (1, 1, 11),
        (20, 4, 123456789),
        (101, 8, 987654321),
        (7, 10, 555),
    ],
)
def test_local_values_form_exact_cover_without_overlap(total: int, workers: int, seed: int):
    assignments = _collect_assignments(total, workers, seed)
    merged = [v for rank_values in assignments for v in rank_values]

    assert len(merged) == total
    assert len(set(merged)) == total
    assert set(merged) == set(range(total))


@pytest.mark.parametrize(("total", "workers"), [(20, 4), (21, 4), (7, 10)])
def test_local_values_are_balanced(total: int, workers: int):
    assignments = _collect_assignments(total, workers, seed=42)
    counts = [len(v) for v in assignments]

    assert sum(counts) == total
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize(("total", "workers", "seed"), [(100, 6, 123), (17, 5, 999)])
def test_local_values_are_deterministic_for_fixed_seed(total: int, workers: int, seed: int):
    first = _collect_assignments(total, workers, seed)
    second = _collect_assignments(total, workers, seed)
    assert first == second


def test_local_values_change_for_different_seeds():
    # Large enough case to make accidental equality effectively impossible.
    first = _collect_assignments(total=128, workers=8, seed=12345)
    second = _collect_assignments(total=128, workers=8, seed=54321)
    assert first != second


@pytest.mark.parametrize(("total", "workers"), [(0, 4), (-5, 4)])
def test_local_values_empty_for_non_positive_total(total: int, workers: int):
    assignments = _collect_assignments(total, workers, seed=1)
    assert assignments == [[] for _ in range(workers)]


@pytest.mark.parametrize(("total", "seed"), [(2, 1), (5, 12), (64, 77), (100, 98765)])
def test_permute_in_range_is_bijection_on_domain(total: int, seed: int):
    permuted = [_permute_in_range(x, total=total, key=seed) for x in range(total)]

    assert all(0 <= x < total for x in permuted)
    assert len(set(permuted)) == total

