import numpy as np


def _mix64(x: int) -> int:
    """SplitMix-like integer mixer used as Feistel round function."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    return x ^ (x >> 31)


def _feistel_pow2(x: int, bits: int, key: int, rounds: int = 4) -> int:
    """Permutation on [0, 2^bits) with an even bit-width Feistel network."""
    half_bits = bits // 2
    mask = (1 << half_bits) - 1

    left = (x >> half_bits) & mask
    right = x & mask

    for rnd in range(rounds):
        k = (int(key) + 0x9E3779B97F4A7C15 * (rnd + 1)) & 0xFFFFFFFFFFFFFFFF
        f = _mix64(right ^ k) & mask
        left, right = right, (left ^ f) & mask

    return (left << half_bits) | right


def _permute_in_range(x: int, total: int, key: int) -> int:
    """Cycle-walk a Feistel permutation down to [0, total)."""
    if total <= 1:
        return 0

    bits = max(1, (total - 1).bit_length())
    if bits % 2 == 1:
        bits += 1

    y = int(x)
    while True:
        y = _feistel_pow2(y, bits=bits, key=key)
        if y < total:
            return y


def local_values(total: int, workers: int, worker_rank: int, base_seed: int):
    """Yield this rank's values using interleaved positions in a global permutation."""
    if total <= 0:
        return

    key = int(base_seed)
    local_count = total // workers + (1 if worker_rank < (total % workers) else 0)

    # Stream in small batches to avoid materializing large arrays.
    for k in range(local_count):
        value = np.int64(worker_rank) + k * np.int64(workers)
        yield _permute_in_range(int(value), total=total, key=key)

