import numpy as np
from mpi4py import MPI

from parameterrun._pseudo_random import local_values

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# rank = 0
# size = 4

n_total = 200

# New seed every run, but identical on all ranks for this run
seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]) if rank == 0 else None
seed = comm.bcast(seed, root=0)


def evaluate_randomness(n_total, n_workers, local_values_fn, trials=5000):
    # counts[r, v] = how often value v is assigned to rank r
    counts = np.zeros((n_workers, n_total), dtype=np.int64)

    for _ in range(trials):
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])

        all_vals = []
        for r in range(n_workers):
            vals = list(local_values_fn(n_total, n_workers, r, seed))
            all_vals.append(vals)
            for v in vals:
                counts[r, v] += 1

        # correctness checks
        merged = [v for vals in all_vals for v in vals]
        assert len(merged) == n_total
        assert len(set(merged)) == n_total
        assert min(merged) >= 0 and max(merged) < n_total

    # expected probability that a value lands on each rank
    local_counts = np.array([n_total // n_workers + (1 if r < (n_total % n_workers) else 0) for r in range(n_workers)],
                            dtype=float, )
    p_rank = local_counts / n_total
    expected = trials * p_rank[:, None]  # shape (workers, 1), broadcast to values

    # z-score per (rank, value): should be mostly within about [-3, 3]
    var = expected * (1 - p_rank[:, None])
    z = (counts - expected) / np.sqrt(np.maximum(var, 1e-12))
    max_abs_z = np.max(np.abs(z))

    print("max |z| across (rank,value):", float(max_abs_z))
    print("mean |z|:", float(np.mean(np.abs(z))))

    # quick pass/fail heuristic
    if max_abs_z < 4.0:
        print("Looks statistically reasonable for this sample size.")
    else:
        print("Potential structure/bias detected; increase trials or use stronger permutation.")


count = n_total // size + (1 if rank < (n_total % size) else 0)

# Print in rank order for easier reading.
for r in range(size):
    comm.Barrier()
    if r == rank:
        print(f"rank={rank} count={count} values=", end="")
        for value in local_values(n_total, size, rank, seed):
            print(value, end=" ")
        print()
comm.Barrier()


def verify_exact_cover(total: int, worker_rank: int, values_iter):
    """Check that global outputs contain each number in [0, total) exactly once."""
    local_vals = np.fromiter(values_iter, dtype=np.int64)

    # Gather variable-length arrays from all ranks to rank 0
    gathered = comm.gather(local_vals, root=0)

    if worker_rank == 0:
        all_vals = np.concatenate(gathered) if gathered else np.array([], dtype=np.int64)

        ok_len = all_vals.size == total
        ok_range = np.all((0 <= all_vals) & (all_vals < total))
        uniq, counts = np.unique(all_vals, return_counts=True)
        ok_unique = uniq.size == total and np.all(counts == 1)

        if ok_len and ok_range and ok_unique:
            print(f"[OK] Exact cover: every value in [0, {total - 1}] appears once.")
        else:
            print("[FAIL] Exact cover check failed.")
            print(f"  total produced: {all_vals.size}, expected: {total}")
            if not ok_range:
                bad = all_vals[(all_vals < 0) | (all_vals >= total)]
                print(f"  out-of-range values: {bad[:10]}{' ...' if bad.size > 10 else ''}")
            if not ok_unique:
                missing = np.setdiff1d(np.arange(total, dtype=np.int64), uniq)
                repeated = uniq[counts > 1]
                print(f"  missing: {missing[:10]}{' ...' if missing.size > 10 else ''}")
                print(f"  repeated: {repeated[:10]}{' ...' if repeated.size > 10 else ''}")


verify_exact_cover(total=n_total, worker_rank=rank, values_iter=local_values(n_total, size, rank, seed), )
