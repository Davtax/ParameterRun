# Checkpointing / snapshots for long runs — design notes

Status: **proposal**, nothing implemented yet.

## Goal

For very large parameter sweeps, periodically save progress so that a failed run (crash, node failure, walltime
limit) can be restarted from the last snapshot instead of from scratch.

Proposed workflow:

1. The user selects the batch size (how often to save) and optionally the file name.
2. After the desired number of calculations, the main process gathers all results.
3. Save or update the snapshot with:
   - the current position in the global run,
   - the current results.
4. Resume the calculation.
5. If the calculation completes successfully, optionally delete the snapshot.
6. `parameterrun` must be able to restart a failed calculation from the last snapshot.

## Key insight: a common model for both backends

Every evaluation already has a flat index `k ∈ [0, n_total)`; `_product_element_from_index` maps it back to a parameter
combination. A checkpointed run is therefore an outer loop: *process a chunk of flat indices → gather → save →
continue*. The snapshot only needs `{completed prefix/indices, results for them, run fingerprint}`. The format and
validation logic can be shared by both backends.

## joblib backend (the easier one)

`pyproject.toml` requires `joblib>=1.3`, which provides `Parallel(..., return_as="generator")`. It yields results **in
order** as they become available, while the worker pool keeps working ahead:

```python
gen = Parallel(n_jobs=n_workers, return_as="generator")(
    delayed(fun)(**{**_get_iteration(index, ...), **fun_kwargs}) for index in remaining_indices
)
for i, r in enumerate(gen, start=n_done):
    results.append(r)
    if (i + 1) % checkpoint_every == 0:
        save_snapshot(...)
```

- No barrier: workers do not wait at chunk boundaries. Calling `Parallel` once per chunk would cause idle workers at
  the end of every chunk while the slowest task finishes.
- Because results come in order, "position in the run" is a single integer `n_done`. Resuming means starting the index
  iterator at `n_done` (e.g. `itertools.islice(product(*indices), n_done, None)`).
- If `fun` raises, everything consumed so far can be saved in a `finally` block; only in-flight work is lost.
- The `n_workers == 1` path is a plain loop, so it is trivial.
- The `tqdm_joblib` monkeypatch counts completed *batches*, not consumed results. With the generator, the patch can be
  dropped and the progress bar updated in the consumer loop (simpler and more accurate).

## MPI backend (harder than it looks)

Today each rank runs all of its work and then does a single `gather`. Checkpointing turns this into **rounds**:

1. **Each round is a synchronisation barrier.** Fast ranks wait for the slowest at every snapshot. The random shuffle
   reduces imbalance, but small batch sizes will waste time.
2. **Prefix property.** In `local_values`, rank `r`'s `k`-th item is permutation position `r + k·size`. If every rank
   finishes the same number of items per round, the completed set is a **contiguous prefix of the global permutation**.
   The position is then fully described by `(seed, n_done)`, and a run can be **resumed with a different number of
   ranks**: new ranks share out positions `n_done … n_total`. This needs an `offset` argument in `local_values`.
3. **The seed must be persisted.** Note: `parameterrun()` currently never passes `seed` to `_parameterrun_mpi`, so every
   run draws a random seed. A resumed run must reuse the seed stored in the snapshot.
4. **Rank failures.** If a rank crashes, the others hang in the next collective, or `MPI_Abort` kills the job. The last
   snapshot survives (the point of the feature), but a failure inside `fun` cannot be saved "gracefully" as in joblib.
5. **HPC walltime.** Runs typically die because SLURM kills the job at its time limit. A time-based trigger ("save
   every 30 min") fits better than a count. Rank 0 must take the decision and `bcast` it so all ranks stay in step.
   Optionally, catch `SIGTERM` (SLURM sends it before `SIGKILL`) to trigger a last save.
6. **Alternative: per-rank snapshot files** (no gather, no barrier). Cheaper at runtime, but the completed set is no
   longer a prefix: explicit index sets must be stored, and resuming with a different `size` becomes awkward. Not
   recommended for a first version.

## Problems common to both backends

- **Snapshot validation (biggest risk).** Resuming a snapshot that belongs to a different run gives silently wrong
  results. Store a fingerprint:
  - `fun` name,
  - `param_names`,
  - `joblib.hash(param_values)`,
  - `fun_kwargs` (hash),
  - `n_total`, `n_outputs`,
  - the seed (MPI),
  - a snapshot format version.

  Changes to the function's *code* cannot be detected reliably (`inspect.getsource` fails for lambdas, C functions,
  `functools.partial`, …), so this must be documented.
- **Atomic writes.** A job killed during a write would corrupt the only snapshot. Write to `file.tmp`, then
  `os.replace` it over the old file; optionally keep the previous snapshot as `.bak`.
- **Quadratic I/O.** Rewriting *all* results at every save costs O(N²/batch) for very large runs. An append-only layout
  (directory with one file per chunk plus a small manifest) avoids it.
- **Serialisation.** Results are arbitrary Python objects (tuples, arrays, …), so it realistically has to be pickle /
  `joblib.dump`. Pickle is not portable across library versions and is unsafe to load from untrusted sources. `.npz`
  only works if every output is a numeric array.
- **Memory.** Rank 0 already holds every result in memory. Snapshots do not change that, but per-chunk files on disk
  would allow a later version to avoid holding everything in RAM.
- **Edge cases.**
  - `n_outputs` is only known after the first evaluation, so store it in the snapshot.
  - The progress bar needs `initial=n_done` when resuming.
  - MPI results must be stored with their original indices so the final sort and reshape still work.
  - Snapshot is complete but the final reshape fails: the snapshot must still be usable.

## Decisions to make

| Decision | Options | Recommendation |
|---|---|---|
| What "batch" means | evaluations globally / per worker / wall-clock time | Support both `checkpoint_every=<int>` (global evaluations) and `checkpoint_interval=<seconds>`; time-based is what HPC users need |
| File layout | single file rewritten / directory with chunks + manifest | Directory: append-only, atomic per chunk, scales |
| Default file name | user-given only / derived from the run's hash | User-given, plus an automatic name from the fingerprint hash for convenience |
| How to resume | explicit `resume=True` / automatic if the file exists | Automatic when the fingerprint matches; **raise an error on mismatch** (never silently overwrite or ignore) |
| Deleting after success | default delete / default keep | Default **keep**: the snapshot *is* the result if the script crashes after `parameterrun` returns (e.g. while saving plots) |
| MPI design | synchronised rounds / per-rank files | Synchronised rounds, using the prefix-of-the-permutation property |
| Resuming with a different `size` | allow / forbid | Allow; it comes for free with the prefix design |
| Exceptions in `fun` | save completed results then re-raise / ignore | joblib: save then re-raise. MPI: rely on the last snapshot |
| Seed | keep internal / expose `seed=` in `parameterrun` | Expose it; it is needed for reproducibility anyway |

## Possible API sketch

```python
result = parameterrun(
    fun,
    param_names,
    param_values,
    checkpoint_every=10_000,          # global evaluations between snapshots (optional)
    checkpoint_interval=1800,         # seconds between snapshots (optional)
    checkpoint_path="run_snapshot",   # file/directory; None -> auto name from fingerprint
    keep_checkpoint=True,             # delete the snapshot after success if False
    seed=None,                        # MPI shuffle seed; stored in / restored from the snapshot
)
```

## Suggested implementation order

1. Backend-independent snapshot module: fingerprint, atomic save/load, format version, validation.
2. joblib backend with `return_as="generator"` (plus the `n_workers == 1` loop), progress bar resume.
3. Expose `seed` in `parameterrun` and add an `offset` to `local_values`.
4. MPI backend in synchronised rounds, with rank 0 deciding when to save (count or time) and broadcasting it.
5. Tests: interrupt-and-resume for each backend, fingerprint mismatch, corrupted/partial snapshot, resume with a
   different number of MPI ranks.
