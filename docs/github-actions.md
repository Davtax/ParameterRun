# GitHub Actions

> **Keep this in sync:** this page describes [.github/workflows/](../.github/workflows/). When a workflow changes
> (triggers, jobs, matrix, steps, conditions), update the chart and the tables below in the same change.

The repository has three workflows. They form one pipeline, from a pull request to a PyPI release:

| Workflow | File | Trigger | What it does |
| --- | --- | --- | --- |
| CI | [ci.yml](../.github/workflows/ci.yml) | `pull_request` into `main` | Tests, lint and type check on 15 OS/Python combinations, plus MPI tests on Linux, macOS and Windows |
| Release | [release.yml](../.github/workflows/release.yml) | `push` to `main` (i.e. a merged PR) | Reads `__version__`, checks that the tag is new and that `CHANGELOG.md` has a matching entry, then creates the GitHub release and tag `vX.Y.Z` |
| Publish | [publish.yml](../.github/workflows/publish.yml) | `workflow_run`: Release completed | If Release succeeded: builds the sdist and wheel, checks them, and uploads them to PyPI |

## Pipeline chart

- **Solid arrow:** confirmed in the workflow files.
- **Dotted arrow:** inferred. It depends on repository settings on GitHub (e.g. branch protection), which are not
  stored in the repository.

```mermaid
flowchart TD
    work(["Work on dev / feature branches"]) -->|"open a PR into main"| pr

    subgraph CI["ci.yml: on pull_request into main"]
        pr{{"pull_request event"}}
        matrix["job: test (15 runs, fail-fast off)<br/>os: ubuntu, macos, windows<br/>python: 3.11, 3.12, 3.13, 3.14, 3.15"]
        msteps["checkout, setup-python, pip install uv, uv sync<br/>pytest -m 'not mpi and not mpi_subprocess' (coverage gate 70%)<br/>ruff check, mypy"]
        mpi["job: test-mpi (7 runs, fail-fast off)<br/>Python 3.12<br/>ubuntu: Open MPI + MPICH + Intel MPI<br/>macos: Open MPI + MPICH<br/>windows: Microsoft MPI + Intel MPI"]
        mpisteps["uv sync --extra mpi, verify native MPI runtime<br/>full MPI suite: 4 ranks<br/>scheduling and idle-worker tests: 8 ranks<br/>pytest -m mpi_subprocess (starts its own mpiexec)"]
        pr --> matrix --> msteps
        pr --> mpi --> mpisteps
    end

    msteps & mpisteps -->|"report"| checks
    checks["Status checks on the PR"]
    merge(["Merge the PR into main"])
    checks -.->|"block the merge only if branch protection requires them"| merge

    merge -->|"push to main"| ver

    subgraph REL["release.yml: on push to main (one run at a time)"]
        ver["Read __version__ from<br/>src/parameterrun/__init__.py, tag = vX.Y.Z"]
        tagq{"tag vX.Y.Z already<br/>on origin?"}
        notes["Extract the '## [X.Y.Z]' section<br/>of CHANGELOG.md"]
        nq{"section found?"}
        gh["gh release create vX.Y.Z<br/>(creates the tag at the merged commit)"]
        fail1(["Release fails:<br/>bump __version__"])
        fail2(["Release fails:<br/>add the CHANGELOG entry"])
        ver --> tagq
        tagq -->|"yes"| fail1
        tagq -->|"no"| notes --> nq
        nq -->|"no"| fail2
        nq -->|"yes"| gh
    end

    gh -->|"Release completed"| okq
    fail1 & fail2 -->|"Release completed"| okq

    subgraph PUB["publish.yml: on workflow_run of Release"]
        okq{"Release<br/>succeeded?"}
        build["checkout the released commit (head_sha)<br/>Python 3.11, uv sync<br/>python -m build, twine check --strict"]
        pypi["pypa/gh-action-pypi-publish<br/>environment 'pypi', OIDC trusted publishing"]
        skip(["publish job skipped"])
        okq -->|"yes"| build --> pypi
        okq -->|"no"| skip
    end
```

## Jobs in detail

### CI ([ci.yml](../.github/workflows/ci.yml))

| Job | Runs on | Steps | Notes |
| --- | --- | --- | --- |
| `test` | ubuntu, macos, windows × Python 3.11–3.15 (15 runs) | `uv sync`, `pytest -m "not mpi and not mpi_subprocess"`, `ruff check .`, `mypy` | `fail-fast: false`, so one failing combination does not cancel the others. `addopts` in `pyproject.toml` adds the 70% coverage gate. Native MPI is only required in `test-mpi`. |
| `test-mpi` | Python 3.12; ubuntu with Open MPI/MPICH/Intel MPI, macos with Open MPI/MPICH, windows with Microsoft MPI/Intel MPI (7 runs) | `mpi4py/setup-mpi`, `uv sync --extra mpi`, verify the runtime vendor, full MPI suite with 4 ranks, `tests/test_mpi_rank_scaling.py` with 8 ranks, then `pytest -m mpi_subprocess` | `fail-fast: false`. Existing integration tests require 4 ranks; the scaling tests run at both 4 and 8 ranks. Subprocess tests launch their own four-rank jobs outside `mpiexec`. |

Permissions are read-only (`contents: read`).

The MPI job starts `uv` once before the launcher so all ranks use the same prepared Python environment. The runtime
check fails immediately if the native MPI library cannot be loaded or its vendor differs from the selected matrix
entry, preventing a preinstalled MPI library from silently replacing the intended runtime. Signal tests use
platform-specific skips where needed: `SIGUSR1` coverage is skipped on Windows, while the existing
`signal.raise_signal(SIGTERM)` checkpoint tests run on all three platforms.

Four ranks select the automatic static schedule; eight ranks select the automatic dynamic schedule (one manager and
seven compute workers). The scaling tests verify which ranks actually compute, completion with fewer points than
ranks, root/all/file-only result delivery, checkpoint resume, and failed-point filling. Explicit static and dynamic
schedules and guided versus single-point chunks are also exercised with idle ranks. MPI ranks each use one numerical
library thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`); the Open MPI setup action enables
oversubscription so the eight-rank checks can run on smaller hosted runners. These are correctness checks, not
performance benchmarks.

The static backend grants each worker permission before collecting its channels in order. This keeps large transfers
from other workers out of MPI's unexpected-message queue while root is receiving a different worker. All ranks then
synchronize before root finalizes the result store. Subprocess regression coverage includes a delayed receiver,
interleaved array/list channels with large arrays, a delayed sender, and root finalization failures; a watchdog dumps
blocked MPI stacks before the outer test timeout if a job hangs.

### Release ([release.yml](../.github/workflows/release.yml))

Runs on every push to `main`. `concurrency: release-main` with `cancel-in-progress: false` queues runs instead of
running them in parallel. Permissions: `contents: write` (needed to create the release and tag).

1. Read `__version__` from `src/parameterrun/__init__.py` with a regex, and set `tag = v<version>`.
2. Fail if that tag already exists on `origin`.
3. Fail if `CHANGELOG.md` has no `## [<version>]` heading. Otherwise its body becomes `release_notes.md`.
4. `gh release create v<version> --target <sha>` creates the release and the tag.

### Publish ([publish.yml](../.github/workflows/publish.yml))

Starts whenever the Release workflow completes, and runs its job only if Release succeeded. It checks out the commit
Release ran on (`workflow_run.head_sha`), builds with `python -m build`, validates with `twine check --strict`, and
uploads with `pypa/gh-action-pypi-publish`. There is no API token in the workflow: it uses the `pypi` environment and
an OIDC token (`id-token: write`), i.e. PyPI trusted publishing.

## What this means in practice

- **CI only runs on pull requests into `main`.** Pushes to `dev` or feature branches are not tested by GitHub Actions.
- **Every merge into `main` is a release.** If `__version__` was not bumped, Release fails at step 2 and nothing is
  published. If the `CHANGELOG.md` section is missing, it fails at step 3. Before merging: run
  `python scripts/bump_version.py …` and move the `[Unreleased]` entries under `## [X.Y.Z] - date`.
- **Release does not wait for CI.** It runs on the push to `main`, independent of the PR checks. Whether a PR with
  failing checks can be merged at all depends on branch protection in the GitHub settings, which is not visible in
  the repository (the dotted arrow in the chart).
- **A failed Release still triggers Publish**, because `workflow_run` fires on any completion; the `if:` condition
  then skips the job.
- `uv.lock` is gitignored, so every `uv sync` in CI resolves dependencies fresh. A new upstream release can therefore
  break CI without any change in this repository.
- Plain `uv sync` also installs the `dev` dependency group, which includes `parameterrun[mpi]`. So mpi4py is installed
  in the non-MPI matrix and in Publish as well, and `--extra mpi` in the MPI job does not change anything. (uv
  installs the `dev` group by default: this is uv's documented behaviour, not something the workflow files state.)
