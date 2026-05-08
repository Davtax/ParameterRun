#!/usr/bin/env bash
set -e

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

NPROCS="$1"
SCRIPT="$2"
shift 2

mkdir -p profiles

OUTFILE="profiles/rank0_scalene.json"

if [ "$NPROCS" -eq 1 ]; then
    uv run scalene run \
        --memory \
        --outfile "$OUTFILE" \
        "$SCRIPT" --- "$@"
else
    mpirun -n 1 uv run scalene run \
        --memory \
        --outfile "$OUTFILE" \
        "$SCRIPT" --- "$@" : \
        -n "$((NPROCS - 1))" uv run python "$SCRIPT" "$@"
fi

echo "Profile saved to: $OUTFILE"