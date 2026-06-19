#!/usr/bin/env bash
# v4 tiny WMH2017 smoke run wrapper
set -euo pipefail

ROOT=""
CONFIG="configs/wmh2017_monai_tiny_smoke.yaml"
RUN_ID=""
SEED="20260616"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  echo "Usage: $0 --root WMH2017_ROOT --config CONFIG --run-id RUN_ID [--seed SEED]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -n "$RUN_ID" ]] || usage
WORK_DIR="$REPO_ROOT/artifacts/runs/$RUN_ID"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

record_failure() {
  python - <<PY
from pathlib import Path
from wmh2017.evidence import record_failed_run
record_failed_run(Path("$WORK_DIR"), run_id="$RUN_ID", seed=int("$SEED"), reason="$1")
print("Recorded FAILED_RUN at $WORK_DIR")
PY
}

if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
  record_failure "MISSING_WMH2017_ROOT"
  echo "WMH2017_ROOT missing; recorded MISSING_WMH2017_ROOT evidence" >&2
  exit 0
fi

if ! python "$REPO_ROOT/scripts/run_wmh2017_e2e.py" \
  --files-root "$ROOT" \
  --run-id "$RUN_ID" \
  --seed "$SEED" \
  --work-dir "$WORK_DIR" \
  --config "$CONFIG" \
  --allow-dirty-git; then
  record_failure "pipeline_command_failed"
  exit 1
fi

python - <<PY
from pathlib import Path
from wmh2017.evidence import materialize_v4_artifacts
paths = materialize_v4_artifacts(Path("$WORK_DIR"), run_id="$RUN_ID", seed=int("$SEED"), status="COMPLETED_OR_FAILED_RUN")
print("Materialized v4 artifacts:")
for p in paths.values():
    print(" ", p)
PY

echo "Tiny smoke run complete: $WORK_DIR"
