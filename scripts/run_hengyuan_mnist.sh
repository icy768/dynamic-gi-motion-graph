#!/usr/bin/env bash
set -euo pipefail
GI_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GI_PROJECT_DIR="$(dirname -- "$GI_SCRIPT_DIR")"
cd "$GI_PROJECT_DIR"
mkdir -p runs
GI_LOG="runs/server_$(date +%Y%m%d_%H%M%S).log"
"${GI_PYTHON:-python}" -u run_server_experiment.py --stage all --require-cuda "$@" 2>&1 | tee "$GI_LOG"
