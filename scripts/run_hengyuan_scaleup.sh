#!/usr/bin/env bash
# Use inside tmux. Optional upload and shutdown are requested by explicit flags.
set -euo pipefail
GI_UPLOAD=0
GI_SHUTDOWN=0
for GI_ARGUMENT in "$@"; do
    case "$GI_ARGUMENT" in
        --upload) GI_UPLOAD=1 ;;
        --shutdown-after-upload) GI_SHUTDOWN=1 ;;
        *) echo "Unknown argument: $GI_ARGUMENT" >&2; exit 2 ;;
    esac
done
if [[ "$GI_SHUTDOWN" == 1 && "$GI_UPLOAD" != 1 ]]; then
    echo "--shutdown-after-upload requires --upload" >&2
    exit 2
fi
GI_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GI_PROJECT_DIR="$(dirname -- "$GI_SCRIPT_DIR")"
cd "$GI_PROJECT_DIR"
GI_PYTHON="${GI_PYTHON:-python}"
GI_RUN="runs/mnist_scaleup_v1"
mkdir -p "$GI_RUN"
command -v flock >/dev/null
exec 9>"$GI_RUN/.launcher.lock"
flock -n 9 || { echo "This experiment is already running." >&2; exit 2; }
GI_LOG="$GI_RUN/launcher_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$GI_LOG") 2>&1
trap 'GI_EXIT=$?; if [[ "$GI_EXIT" != 0 ]]; then echo "Experiment stopped with error $GI_EXIT. Saved epochs can be resumed with the same command. Instance is still running."; fi' EXIT
if [[ "$GI_UPLOAD" == 1 ]]; then
    command -v oss >/dev/null
    oss ls -s -d oss:// >/dev/null
fi
if [[ "$GI_SHUTDOWN" == 1 ]]; then
    if [[ "$GI_PROJECT_DIR" != /hy-tmp/gi_project ]]; then
        echo "Automatic shutdown is only enabled from /hy-tmp/gi_project" >&2
        exit 2
    fi
    command -v shutdown >/dev/null
fi
"$GI_PYTHON" -u run_server_experiment.py --config configs/mnist_scaleup.json --stage all --require-cuda
"$GI_PYTHON" package_gi_project.py --results --config configs/mnist_scaleup.json
GI_ARCHIVE="dist/gi_results_mnist_scaleup_v1.zip"
GI_CHECKSUM="dist/gi_results_mnist_scaleup_v1_checksums.zip"
if [[ "$GI_UPLOAD" == 1 ]]; then
    oss cp "$GI_ARCHIVE" oss://
    oss cp "$GI_CHECKSUM" oss://
    echo "Results and checksum uploaded successfully."
fi
if [[ "$GI_SHUTDOWN" == 1 ]]; then
    echo "All stages and both uploads passed; shutting down instance."
    shutdown
else
    echo "Completed: $GI_ARCHIVE. Instance remains running; stop it after saving the results."
fi
