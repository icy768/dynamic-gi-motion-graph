#!/usr/bin/env bash
set -euo pipefail
GI_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$GI_SCRIPT_DIR")"
GI_PYTHON="${GI_PYTHON:-python}"
if ! "$GI_PYTHON" -c 'import torch, torchvision; assert torch.__version__.split("+")[0] == "2.9.1"; assert torchvision.__version__.split("+")[0] == "0.24.1"; assert torch.version.cuda == "12.8"' >/dev/null 2>&1; then
    "$GI_PYTHON" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
fi
"$GI_PYTHON" -m pip install -r requirements-hengyuan.txt
"$GI_PYTHON" -c 'import torch, torchvision; print("torch:", torch.__version__, "torchvision:", torchvision.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
"$GI_PYTHON" run_server_experiment.py --config configs/mnist_scaleup.json --stage check --require-cuda
