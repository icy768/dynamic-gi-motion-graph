"""Notebook-scale numerical checks for GI operators and motion warping."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

PROJECT = Path(r"D:\download\DynGhost-mini-repro")
sys.path.insert(0, str(PROJECT))
from train_madygraph_gi_plugin import gi_forward, physical_residual
from train_graph_protocol_v2 import IMAGE_SIZE
from train_oracle_motion_tgnn import warp_feature


def main() -> None:
    torch.manual_seed(20260930)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    height = width = IMAGE_SIZE
    m = 128
    patterns = torch.randint(0, 2, (m, height, width), device=device).to(dtype)
    matrix = patterns.reshape(m, -1) / math.sqrt(height * width)
    x = torch.randn(2, 3, height * width, device=device, dtype=dtype)
    z = torch.randn(2, 3, m, device=device, dtype=dtype)
    left = ((x @ matrix.T) * z).sum()
    right = (x * (z @ matrix).reshape_as(x)).sum()
    adjoint_error = float((left - right).abs() / torch.maximum(left.abs(), right.abs()).clamp_min(1))

    target = torch.rand(2, 3, 1, height, width, device=device, dtype=dtype)
    coarse = torch.rand_like(target)
    measured = gi_forward(target, patterns)
    residual = physical_residual(measured, coarse, patterns)
    reference = ((measured - coarse.reshape(2, 3, -1) @ matrix.T) @ matrix / m)
    residual_error = float((residual.flatten(2) - reference).abs().max())
    # The public residual API receives bucket values, never the target frame.
    api_parameters = list(__import__("inspect").signature(physical_residual).parameters)

    feature = torch.zeros(1, 1, 16, 16, device=device)
    feature[0, 0, 8, 8] = 1
    identity = float((warp_feature(feature, torch.tensor([0.], device=device), torch.tensor([0.], device=device)) - feature).abs().max())
    shifted = warp_feature(feature, torch.tensor([4.], device=device), torch.tensor([-4.], device=device))
    peak = torch.nonzero(shifted[0, 0] == shifted.max(), as_tuple=False)[0].tolist()
    integer_shift_error = float((shifted[0, 0, 7, 9] - 1).abs())
    half = warp_feature(feature, torch.tensor([2.], device=device), torch.tensor([0.], device=device))
    half_shift_left = float(half[0, 0, 8, 8])
    half_shift_right = float(half[0, 0, 8, 8 + 1])
    payload = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dtype": str(dtype),
        "measurement": {
            "adjoint_relative_error": adjoint_error,
            "residual_max_abs_error_vs_explicit_formula": residual_error,
            "residual_api_parameters": api_parameters,
        },
        "warp": {
            "zero_displacement_max_abs_error": identity,
            "dx_plus4_dy_minus4_peak_yx": peak,
            "integer_translation_target_max_abs_error": integer_shift_error,
            "half_shift_sample_at_original": half_shift_left,
            "half_shift_sample_one_feature_cell_right": half_shift_right,
            "pixel_to_feature_scale": 4,
        },
    }
    out = Path(__file__).resolve().parent / "notebook_verification_outputs"
    out.mkdir(exist_ok=True)
    (out / "operator_warp_checks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    assert adjoint_error < 1e-12
    assert residual_error < 1e-12
    assert identity < 1e-6 and peak == [7, 9] and integer_shift_error < 1e-6
    assert abs(half_shift_left - 0.5) < 1e-6 and abs(half_shift_right - 0.5) < 1e-6
    assert api_parameters == ["bucket", "coarse", "patterns"]


if __name__ == "__main__":
    main()
