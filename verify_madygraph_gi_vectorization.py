"""Verify vectorized graph implementation against the recorded v2 benchmark."""

from __future__ import annotations

from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_graph_protocol_v2 import (
    BouncingMovingMNIST, IMAGE_SIZE, RATES, SEED, TEST_N, VAL_N,
)
from train_madygraph_gi_plugin import (
    BASE_DIR, MOTION_PATH, PROJECT, ResidualRefiner, metric_rows, precompute,
)
from train_spatial_motion_diagnostic import SpatialMotionNet


HERE = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / 'madygraph_gi_plugin_v2_outputs'


def main():
    sys.path[:0] = [str(PROJECT), str(HERE)]
    from mini_repro_dynghost import make_patterns
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED + 700)
    rng.permutation(60000)
    indices = rng.permutation(10000)[VAL_N + TEST_N:VAL_N + 2 * TEST_N]
    test = BouncingMovingMNIST(PROJECT / "data", False, indices, SEED + 704)
    loader = DataLoader(test, batch_size=4, shuffle=False)
    bank = {rate: make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device) for rate in RATES}
    transformer = MiniSpatioTemporalTransformer().to(device)
    transformer.load_state_dict(torch.load(BASE_DIR / "best_transformer.pt", map_location=device))
    motion_net = SpatialMotionNet().to(device)
    motion_net.load_state_dict(torch.load(MOTION_PATH, map_location=device))
    data = precompute(loader, transformer, motion_net, bank, device)
    manifest = json.loads((OUT / "data_manifest.json").read_text(encoding="utf-8"))
    for rate in RATES:
        data[rate]["residual"] = (
            data[rate]["residual"] / manifest["measurement_residual_train_rms_by_rate"][str(rate)]
        ).clamp(-8, 8)
    reports = {}
    for kind in ("cnn", "fixed", "motion"):
        model = ResidualRefiner(kind).to(device)
        model.load_state_dict(torch.load(OUT / f"best_{kind}_refiner.pt", map_location=device))
        model.eval()
        reports[kind] = {}
        with torch.inference_mode():
            for rate in RATES:
                current = data[rate]
                rows = []
                for start in range(0, len(current["target"]), 4):
                    batch = {key: val[start:start + 4].to(device) for key, val in current.items()}
                    pred = model(batch["coarse"], batch["residual"], batch["motion"]).clamp(0, 1)
                    rows.extend(metric_rows(pred, batch["target"], batch["labels"], batch["indices"]))
                reports[kind][str(rate)] = {
                    key: float(np.mean([row[key] for row in rows]))
                    for key in ("psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error")
                }
    motion_model = ResidualRefiner("motion").to(device)
    motion_model.load_state_dict(torch.load(OUT / "best_motion_refiner.pt", map_location=device))
    motion_model.eval()
    ablations = {}
    with torch.inference_mode():
        for rate in RATES:
            current = data[rate]
            original_motion = current["motion"]
            variants = {
                "estimated_correct": original_motion,
                "zero_motion_at_inference": torch.zeros_like(original_motion),
                "wrong_sequence_motion_roll37": torch.roll(original_motion, shifts=37, dims=0),
                "reversed_motion_at_inference": -original_motion,
            }
            ablations[str(rate)] = {}
            for variant, motion in variants.items():
                rows = []
                for start in range(0, len(current["target"]), 4):
                    end = start + 4
                    batch = {key: val[start:end].to(device) for key, val in current.items()}
                    pred = motion_model(
                        batch["coarse"], batch["residual"], motion[start:end].to(device)
                    ).clamp(0, 1)
                    rows.extend(metric_rows(pred, batch["target"], batch["labels"], batch["indices"]))
                ablations[str(rate)][variant] = {
                    key: float(np.mean([row[key] for row in rows]))
                    for key in ("psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error")
                }
    reports["motion_input_ablation"] = ablations
    result_path = OUT / "vectorized_equivalence_metrics.json"
    result_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"saved={result_path}")


if __name__ == "__main__":
    main()
