"""Run a small validation-only end-to-end smoke check; not a benchmark."""
from __future__ import annotations

from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT = PROJECT_ROOT
sys.path[:0] = [str(PROJECT), str(HERE)]

from mini_repro_dynghost import make_patterns
from train_graph_protocol_v2 import IMAGE_SIZE, RATES, SEED, make_splits
from train_madygraph_gi_plugin import ResidualRefiner, gi_forward, physical_residual
from train_spatial_motion_diagnostic import SpatialMotionNet
from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
from train_mixed_sampling import coarse_sequence


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation, _ = make_splits(PROJECT)
    loader = DataLoader(validation, batch_size=4, shuffle=False)
    transformer = MiniSpatioTemporalTransformer().to(device)
    transformer.load_state_dict(torch.load(HERE / "graph_protocol_v2_trained" / "best_transformer.pt", map_location=device))
    motion_net = SpatialMotionNet().to(device)
    motion_net.load_state_dict(torch.load(PROJECT_ROOT / 'spatial_motion_outputs' / "best_spatial_motion_net.pt", map_location=device))
    refiners = {}
    for name in ("cnn", "fixed", "motion"):
        model = ResidualRefiner(name).to(device)
        model.load_state_dict(torch.load(PROJECT_ROOT / 'madygraph_gi_plugin_v2_outputs' / f"best_{name}_refiner.pt", map_location=device))
        model.eval()
        refiners[name] = model
    transformer.eval(); motion_net.eval()
    records = []
    with torch.inference_mode():
        for target, labels, truth_motion, speeds, indices in loader:
            target = target.to(device)
            b, t = target.shape[:2]
            for rate in RATES:
                patterns = make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
                dgi = coarse_sequence(target, patterns)
                coarse = transformer(dgi)
                bucket = gi_forward(target, patterns)
                residual = physical_residual(bucket, coarse, patterns)
                pairs = torch.cat((dgi[:, :-1], dgi[:, 1:]), dim=2)
                motion = motion_net(pairs.reshape(b * (t - 1), 2, IMAGE_SIZE, IMAGE_SIZE)).reshape(b, t - 1, 2)
                predictions = {"transformer": coarse}
                for name, model in refiners.items():
                    predictions[name] = model(coarse, residual, motion)
                for name, prediction in predictions.items():
                    clipped = prediction.clamp(0, 1)
                    consistency = (gi_forward(clipped, patterns) - bucket).norm(dim=-1) / bucket.norm(dim=-1).clamp_min(1e-12)
                    mse = (clipped - target).square().mean((1, 2, 3, 4))
                    records.extend({
                        "rate": rate,
                        "method": name,
                        "psnr_db": float(10 * torch.log10(1 / mse[i].clamp_min(1e-12))),
                        "relative_measurement_residual": float(consistency[i].mean()),
                        "motion_epe_px": float((motion - truth_motion.to(device)).norm(dim=-1)[i].mean()),
                    } for i in range(b))
    summaries = []
    for rate in RATES:
        for method in ("transformer", "cnn", "fixed", "motion"):
            group = [r for r in records if r["rate"] == rate and r["method"] == method]
            summaries.append({
                "rate": rate,
                "method": method,
                "n_sequences": len(group),
                "mean_psnr_db": sum(r["psnr_db"] for r in group) / len(group),
                "mean_relative_measurement_residual": sum(r["relative_measurement_residual"] for r in group) / len(group),
                "mean_motion_epe_px": sum(r["motion_epe_px"] for r in group) / len(group),
            })
    out = PROJECT_ROOT / 'notebook_verification_outputs'
    out.mkdir(exist_ok=True)
    payload = {
        "purpose": "full validation-set end-to-end diagnostic; not the held-out final test",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "summary": summaries,
    }
    (out / "end_to_end_smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
