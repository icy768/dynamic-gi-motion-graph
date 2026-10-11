"""Train and evaluate the graph using position-preserving estimated edges."""

from __future__ import annotations

from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_graph_protocol_v2 import (
    IMAGE_SIZE,
    RATES,
    SEED,
    JointPairGraph,
    evaluate,
    make_splits,
    save_csv,
    seed_all,
    train_reconstructor,
)
from train_spatial_motion_diagnostic import SpatialMotionNet


PROJECT = PROJECT_ROOT
BASELINE = Path("graph_protocol_v2_trained")
MOTION = Path("spatial_motion_outputs/best_spatial_motion_net.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / 'spatial_estimated_graph_outputs')
    args = parser.parse_args()
    sys.path[:0] = [str(PROJECT), str(Path(__file__).resolve().parent)]
    from mini_repro_dynghost import make_patterns
    from train_oracle_motion_tgnn import OracleMotionTGNN
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train, val, test = make_splits(PROJECT)
    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    val_loader = DataLoader(val, batch_size=8, shuffle=False)
    test_loader = DataLoader(test, batch_size=4, shuffle=False)
    bank = {rate: make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device) for rate in RATES}

    transformer = MiniSpatioTemporalTransformer().to(device)
    transformer.load_state_dict(torch.load(BASELINE / "best_transformer.pt", map_location=device))
    zero_graph = OracleMotionTGNN().to(device)
    zero_graph.load_state_dict(torch.load(BASELINE / "best_zero_graph.pt", map_location=device))
    oracle_graph = OracleMotionTGNN().to(device)
    oracle_graph.load_state_dict(torch.load(BASELINE / "best_oracle_pair_graph.pt", map_location=device))

    motion_net = SpatialMotionNet().to(device)
    motion_net.load_state_dict(torch.load(MOTION, map_location=device))
    # First establish the value of the estimated graph without allowing the
    # reconstruction objective to destroy the already validated motion head.
    for parameter in motion_net.parameters():
        parameter.requires_grad_(False)
    seed_all(SEED)
    joint = JointPairGraph(motion_net, OracleMotionTGNN().to(device)).to(device)
    history = train_reconstructor(
        "GraphEstimatedPair", joint, train_loader, val_loader, bank, device,
        args.epochs, output / "best_spatial_estimated_graph.pt", motion_weight=0.0,
    )

    detail = []
    for rate in RATES:
        rows = evaluate(transformer, zero_graph, oracle_graph, joint, test_loader, bank[rate], device)
        for row in rows:
            detail.append({"pattern_count": rate, "sampling_ratio": rate / IMAGE_SIZE**2, **row})

    methods = ("Transformer", "GraphZero", "GraphOraclePair", "GraphEstimatedPair")
    summary = []
    gains = []
    index = {(r["pattern_count"], r["method"], r["sequence_id"]): r for r in detail}
    for rate in RATES:
        for speed_bin in ("1-2", "2-4", "4-6", "all"):
            for method in methods:
                selected = [r for r in detail if r["pattern_count"] == rate and r["method"] == method and (speed_bin == "all" or r["speed_bin"] == speed_bin)]
                row = {"pattern_count": rate, "speed_bin": speed_bin, "method": method, "sequences": len(selected)}
                for key in ("psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error"):
                    values = np.asarray([r[key] for r in selected], dtype=np.float64)
                    row[f"{key}_mean"] = float(values.mean())
                    row[f"{key}_std"] = float(values.std(ddof=1))
                row["motion_epe_pixels_mean"] = float(np.mean([float(r["motion_epe_pixels"]) for r in selected])) if method == "GraphEstimatedPair" else ""
                summary.append(row)
                print(f"patterns={rate:3d} speed={speed_bin:3s} method={method:18s} PSNR={row['psnr_db_mean']:.2f} SSIM={row['ssim_mean']:.3f}")
        for method in methods[1:]:
            for speed_bin in ("1-2", "2-4", "4-6", "all"):
                ids = [r["sequence_id"] for r in detail if r["pattern_count"] == rate and r["method"] == "Transformer" and (speed_bin == "all" or r["speed_bin"] == speed_bin)]
                delta = np.asarray([index[(rate, method, i)]["psnr_db"] - index[(rate, "Transformer", i)]["psnr_db"] for i in ids])
                gains.append({
                    "pattern_count": rate,
                    "speed_bin": speed_bin,
                    "method": method,
                    "paired_psnr_gain_mean_db": float(delta.mean()),
                    "paired_psnr_gain_std_db": float(delta.std(ddof=1)),
                    "paired_win_rate": float(np.mean(delta > 0)),
                })
    save_csv(output / "benchmark_detail.csv", detail)
    save_csv(output / "benchmark_summary.csv", summary)
    save_csv(output / "paired_gains.csv", gains)
    (output / "results.json").write_text(json.dumps({"history": history, "summary": summary, "paired_gains": gains}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
