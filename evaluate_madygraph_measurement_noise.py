"""Evaluate clean-trained GI refiners under bucket-domain Gaussian read noise."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_graph_protocol_v2 import (
    BouncingMovingMNIST, IMAGE_SIZE, RATES, SEED, TEST_N, VAL_N,
    speed_bin_name,
)
from train_madygraph_gi_plugin import (
    BASE_DIR, MOTION_PATH, PROJECT, ResidualRefiner, metric_rows,
)
from train_spatial_motion_diagnostic import SpatialMotionNet


HERE = Path(__file__).resolve().parent
OUT = HERE / "madygraph_gi_plugin_noise_outputs"
TEST_BLOCK = 2
NOISE_LEVELS = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50)
NOISE_REPEATS = 3


def noisy_dgi_and_residual(target, patterns, noise_rel, generator):
    batch, frames, _, height, width = target.shape
    n_patterns = patterns.shape[0]
    raw_patterns = patterns.reshape(n_patterns, -1)
    operator = raw_patterns / np.sqrt(height * width)
    x = target.reshape(batch * frames, -1)
    buckets = x @ operator.T
    scale = buckets.std(dim=1, keepdim=True, unbiased=False)
    noise = torch.randn(
        buckets.shape, generator=generator, device=buckets.device, dtype=buckets.dtype
    ) * scale * noise_rel
    noisy_buckets = buckets + noise

    normalized_bucket = noisy_buckets / (patterns.sum((1, 2))[None] / np.sqrt(height * width)).clamp_min(1e-8)
    centered_bucket = normalized_bucket - normalized_bucket.mean(dim=1, keepdim=True)
    centered_patterns = patterns - patterns.mean(dim=0, keepdim=True)
    dgi = torch.einsum("nm,mhw->nhw", centered_bucket, centered_patterns) / n_patterns
    low = dgi.amin(dim=(-2, -1), keepdim=True)
    high = dgi.amax(dim=(-2, -1), keepdim=True)
    dgi = ((dgi - low) / (high - low + 1e-8))[:, None].reshape_as(target)

    coarse_input = dgi
    return coarse_input, noisy_buckets.reshape(batch, frames, n_patterns), operator


def summarize(rows):
    result = []
    for noise in NOISE_LEVELS:
        for rate in RATES:
            for method in sorted({row["method"] for row in rows}):
                group = [row for row in rows if row["noise_rel"] == noise and row["pattern_count"] == rate and row["method"] == method]
                entry = {"noise_rel": noise, "pattern_count": rate, "method": method, "sequences": len(group)}
                for key in ("psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error"):
                    vals = [row[key] for row in group]
                    entry[f"{key}_mean"] = float(np.mean(vals))
                    entry[f"{key}_std"] = float(np.std(vals, ddof=1))
                result.append(entry)
    return result


def main():
    sys.path[:0] = [str(PROJECT), str(HERE)]
    from mini_repro_dynghost import make_patterns
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED + 700)
    rng.permutation(60000)
    indices = rng.permutation(10000)[VAL_N + TEST_N * TEST_BLOCK:VAL_N + TEST_N * (TEST_BLOCK + 1)]
    test = BouncingMovingMNIST(PROJECT / "data", False, indices, SEED + 703 + TEST_BLOCK)
    loader = DataLoader(test, batch_size=4, shuffle=False)

    transformer = MiniSpatioTemporalTransformer().to(device).eval()
    transformer.load_state_dict(torch.load(BASE_DIR / "best_transformer.pt", map_location=device))
    motion_net = SpatialMotionNet().to(device).eval()
    motion_net.load_state_dict(torch.load(MOTION_PATH, map_location=device))
    manifest = json.loads((HERE / "madygraph_gi_plugin_v2_outputs" / "data_manifest.json").read_text(encoding="utf-8"))
    models = {}
    for kind in ("cnn", "fixed", "motion"):
        model = ResidualRefiner(kind).to(device).eval()
        model.load_state_dict(torch.load(HERE / "madygraph_gi_plugin_v2_outputs" / f"best_{kind}_refiner.pt", map_location=device))
        models[kind] = model

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for rate in RATES:
            patterns = make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
            for noise_index, noise_rel in enumerate(NOISE_LEVELS):
                for noise_repeat in range(NOISE_REPEATS):
                    generator = torch.Generator(device=device).manual_seed(
                        SEED + 12000 + rate + noise_index + noise_repeat * 1000
                    )
                    for target, labels, _, speeds, sample_ids in loader:
                        target = target.to(device)
                        dgi, noisy_buckets, operator = noisy_dgi_and_residual(
                            target, patterns, noise_rel, generator
                        )
                        coarse = transformer(dgi)
                        b, t = dgi.shape[:2]
                        x0 = coarse.reshape(b, t, -1)
                        predicted = x0 @ operator.T
                        residual = ((noisy_buckets - predicted) @ operator) / rate
                        residual = residual.reshape_as(coarse)
                        residual = (
                            residual / manifest["measurement_residual_train_rms_by_rate"][str(rate)]
                        ).clamp(-8, 8)
                        pair = torch.cat((dgi[:, :-1], dgi[:, 1:]), dim=2)
                        motion = motion_net(pair.reshape(b * (t - 1), 2, IMAGE_SIZE, IMAGE_SIZE)).reshape(b, t - 1, 2)

                        outputs = {"TransformerCoarse": coarse.clamp(0, 1)}
                        for kind, model in models.items():
                            outputs[{"cnn": "CNNResidual", "fixed": "FixedGraphResidual", "motion": "MotionGraphResidual"}[kind]] = model(
                                coarse, residual, motion
                            ).clamp(0, 1)
                        for name, pred in outputs.items():
                            metric_batch = metric_rows(pred, target, labels, sample_ids)
                            for sample_index, row in enumerate(metric_batch):
                                rows.append({
                                    "noise_rel": noise_rel,
                                    "noise_repeat": noise_repeat,
                                    "bucket_snr_db_approx": (float("inf") if noise_rel == 0 else float(-20 * np.log10(noise_rel))),
                                    "pattern_count": rate,
                                    "speed_pixels_per_frame": float(speeds[sample_index]),
                                    "method": name,
                                    **row,
                                })

    for row in rows:
        row["speed_bin"] = speed_bin_name(row["speed_pixels_per_frame"])
    summary = summarize(rows)
    with (OUT / "noise_detail.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "noise_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    result = {
        "noise_model": "independent additive Gaussian bucket read noise; per-frame sigma = noise_rel * clean bucket standard deviation",
        "approximate_bucket_snr_db": {str(level): (None if level == 0 else float(-20 * np.log10(level))) for level in NOISE_LEVELS},
        "test_block": TEST_BLOCK,
        "test_sequences": len(test),
        "independent_noise_draws_per_level": NOISE_REPEATS,
        "summary": summary,
    }
    (OUT / "noise_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
