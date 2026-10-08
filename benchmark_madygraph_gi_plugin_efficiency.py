"""Measure batch-1 inference latency for the MadyGraph-inspired GI pilot."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_graph_protocol_v2 import (
    BouncingMovingMNIST, IMAGE_SIZE, RATES, SEED, TEST_N, VAL_N,
)
from train_madygraph_gi_plugin import (
    BASE_DIR, MOTION_PATH, PROJECT, ResidualRefiner, physical_residual,
)
from train_spatial_motion_diagnostic import SpatialMotionNet


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "madygraph_gi_plugin_v2_outputs" / "efficiency.json"


def measure(fn, device, warmup=15, repeats=60):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": float(np.median(samples)),
        "p90_ms": float(np.percentile(samples, 90)),
        "mean_ms": float(np.mean(samples)),
    }


def main():
    sys.path[:0] = [str(PROJECT), str(HERE)]
    from mini_repro_dynghost import make_patterns
    from train_mixed_sampling import coarse_sequence
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_rng = np.random.default_rng(SEED + 700)
    split_rng.permutation(60000)
    heldout_indices = split_rng.permutation(10000)
    test_set = BouncingMovingMNIST(
        PROJECT / "data", False,
        heldout_indices[VAL_N + TEST_N:VAL_N + 2 * TEST_N], SEED + 704,
    )
    target = next(iter(DataLoader(test_set, batch_size=1, shuffle=False)))[0].to(device)
    transformer = MiniSpatioTemporalTransformer().to(device).eval()
    transformer.load_state_dict(torch.load(BASE_DIR / "best_transformer.pt", map_location=device))
    motion_net = SpatialMotionNet().to(device).eval()
    motion_net.load_state_dict(torch.load(MOTION_PATH, map_location=device))
    refiner_cnn = ResidualRefiner("cnn").to(device).eval()
    refiner_cnn.load_state_dict(torch.load(HERE / "madygraph_gi_plugin_v2_outputs" / "best_cnn_refiner.pt", map_location=device))
    refiner_fixed = ResidualRefiner("fixed").to(device).eval()
    refiner_fixed.load_state_dict(torch.load(HERE / "madygraph_gi_plugin_v2_outputs" / "best_fixed_refiner.pt", map_location=device))
    refiner_motion = ResidualRefiner("motion").to(device).eval()
    refiner_motion.load_state_dict(torch.load(HERE / "madygraph_gi_plugin_v2_outputs" / "best_motion_refiner.pt", map_location=device))

    manifest = json.loads((HERE / "madygraph_gi_plugin_v2_outputs" / "data_manifest.json").read_text(encoding="utf-8"))
    results = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "batch_size": 1, "frames": 8, "resolution": [IMAGE_SIZE, IMAGE_SIZE], "warmup": 15, "repeats": 60, "rates": {}}
    for rate in RATES:
        patterns = make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
        dgi = coarse_sequence(target, patterns)
        coarse = transformer(dgi)
        residual = physical_residual(target, coarse, patterns)
        residual = (residual / manifest["measurement_residual_train_rms_by_rate"][str(rate)]).clamp(-8, 8)
        b, frames = coarse.shape[:2]
        pairs = torch.cat((dgi[:, :-1], dgi[:, 1:]), dim=2)
        estimated = motion_net(pairs.reshape(b * (frames - 1), 2, IMAGE_SIZE, IMAGE_SIZE)).reshape(b, frames - 1, 2)

        def base():
            return transformer(coarse_sequence(target, patterns))

        def cnn_full():
            d = coarse_sequence(target, patterns)
            c = transformer(d)
            r = physical_residual(target, c, patterns)
            r = (r / manifest["measurement_residual_train_rms_by_rate"][str(rate)]).clamp(-8, 8)
            return refiner_cnn(c, r, estimated)

        def fixed_full():
            d = coarse_sequence(target, patterns)
            c = transformer(d)
            r = physical_residual(target, c, patterns)
            r = (r / manifest["measurement_residual_train_rms_by_rate"][str(rate)]).clamp(-8, 8)
            return refiner_fixed(c, r, estimated)

        def motion_full():
            d = coarse_sequence(target, patterns)
            c = transformer(d)
            r = physical_residual(target, c, patterns)
            r = (r / manifest["measurement_residual_train_rms_by_rate"][str(rate)]).clamp(-8, 8)
            p = torch.cat((d[:, :-1], d[:, 1:]), dim=2)
            m = motion_net(p.reshape(b * (frames - 1), 2, IMAGE_SIZE, IMAGE_SIZE)).reshape(b, frames - 1, 2)
            return refiner_motion(c, r, m)

        results["rates"][str(rate)] = {
            "transformer_coarse_end_to_end": measure(base, device),
            "cnn_plugin_end_to_end": measure(cnn_full, device),
            "fixed_graph_plugin_end_to_end": measure(fixed_full, device),
            "motion_graph_plugin_end_to_end": measure(motion_full, device),
        }
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
