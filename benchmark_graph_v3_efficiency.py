"""Measure comparable model-only inference cost for the final diagnostic models."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from train_graph_protocol_v2 import JointPairGraph
from train_spatial_motion_diagnostic import SpatialMotionNet


PROJECT = Path(r"D:\download\DynGhost-mini-repro")


def measure(name, model, sample, forward, repeats=50):
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(repeats):
            forward()
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000 / repeats
    return {
        "method": name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "latency_ms_batch1_sequence8": latency,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    sys.path.insert(0, str(PROJECT))
    from train_oracle_motion_tgnn import OracleMotionTGNN
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
    from evaluate_graph_protocol_v2 import per_pair_reconstruct

    device = torch.device("cuda")
    transformer = MiniSpatioTemporalTransformer().to(device)
    transformer.load_state_dict(torch.load("graph_protocol_v2_trained/best_transformer.pt", map_location=device))
    oracle = OracleMotionTGNN().to(device)
    oracle.load_state_dict(torch.load("graph_protocol_v2_trained/best_oracle_pair_graph.pt", map_location=device))
    motion = SpatialMotionNet().to(device)
    motion.load_state_dict(torch.load("spatial_motion_outputs/best_spatial_motion_net.pt", map_location=device))
    estimated = JointPairGraph(motion, OracleMotionTGNN().to(device)).to(device)
    estimated.load_state_dict(torch.load("spatial_estimated_graph_outputs/best_spatial_estimated_graph.pt", map_location=device))
    sample = torch.rand(1, 8, 1, 64, 64, device=device)
    pair = torch.rand(1, 7, 2, device=device) * 4
    rows = [
        measure("Transformer", transformer, sample, lambda: transformer(sample)),
        measure("GraphOraclePair", oracle, sample, lambda: per_pair_reconstruct(oracle, sample, pair)),
        measure("GraphEstimatedPair", estimated, sample, lambda: estimated(sample)),
    ]
    for row in rows:
        print(row)
    Path("spatial_estimated_graph_outputs/efficiency.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
