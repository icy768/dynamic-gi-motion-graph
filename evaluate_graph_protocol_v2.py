"""Graph protocol v2 diagnostics.

This script does not retrain models.  It reuses the existing Transformer and
JointMotionTGNN checkpoints and isolates the effect of temporal graph routing:

* zero motion
* the current sequence-mean estimated motion
* per-pair estimated motion
* per-pair oracle motion

Evaluation is performed on deterministic constant-speed and acceleration
Moving-MNIST sequences.  All methods see the same targets, speckles and DGI
inputs for every comparison.
"""

from __future__ import annotations

from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = PROJECT_ROOT
DEFAULT_OUT = PROJECT_ROOT / 'graph_protocol_v2_outputs'
SEED = 20260902
SEQ_LEN = 8
IMAGE_SIZE = 64


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ControlledMovingMNIST(Dataset):
    """Deterministic sequences with known frame-to-frame translations."""

    def __init__(
        self,
        root: Path,
        length: int,
        seed: int,
        scenario: str,
    ) -> None:
        self.mnist = datasets.MNIST(
            root=root, train=False, download=False, transform=transforms.ToTensor()
        )
        self.length = length
        self.scenario = scenario
        rng = np.random.default_rng(seed)
        # Use a deterministic permutation rather than sampling with replacement.
        self.indices = rng.permutation(len(self.mnist))[:length]
        self.start_x = rng.uniform(-4.0, 4.0, size=length)
        self.start_y = rng.uniform(-4.0, 4.0, size=length)
        self.angle = rng.uniform(0.0, 2.0 * math.pi, size=length)

        ranges = {
            "speed_1_2": (1.0, 2.0),
            "speed_2_4": (2.0, 4.0),
            "speed_4_6": (4.0, 6.0),
            "speed_6_8": (6.0, 8.0),
        }
        if scenario in ranges:
            lo, hi = ranges[scenario]
            self.speed0 = rng.uniform(lo, hi, size=length)
            self.acceleration = np.zeros(length, dtype=np.float64)
        elif scenario == "acceleration":
            self.speed0 = rng.uniform(1.0, 2.0, size=length)
            self.acceleration = rng.uniform(0.45, 0.75, size=length)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        image, label = self.mnist[int(self.indices[index])]
        image = F.interpolate(
            image[None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0]
        theta0 = float(self.angle[index])
        x = float(self.start_x[index])
        y = float(self.start_y[index])
        frames = []
        pair_motion = []
        for frame_index in range(SEQ_LEN):
            theta = torch.tensor(
                [[[1.0, 0.0, -2.0 * x / IMAGE_SIZE],
                  [0.0, 1.0, -2.0 * y / IMAGE_SIZE]]],
                dtype=torch.float32,
            )
            grid = F.affine_grid(
                theta, (1, 1, IMAGE_SIZE, IMAGE_SIZE), align_corners=False
            )
            frames.append(
                F.grid_sample(
                    image[None], grid, align_corners=False, padding_mode="zeros"
                )[0]
            )
            if frame_index + 1 < SEQ_LEN:
                speed = float(self.speed0[index] + self.acceleration[index] * frame_index)
                dx = speed * math.cos(theta0)
                dy = speed * math.sin(theta0)
                pair_motion.append((dx, dy))
                x += dx
                y += dy
        return (
            torch.stack(frames),
            int(label),
            torch.tensor(pair_motion, dtype=torch.float32),
            int(self.indices[index]),
        )

    def manifest(self) -> list[dict]:
        return [
            {
                "sequence_id": i,
                "mnist_test_index": int(self.indices[i]),
                "scenario": self.scenario,
                "start_x": float(self.start_x[i]),
                "start_y": float(self.start_y[i]),
                "angle_rad": float(self.angle[i]),
                "initial_speed_pixels": float(self.speed0[i]),
                "acceleration_pixels_per_frame2": float(self.acceleration[i]),
            }
            for i in range(self.length)
        ]


def per_pair_reconstruct(model: nn.Module, x: torch.Tensor, pair_motion: torch.Tensor):
    """Run OracleMotionTGNN weights with one displacement per adjacent pair."""
    from train_oracle_motion_tgnn import DIM, spatial_neighbor_mean, warp_feature

    batch, frames, channels, height, width = x.shape
    flat = x.reshape(batch * frames, channels, height, width)
    enc1 = model.enc1(flat)
    enc2 = model.enc2(enc1)
    feature_height, feature_width = enc2.shape[-2:]
    sequence = enc2.reshape(batch, frames, DIM, feature_height, feature_width)
    for block in model.blocks:
        updates = []
        for frame_index in range(frames):
            current = sequence[:, frame_index]
            spatial = spatial_neighbor_mean(current)
            previous = current if frame_index == 0 else warp_feature(
                sequence[:, frame_index - 1],
                pair_motion[:, frame_index - 1, 0],
                pair_motion[:, frame_index - 1, 1],
            )
            following = current if frame_index + 1 == frames else warp_feature(
                sequence[:, frame_index + 1],
                -pair_motion[:, frame_index, 0],
                -pair_motion[:, frame_index, 1],
            )
            update = block.self_proj(current)
            update = update + block.spatial_proj(spatial)
            update = update + block.prev_proj(previous) + block.next_proj(following)
            updates.append(F.gelu(block.norm(update)) + current)
        sequence = torch.stack(updates, dim=1)
    graph = sequence.reshape(batch * frames, DIM, feature_height, feature_width)
    graph = F.relu(model.up1(graph))
    graph = model.fuse1(torch.cat([graph, enc1], dim=1))
    graph = F.relu(model.up2(graph))
    output = model.out(torch.cat([graph, flat], dim=1))
    return torch.sigmoid(output).reshape(batch, frames, 1, height, width)


def estimated_pair_motion(motion_net: nn.Module, coarse: torch.Tensor) -> torch.Tensor:
    batch, frames, _, height, width = coarse.shape
    pairs = torch.cat((coarse[:, :-1], coarse[:, 1:]), dim=2)
    pairs = pairs.reshape(batch * (frames - 1), 2, height, width)
    return motion_net(pairs).reshape(batch, frames - 1, 2)


def foreground_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target > 0.05
    squared = (pred - target).square() * mask
    count = mask.sum((1, 2, 3, 4)).clamp_min(1)
    mse = squared.sum((1, 2, 3, 4)) / count
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def metric_rows(
    pred: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    mnist_indices: torch.Tensor,
) -> list[dict]:
    batch, frames, channels, height, width = target.shape
    mse = (pred - target).square().mean((1, 2, 3, 4))
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    score = ssim(
        pred.reshape(batch * frames, channels, height, width),
        target.reshape(batch * frames, channels, height, width),
        data_range=1.0,
        size_average=False,
    ).reshape(batch, frames).mean(1)
    temporal = (
        (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])
    ).abs().mean((1, 2, 3, 4))
    fg_psnr = foreground_psnr(pred, target)
    return [
        {
            "label": int(labels[i]),
            "mnist_test_index": int(mnist_indices[i]),
            "mse": float(mse[i]),
            "psnr_db": float(psnr[i]),
            "ssim": float(score[i]),
            "foreground_psnr_db": float(fg_psnr[i]),
            "temporal_difference_error": float(temporal[i]),
        }
        for i in range(batch)
    ]


def summarize(rows: list[dict]) -> dict:
    result = {"sequences": len(rows)}
    for key in ("mse", "psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=1))
    return result


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sequences", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pattern-counts", type=int, nargs="+", default=(128, 256))
    args = parser.parse_args()

    project = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(project), str(HERE)]
    seed_all(SEED)

    from mini_repro_dynghost import make_patterns
    from train_dgi_motion_estimator import DgiMotionNet
    from train_mixed_sampling import coarse_sequence
    from train_oracle_motion_tgnn import OracleMotionTGNN
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    transformer = MiniSpatioTemporalTransformer().to(device)
    transformer.load_state_dict(torch.load(
        project / str(PROJECT_ROOT / 'spatiotemporal_transformer_outputs') / "best_spatiotemporal_transformer.pt",
        map_location=device,
    ))
    transformer.eval()

    joint_state = torch.load(
        PROJECT_ROOT / 'joint_motion_tgnn_outputs' / "best_joint_motion_tgnn.pt",
        map_location=device,
    )
    motion_net = DgiMotionNet().to(device)
    motion_net.load_state_dict({
        key.removeprefix("motion_net."): value
        for key, value in joint_state.items()
        if key.startswith("motion_net.")
    })
    reconstructor = OracleMotionTGNN().to(device)
    reconstructor.load_state_dict({
        key.removeprefix("reconstructor."): value
        for key, value in joint_state.items()
        if key.startswith("reconstructor.")
    })
    motion_net.eval()
    reconstructor.eval()

    patterns = {
        count: make_patterns(count, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
        for count in args.pattern_counts
    }
    scenarios = ("speed_1_2", "speed_2_4", "speed_4_6", "speed_6_8", "acceleration")
    methods = (
        "Transformer",
        "GraphZero",
        "GraphEstimatedGlobal",
        "GraphEstimatedPair",
        "GraphOraclePair",
    )
    detail: list[dict] = []
    summary: list[dict] = []
    motion_summary: list[dict] = []
    manifests = {}
    start = time.time()

    for scenario_index, scenario in enumerate(scenarios):
        dataset = ControlledMovingMNIST(
            project / "data", args.sequences, SEED + 1000 + scenario_index, scenario
        )
        manifests[scenario] = dataset.manifest()
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for pattern_count in args.pattern_counts:
            method_rows = {method: [] for method in methods}
            motion_errors = []
            offset = 0
            with torch.inference_mode():
                for target, labels, truth_pair, mnist_indices in loader:
                    target = target.to(device)
                    truth_pair = truth_pair.to(device)
                    coarse = coarse_sequence(target, patterns[pattern_count])
                    estimate_pair = estimated_pair_motion(motion_net, coarse)
                    estimate_global = estimate_pair.mean(1).clamp(-5.0, 5.0)
                    estimate_pair_clamped = estimate_pair.clamp(-5.0, 5.0)
                    zero_pair = torch.zeros_like(truth_pair)
                    predictions = {
                        "Transformer": transformer(coarse),
                        "GraphZero": per_pair_reconstruct(reconstructor, coarse, zero_pair),
                        "GraphEstimatedGlobal": reconstructor(coarse, estimate_global),
                        "GraphEstimatedPair": per_pair_reconstruct(
                            reconstructor, coarse, estimate_pair_clamped
                        ),
                        "GraphOraclePair": per_pair_reconstruct(reconstructor, coarse, truth_pair),
                    }
                    pair_epe = torch.linalg.vector_norm(estimate_pair - truth_pair, dim=-1)
                    for batch_index in range(target.shape[0]):
                        motion_errors.append({
                            "scenario": scenario,
                            "pattern_count": pattern_count,
                            "sequence_id": offset + batch_index,
                            "mnist_test_index": int(mnist_indices[batch_index]),
                            "true_speed_mean": float(torch.linalg.vector_norm(
                                truth_pair[batch_index], dim=-1
                            ).mean()),
                            "estimated_pair_epe_mean": float(pair_epe[batch_index].mean()),
                            "estimated_pair_within_1px_ratio": float(
                                (pair_epe[batch_index] <= 1.0).float().mean()
                            ),
                        })
                    for method, prediction in predictions.items():
                        rows = metric_rows(prediction, target, labels, mnist_indices)
                        for batch_index, row in enumerate(rows):
                            record = {
                                "scenario": scenario,
                                "pattern_count": pattern_count,
                                "sampling_ratio": pattern_count / IMAGE_SIZE**2,
                                "method": method,
                                "sequence_id": offset + batch_index,
                                **row,
                            }
                            detail.append(record)
                            method_rows[method].append(record)
                    offset += target.shape[0]

            for method in methods:
                row = {
                    "scenario": scenario,
                    "pattern_count": pattern_count,
                    "sampling_ratio": pattern_count / IMAGE_SIZE**2,
                    "method": method,
                    **summarize(method_rows[method]),
                }
                summary.append(row)
                print(
                    f"scenario={scenario:12s} patterns={pattern_count:4d} "
                    f"method={method:20s} PSNR={row['psnr_db_mean']:.2f} "
                    f"SSIM={row['ssim_mean']:.3f} FG={row['foreground_psnr_db_mean']:.2f}"
                )
            epe = np.asarray([row["estimated_pair_epe_mean"] for row in motion_errors])
            within = np.asarray([row["estimated_pair_within_1px_ratio"] for row in motion_errors])
            motion_summary.append({
                "scenario": scenario,
                "pattern_count": pattern_count,
                "sequences": len(motion_errors),
                "estimated_pair_epe_mean": float(epe.mean()),
                "estimated_pair_epe_std": float(epe.std(ddof=1)),
                "estimated_pair_within_1px_ratio_mean": float(within.mean()),
            })

    # Paired gains against the Transformer on exactly the same sequences.
    gains = []
    by_key = {
        (row["scenario"], row["pattern_count"], row["method"], row["sequence_id"]): row
        for row in detail
    }
    for scenario in scenarios:
        for pattern_count in args.pattern_counts:
            for method in methods[1:]:
                delta = np.asarray([
                    by_key[(scenario, pattern_count, method, i)]["psnr_db"]
                    - by_key[(scenario, pattern_count, "Transformer", i)]["psnr_db"]
                    for i in range(args.sequences)
                ])
                gains.append({
                    "scenario": scenario,
                    "pattern_count": pattern_count,
                    "method": method,
                    "paired_psnr_gain_mean_db": float(delta.mean()),
                    "paired_psnr_gain_std_db": float(delta.std(ddof=1)),
                    "paired_win_rate": float((delta > 0).mean()),
                })

    save_csv(output / "benchmark_detail.csv", detail)
    save_csv(output / "benchmark_summary.csv", summary)
    save_csv(output / "motion_summary.csv", motion_summary)
    save_csv(output / "paired_gains.csv", gains)
    (output / "data_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "config": {
            "seed": SEED,
            "sequences_per_scenario": args.sequences,
            "pattern_counts": list(args.pattern_counts),
            "scenarios": list(scenarios),
            "checkpoint": "existing JointMotionTGNN; no retraining",
            "purpose": "isolate temporal graph routing and motion-estimation limits",
        },
        "summary": summary,
        "motion_summary": motion_summary,
        "paired_gains": gains,
        "elapsed_seconds": time.time() - start,
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"elapsed_seconds={result['elapsed_seconds']:.1f}")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
