"""Train a fair dynamic-motion protocol for Transformer and graph variants.

The dataset uses deterministic bouncing translations so fast objects remain in
the field of view.  Every model sees identical sequences, speckles, losses and
training updates.  Models:

* MiniSpatioTemporalTransformer
* Zero-motion graph
* Oracle per-pair motion graph (mechanism upper bound)
* Estimated per-pair joint graph (deployable)
"""

from __future__ import annotations

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
DEFAULT_PROJECT = Path(r"D:\download\DynGhost-mini-repro")
DEFAULT_OUT = HERE / "graph_protocol_v2_trained"
SEED = 20260902
IMAGE_SIZE = 64
SEQ_LEN = 8
TRAIN_N, VAL_N, TEST_N = 800, 120, 120
BATCH = 8
RATES = (128, 256)
SPEED_BINS = ((1.0, 2.0), (2.0, 4.0), (4.0, 6.0))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BouncingMovingMNIST(Dataset):
    def __init__(
        self,
        root: Path,
        train: bool,
        source_indices: np.ndarray,
        seed: int,
        max_offset: float = 10.0,
    ) -> None:
        self.mnist = datasets.MNIST(
            root=root, train=train, download=False, transform=transforms.ToTensor()
        )
        self.indices = np.asarray(source_indices, dtype=np.int64)
        self.length = len(self.indices)
        self.max_offset = max_offset
        rng = np.random.default_rng(seed)
        self.start_x = rng.uniform(-max_offset, max_offset, size=self.length)
        self.start_y = rng.uniform(-max_offset, max_offset, size=self.length)
        self.angle = rng.uniform(0.0, 2.0 * math.pi, size=self.length)
        # Balanced speed strata independent of MNIST identity.
        order = rng.permutation(self.length)
        self.speed = np.empty(self.length, dtype=np.float64)
        for rank, sample_index in enumerate(order):
            lo, hi = SPEED_BINS[rank % len(SPEED_BINS)]
            self.speed[sample_index] = rng.uniform(lo, hi)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        image, label = self.mnist[int(self.indices[index])]
        image = F.interpolate(
            image[None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0]
        x = float(self.start_x[index])
        y = float(self.start_y[index])
        vx = float(self.speed[index] * math.cos(self.angle[index]))
        vy = float(self.speed[index] * math.sin(self.angle[index]))
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
            frames.append(F.grid_sample(
                image[None], grid, align_corners=False, padding_mode="zeros"
            )[0])
            if frame_index + 1 < SEQ_LEN:
                if x + vx < -self.max_offset or x + vx > self.max_offset:
                    vx = -vx
                if y + vy < -self.max_offset or y + vy > self.max_offset:
                    vy = -vy
                x += vx
                y += vy
                pair_motion.append((vx, vy))
        return (
            torch.stack(frames),
            int(label),
            torch.tensor(pair_motion, dtype=torch.float32),
            float(self.speed[index]),
            int(self.indices[index]),
        )

    def manifest(self) -> list[dict]:
        return [
            {
                "sequence_id": i,
                "mnist_index": int(self.indices[i]),
                "start_x": float(self.start_x[i]),
                "start_y": float(self.start_y[i]),
                "angle_rad": float(self.angle[i]),
                "speed_pixels_per_frame": float(self.speed[i]),
                "max_offset": float(self.max_offset),
            }
            for i in range(self.length)
        ]


class JointPairGraph(nn.Module):
    def __init__(self, motion_net: nn.Module, reconstructor: nn.Module) -> None:
        super().__init__()
        self.motion_net = motion_net
        self.reconstructor = reconstructor

    def estimate_pair(self, coarse: torch.Tensor) -> torch.Tensor:
        batch, frames, _, height, width = coarse.shape
        pairs = torch.cat((coarse[:, :-1], coarse[:, 1:]), dim=2)
        pairs = pairs.reshape(batch * (frames - 1), 2, height, width)
        return self.motion_net(pairs).reshape(batch, frames - 1, 2)

    def forward(self, coarse: torch.Tensor):
        from evaluate_graph_protocol_v2 import per_pair_reconstruct
        pair = self.estimate_pair(coarse).clamp(-8.0, 8.0)
        return per_pair_reconstruct(self.reconstructor, coarse, pair), pair


def make_splits(project: Path):
    rng = np.random.default_rng(SEED + 700)
    train_perm = rng.permutation(60000)
    test_perm = rng.permutation(10000)
    train = BouncingMovingMNIST(
        project / "data", True, train_perm[:TRAIN_N], SEED + 701
    )
    val = BouncingMovingMNIST(
        project / "data", False, test_perm[:VAL_N], SEED + 702
    )
    test = BouncingMovingMNIST(
        project / "data", False, test_perm[VAL_N:VAL_N + TEST_N], SEED + 703
    )
    assert set(val.indices.tolist()).isdisjoint(set(test.indices.tolist()))
    return train, val, test


def speed_bin_name(speed: float) -> str:
    for lo, hi in SPEED_BINS:
        if lo <= speed <= hi:
            return f"{lo:g}-{hi:g}"
    return "outside"


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    transformer: nn.Module,
    zero_graph: nn.Module,
    oracle_graph: nn.Module,
    joint_graph: JointPairGraph,
    loader: DataLoader,
    patterns: torch.Tensor,
    device: torch.device,
) -> list[dict]:
    from evaluate_graph_protocol_v2 import metric_rows, per_pair_reconstruct
    from train_mixed_sampling import coarse_sequence

    for model in (transformer, zero_graph, oracle_graph, joint_graph):
        model.eval()
    rows = []
    offset = 0
    with torch.inference_mode():
        for target, labels, truth_pair, speeds, mnist_indices in loader:
            target = target.to(device)
            truth_pair = truth_pair.to(device)
            coarse = coarse_sequence(target, patterns)
            estimated_pred, estimated_pair = joint_graph(coarse)
            predictions = {
                "Transformer": transformer(coarse),
                "GraphZero": per_pair_reconstruct(
                    zero_graph, coarse, torch.zeros_like(truth_pair)
                ),
                "GraphOraclePair": per_pair_reconstruct(oracle_graph, coarse, truth_pair),
                "GraphEstimatedPair": estimated_pred,
            }
            pair_epe = torch.linalg.vector_norm(estimated_pair - truth_pair, dim=-1).mean(1)
            for method, prediction in predictions.items():
                batch_rows = metric_rows(prediction, target, labels, mnist_indices)
                for i, row in enumerate(batch_rows):
                    rows.append({
                        "method": method,
                        "sequence_id": offset + i,
                        "speed_pixels_per_frame": float(speeds[i]),
                        "speed_bin": speed_bin_name(float(speeds[i])),
                        "motion_epe_pixels": float(pair_epe[i]) if method == "GraphEstimatedPair" else "",
                        **row,
                    })
            offset += target.shape[0]
    return rows


def validation_mse(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    bank: dict[int, torch.Tensor],
    device: torch.device,
) -> float:
    from evaluate_graph_protocol_v2 import per_pair_reconstruct
    from train_mixed_sampling import coarse_sequence

    model.eval()
    values = []
    with torch.inference_mode():
        for pattern_count in RATES:
            for target, _, truth_pair, _, _ in loader:
                target = target.to(device)
                truth_pair = truth_pair.to(device)
                coarse = coarse_sequence(target, bank[pattern_count])
                if model_name == "Transformer":
                    prediction = model(coarse)
                elif model_name == "GraphZero":
                    prediction = per_pair_reconstruct(model, coarse, torch.zeros_like(truth_pair))
                elif model_name == "GraphOraclePair":
                    prediction = per_pair_reconstruct(model, coarse, truth_pair)
                elif model_name == "GraphEstimatedPair":
                    prediction, _ = model(coarse)
                else:
                    raise ValueError(model_name)
                values.extend((prediction - target).square().mean((1, 2, 3, 4)).tolist())
    return float(np.mean(values))


def train_motion_net(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bank: dict[int, torch.Tensor],
    device: torch.device,
    epochs: int,
    checkpoint: Path,
) -> list[dict]:
    from train_mixed_sampling import coarse_sequence

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = count = 0
        for batch_index, (target, _, truth_pair, _, _) in enumerate(train_loader):
            target = target.to(device)
            truth_pair = truth_pair.to(device)
            coarse = coarse_sequence(target, bank[RATES[(batch_index + epoch - 1) % len(RATES)]])
            batch, frames = target.shape[:2]
            pairs = torch.cat((coarse[:, :-1], coarse[:, 1:]), dim=2)
            prediction = model(pairs.reshape(batch * (frames - 1), 2, IMAGE_SIZE, IMAGE_SIZE))
            prediction = prediction.reshape(batch, frames - 1, 2)
            loss = F.smooth_l1_loss(prediction, truth_pair)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * batch
            count += batch

        model.eval()
        errors = []
        with torch.inference_mode():
            for target, _, truth_pair, _, _ in val_loader:
                target = target.to(device)
                truth_pair = truth_pair.to(device)
                coarse = coarse_sequence(target, bank[128])
                batch, frames = target.shape[:2]
                pairs = torch.cat((coarse[:, :-1], coarse[:, 1:]), dim=2)
                estimate = model(pairs.reshape(batch * (frames - 1), 2, IMAGE_SIZE, IMAGE_SIZE))
                estimate = estimate.reshape(batch, frames - 1, 2)
                errors.extend(torch.linalg.vector_norm(estimate - truth_pair, dim=-1).flatten().tolist())
        val_epe = float(np.mean(errors))
        history.append({"epoch": epoch, "train_loss": total / count, "val_epe": val_epe})
        print(f"MotionNet epoch={epoch:02d}/{epochs} loss={total/count:.5f} val_epe={val_epe:.3f}")
        if val_epe < best:
            best = val_epe
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return history


def train_reconstructor(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bank: dict[int, torch.Tensor],
    device: torch.device,
    epochs: int,
    checkpoint: Path,
    motion_weight: float = 0.05,
) -> list[dict]:
    from evaluate_graph_protocol_v2 import per_pair_reconstruct
    from train_frame_cnn import reconstruction_loss
    from train_mixed_sampling import coarse_sequence, temporal_error

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = rec_total = motion_total = count = 0.0
        for batch_index, (target, _, truth_pair, _, _) in enumerate(train_loader):
            target = target.to(device)
            truth_pair = truth_pair.to(device)
            coarse = coarse_sequence(target, bank[RATES[(batch_index + epoch - 1) % len(RATES)]])
            if model_name == "Transformer":
                prediction = model(coarse)
                motion_loss = torch.zeros((), device=device)
            elif model_name == "GraphZero":
                prediction = per_pair_reconstruct(model, coarse, torch.zeros_like(truth_pair))
                motion_loss = torch.zeros((), device=device)
            elif model_name == "GraphOraclePair":
                prediction = per_pair_reconstruct(model, coarse, truth_pair)
                motion_loss = torch.zeros((), device=device)
            elif model_name == "GraphEstimatedPair":
                prediction, estimate_pair = model(coarse)
                motion_loss = F.smooth_l1_loss(estimate_pair, truth_pair)
            else:
                raise ValueError(model_name)
            flat_target = target.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            rec_loss = reconstruction_loss(prediction.reshape_as(flat_target), flat_target)
            rec_loss = rec_loss + 0.15 * temporal_error(prediction, target)
            loss = rec_loss + motion_weight * motion_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = target.shape[0]
            total += loss.item() * batch_size
            rec_total += rec_loss.item() * batch_size
            motion_total += motion_loss.item() * batch_size
            count += batch_size
        val_mse = validation_mse(model_name, model, val_loader, bank, device)
        history.append({
            "epoch": epoch,
            "train_loss": total / count,
            "reconstruction_loss": rec_total / count,
            "motion_loss": motion_total / count,
            "val_mse": val_mse,
        })
        print(
            f"{model_name} epoch={epoch:02d}/{epochs} loss={total/count:.5f} "
            f"motion={motion_total/count:.4f} val_mse={val_mse:.6f}"
        )
        if val_mse < best:
            best = val_mse
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--motion-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    args = parser.parse_args()

    project = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(project), str(HERE)]
    seed_all(SEED)

    from mini_repro_dynghost import make_patterns
    from train_dgi_motion_estimator import DgiMotionNet
    from train_oracle_motion_tgnn import OracleMotionTGNN
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    train_set, val_set, test_set = make_splits(project)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
    bank = {
        count: make_patterns(count, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
        for count in RATES
    }
    manifests = {
        "train": train_set.manifest(),
        "validation": val_set.manifest(),
        "test": test_set.manifest(),
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    start = time.time()
    histories = {}
    checkpoints = {}

    seed_all(SEED)
    transformer = MiniSpatioTemporalTransformer().to(device)
    checkpoints["Transformer"] = output / "best_transformer.pt"
    histories["Transformer"] = train_reconstructor(
        "Transformer", transformer, train_loader, val_loader, bank, device,
        args.epochs, checkpoints["Transformer"]
    )

    seed_all(SEED)
    zero_graph = OracleMotionTGNN().to(device)
    checkpoints["GraphZero"] = output / "best_zero_graph.pt"
    histories["GraphZero"] = train_reconstructor(
        "GraphZero", zero_graph, train_loader, val_loader, bank, device,
        args.epochs, checkpoints["GraphZero"]
    )

    seed_all(SEED)
    oracle_graph = OracleMotionTGNN().to(device)
    checkpoints["GraphOraclePair"] = output / "best_oracle_pair_graph.pt"
    histories["GraphOraclePair"] = train_reconstructor(
        "GraphOraclePair", oracle_graph, train_loader, val_loader, bank, device,
        args.epochs, checkpoints["GraphOraclePair"]
    )

    seed_all(SEED)
    motion_net = DgiMotionNet().to(device)
    motion_checkpoint = output / "best_pair_motion_net.pt"
    histories["MotionNet"] = train_motion_net(
        motion_net, train_loader, val_loader, bank, device,
        args.motion_epochs, motion_checkpoint
    )
    joint_graph = JointPairGraph(motion_net, OracleMotionTGNN().to(device)).to(device)
    checkpoints["GraphEstimatedPair"] = output / "best_estimated_pair_graph.pt"
    histories["GraphEstimatedPair"] = train_reconstructor(
        "GraphEstimatedPair", joint_graph, train_loader, val_loader, bank, device,
        args.epochs, checkpoints["GraphEstimatedPair"]
    )

    detail = []
    for pattern_count in RATES:
        rows = evaluate(
            transformer, zero_graph, oracle_graph, joint_graph,
            test_loader, bank[pattern_count], device
        )
        for row in rows:
            detail.append({
                "pattern_count": pattern_count,
                "sampling_ratio": pattern_count / IMAGE_SIZE**2,
                **row,
            })

    summary = []
    methods = ("Transformer", "GraphZero", "GraphOraclePair", "GraphEstimatedPair")
    for pattern_count in RATES:
        for speed_bin in ("1-2", "2-4", "4-6", "all"):
            for method in methods:
                selected = [
                    row for row in detail
                    if row["pattern_count"] == pattern_count
                    and row["method"] == method
                    and (speed_bin == "all" or row["speed_bin"] == speed_bin)
                ]
                result = {
                    "pattern_count": pattern_count,
                    "speed_bin": speed_bin,
                    "method": method,
                    "sequences": len(selected),
                }
                for key in ("psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error"):
                    values = np.asarray([row[key] for row in selected], dtype=np.float64)
                    result[f"{key}_mean"] = float(values.mean())
                    result[f"{key}_std"] = float(values.std(ddof=1))
                if method == "GraphEstimatedPair":
                    epe = np.asarray([float(row["motion_epe_pixels"]) for row in selected])
                    result["motion_epe_pixels_mean"] = float(epe.mean())
                else:
                    result["motion_epe_pixels_mean"] = ""
                summary.append(result)
                print(
                    f"patterns={pattern_count:3d} speed={speed_bin:3s} method={method:18s} "
                    f"PSNR={result['psnr_db_mean']:.2f} SSIM={result['ssim_mean']:.3f}"
                )

    gains = []
    index = {
        (row["pattern_count"], row["method"], row["sequence_id"]): row
        for row in detail
    }
    for pattern_count in RATES:
        for method in methods[1:]:
            for speed_bin in ("1-2", "2-4", "4-6", "all"):
                ids = [
                    row["sequence_id"] for row in detail
                    if row["pattern_count"] == pattern_count
                    and row["method"] == "Transformer"
                    and (speed_bin == "all" or row["speed_bin"] == speed_bin)
                ]
                delta = np.asarray([
                    index[(pattern_count, method, sequence_id)]["psnr_db"]
                    - index[(pattern_count, "Transformer", sequence_id)]["psnr_db"]
                    for sequence_id in ids
                ])
                gains.append({
                    "pattern_count": pattern_count,
                    "speed_bin": speed_bin,
                    "method": method,
                    "paired_psnr_gain_mean_db": float(delta.mean()),
                    "paired_psnr_gain_std_db": float(delta.std(ddof=1)),
                    "paired_win_rate": float((delta > 0).mean()),
                })

    save_csv(output / "benchmark_detail.csv", detail)
    save_csv(output / "benchmark_summary.csv", summary)
    save_csv(output / "paired_gains.csv", gains)
    for name, history in histories.items():
        save_csv(output / f"history_{name}.csv", history)
    result = {
        "config": {
            "seed": SEED,
            "train_sequences": TRAIN_N,
            "validation_sequences": VAL_N,
            "test_sequences": TEST_N,
            "rates": list(RATES),
            "epochs": args.epochs,
            "motion_epochs": args.motion_epochs,
            "motion": "per-pair bouncing translation within +/-10 pixels",
            "speed_bins": SPEED_BINS,
            "data_split": "disjoint MNIST IDs for validation and test",
        },
        "parameters": {
            "Transformer": sum(p.numel() for p in transformer.parameters()),
            "GraphZero": sum(p.numel() for p in zero_graph.parameters()),
            "GraphOraclePair": sum(p.numel() for p in oracle_graph.parameters()),
            "GraphEstimatedPair": sum(p.numel() for p in joint_graph.parameters()),
        },
        "summary": summary,
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
