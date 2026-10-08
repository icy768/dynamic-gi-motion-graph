"""Train a position-preserving motion head on adjacent DGI frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_graph_protocol_v2 import (
    IMAGE_SIZE,
    RATES,
    SEED,
    make_splits,
    seed_all,
    train_motion_net,
)


PROJECT = Path(r"D:\download\DynGhost-mini-repro")


class SpatialMotionNet(nn.Module):
    """Preserve a coarse spatial grid instead of pooling translation away."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, 5, stride=2, padding=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv2d(24, 40, 3, stride=2, padding=1),
            nn.GroupNorm(5, 40),
            nn.GELU(),
            nn.Conv2d(40, 48, 3, stride=2, padding=1),
            nn.GroupNorm(6, 48),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 192),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        first, second = pairs[:, :1], pairs[:, 1:]
        first = (first - first.mean((-2, -1), keepdim=True)) / (
            first.std((-2, -1), keepdim=True) + 1e-5
        )
        second = (second - second.mean((-2, -1), keepdim=True)) / (
            second.std((-2, -1), keepdim=True) + 1e-5
        )
        features = torch.cat((first, second, second - first, first * second), dim=1)
        return self.head(self.features(features))


def evaluate_motion(model, loader, bank, device):
    from train_mixed_sampling import coarse_sequence

    model.eval()
    summary = []
    with torch.inference_mode():
        for rate in RATES:
            errors = []
            estimates = []
            truths = []
            for target, _, truth, _, _ in loader:
                target, truth = target.to(device), truth.to(device)
                coarse = coarse_sequence(target, bank[rate])
                batch, frames = coarse.shape[:2]
                pairs = torch.cat((coarse[:, :-1], coarse[:, 1:]), dim=2)
                estimate = model(
                    pairs.reshape(batch * (frames - 1), 2, IMAGE_SIZE, IMAGE_SIZE)
                ).reshape(batch, frames - 1, 2)
                errors.extend(torch.linalg.vector_norm(estimate - truth, dim=-1).flatten().cpu().tolist())
                estimates.append(estimate.cpu())
                truths.append(truth.cpu())
            error = np.asarray(errors)
            estimate_np = torch.cat(estimates).reshape(-1, 2).numpy()
            truth_np = torch.cat(truths).reshape(-1, 2).numpy()
            summary.append({
                "pattern_count": rate,
                "mean_epe": float(error.mean()),
                "median_epe": float(np.median(error)),
                "within_1px_ratio": float(np.mean(error <= 1.0)),
                "corr_dx": float(np.corrcoef(estimate_np[:, 0], truth_np[:, 0])[0, 1]),
                "corr_dy": float(np.corrcoef(estimate_np[:, 1], truth_np[:, 1])[0, 1]),
            })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("spatial_motion_outputs"))
    args = parser.parse_args()
    sys.path[:0] = [str(PROJECT), str(Path(__file__).resolve().parent)]
    from mini_repro_dynghost import make_patterns

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train, val, test = make_splits(PROJECT)
    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    val_loader = DataLoader(val, batch_size=8, shuffle=False)
    test_loader = DataLoader(test, batch_size=8, shuffle=False)
    bank = {rate: make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device) for rate in RATES}
    model = SpatialMotionNet().to(device)
    checkpoint = output / "best_spatial_motion_net.pt"
    history = train_motion_net(
        model, train_loader, val_loader, bank, device, args.epochs, checkpoint
    )
    summary = evaluate_motion(model, test_loader, bank, device)
    for row in summary:
        print(row)
    (output / "results.json").write_text(
        json.dumps({"history": history, "summary": summary}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
