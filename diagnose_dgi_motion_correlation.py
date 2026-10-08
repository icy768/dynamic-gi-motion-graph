"""Check whether adjacent DGI frames retain recoverable translation information."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_graph_protocol_v2 import IMAGE_SIZE, RATES, SEED, make_splits, seed_all


PROJECT = Path(r"D:\download\DynGhost-mini-repro")


def candidate_motion(device: torch.device, radius: int = 7) -> torch.Tensor:
    values = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=1)


def correlation_motion(pairs: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Integer translation from normalized cross-correlation over a bounded window."""
    from train_oracle_motion_tgnn import warp_feature

    previous, following = pairs[:, :1], pairs[:, 1:]
    # Remove DGI offset/scale so the score measures structure rather than brightness.
    previous = (previous - previous.mean((-2, -1), keepdim=True)) / (
        previous.std((-2, -1), keepdim=True) + 1e-5
    )
    following = (following - following.mean((-2, -1), keepdim=True)) / (
        following.std((-2, -1), keepdim=True) + 1e-5
    )
    batch, _, height, width = previous.shape
    count = candidates.shape[0]
    expanded = previous[:, None].expand(batch, count, 1, height, width)
    expanded = expanded.reshape(batch * count, 1, height, width)
    dx = candidates[:, 0][None].expand(batch, count).reshape(-1)
    dy = candidates[:, 1][None].expand(batch, count).reshape(-1)
    warped = warp_feature(expanded, dx, dy).reshape(batch, count, height, width)
    reference = following[:, None, 0]
    score = (warped * reference).mean((-2, -1))
    return candidates[score.argmax(1)]


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(Path(__file__).resolve().parent)]
    from mini_repro_dynghost import make_patterns
    from train_mixed_sampling import coarse_sequence

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test = make_splits(PROJECT)
    loader = DataLoader(test, batch_size=4, shuffle=False)
    candidates = candidate_motion(device)
    for rate in RATES:
        patterns = make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
        errors = []
        target_errors = []
        estimates = []
        truths = []
        with torch.inference_mode():
            for target, _, truth, _, _ in loader:
                target, truth = target.to(device), truth.to(device)
                coarse = coarse_sequence(target, patterns)
                batch, frames = coarse.shape[:2]
                pairs = torch.cat((coarse[:, :-1], coarse[:, 1:]), dim=2)
                pair_flat = pairs.reshape(batch * (frames - 1), 2, IMAGE_SIZE, IMAGE_SIZE)
                estimate = correlation_motion(pair_flat, candidates).reshape(batch, frames - 1, 2)
                target_pairs = torch.cat((target[:, :-1], target[:, 1:]), dim=2)
                target_flat = target_pairs.reshape(batch * (frames - 1), 2, IMAGE_SIZE, IMAGE_SIZE)
                target_estimate = correlation_motion(target_flat, candidates).reshape(batch, frames - 1, 2)
                errors.extend(torch.linalg.vector_norm(estimate - truth, dim=-1).flatten().cpu().tolist())
                target_errors.extend(
                    torch.linalg.vector_norm(target_estimate - truth, dim=-1).flatten().cpu().tolist()
                )
                estimates.append(estimate.cpu())
                truths.append(truth.cpu())
        errors_np = np.asarray(errors)
        estimate_np = torch.cat(estimates).numpy().reshape(-1, 2)
        truth_np = torch.cat(truths).numpy().reshape(-1, 2)
        component_corr = [np.corrcoef(estimate_np[:, i], truth_np[:, i])[0, 1] for i in range(2)]
        print({
            "patterns": rate,
            "mean_epe": float(errors_np.mean()),
            "median_epe": float(np.median(errors_np)),
            "within_1px": float(np.mean(errors_np <= 1.0)),
            "corr_dx": float(component_corr[0]),
            "corr_dy": float(component_corr[1]),
            "ground_truth_frame_mean_epe": float(np.mean(target_errors)),
        })


if __name__ == "__main__":
    main()
