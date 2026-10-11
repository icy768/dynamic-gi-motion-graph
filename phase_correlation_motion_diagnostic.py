"""评估低采样 DGI 序列的相位相关全局平移估计。

估计器输入仅为相邻两帧 DGI 图；模拟器真值位移仅用于离线评估。
"""

from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = PROJECT_ROOT
RATES = (128, 256, 512, 1024)
MAX_SHIFT_PIXELS = 5


def phase_correlation_shift(previous, following, max_shift=MAX_SHIFT_PIXELS):
    """以相位相关估计 previous -> following 的整数像素平移，返回 (dx, dy)。"""
    height, width = previous.shape[-2:]
    window = torch.hann_window(height, device=previous.device)[:, None] * torch.hann_window(width, device=previous.device)[None, :]
    first = torch.fft.rfft2(previous * window)
    second = torch.fft.rfft2(following * window)
    # following 是 previous 平移后的图；采用该乘积顺序使峰值符号对应
    # previous -> following 的 (dx, dy)，而不是反向位移。
    cross_power = second * first.conj()
    cross_power = cross_power / cross_power.abs().clamp_min(1e-8)
    correlation = torch.fft.irfft2(cross_power, s=(height, width)).real
    # 仅在本模拟器允许的最大速度范围内选择峰值，避免周期边界的伪峰。
    dy_values = torch.arange(-max_shift, max_shift + 1, device=previous.device)
    dx_values = torch.arange(-max_shift, max_shift + 1, device=previous.device)
    yy, xx = torch.meshgrid(dy_values, dx_values, indexing="ij")
    indices_y = yy.remainder(height)
    indices_x = xx.remainder(width)
    local = correlation[:, indices_y, indices_x]
    best = local.reshape(local.shape[0], -1).argmax(dim=1)
    return torch.stack((xx.reshape(-1)[best], yy.reshape(-1)[best]), dim=1).to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / 'estimated_motion_outputs')
    args = parser.parse_args()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
    from train_mixed_sampling import coarse_sequence
    from train_oracle_motion_tgnn import MovingMNISTWithMotion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(MovingMNISTWithMotion(project / "data", False, 100, SEED + 300), batch_size=10, shuffle=False)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = [], []
    for pattern_count in RATES:
        patterns = make_patterns(pattern_count, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
        for sequence_start, (target, labels, motion) in enumerate(loader):
            target = target.to(device)
            coarse = coarse_sequence(target, patterns)[:, :, 0]
            estimates = phase_correlation_shift(coarse[:, :-1].reshape(-1, IMAGE_SIZE, IMAGE_SIZE), coarse[:, 1:].reshape(-1, IMAGE_SIZE, IMAGE_SIZE)).reshape(target.shape[0], -1, 2).cpu()
            for item in range(target.shape[0]):
                true_motion = motion[item].to(torch.float32)
                for frame in range(estimates.shape[1]):
                    estimate = estimates[item, frame]
                    error = torch.linalg.vector_norm(estimate - true_motion).item()
                    detail.append({"pattern_count": pattern_count, "sampling_ratio": pattern_count / IMAGE_SIZE**2, "sequence_id": sequence_start * loader.batch_size + item, "frame_pair": frame, "label": int(labels[item]), "true_dx": float(true_motion[0]), "true_dy": float(true_motion[1]), "estimated_dx": float(estimate[0]), "estimated_dy": float(estimate[1]), "vector_error_pixels": error, "zero_motion_error_pixels": float(torch.linalg.vector_norm(true_motion))})
        rows = [row for row in detail if row["pattern_count"] == pattern_count]
        error = torch.tensor([row["vector_error_pixels"] for row in rows])
        zero_error = torch.tensor([row["zero_motion_error_pixels"] for row in rows])
        summary.append({"pattern_count": pattern_count, "sampling_ratio": pattern_count / IMAGE_SIZE**2, "pairs": len(rows), "phase_correlation_vector_mae_pixels": float(error.mean()), "phase_correlation_vector_rmse_pixels": float(error.square().mean().sqrt()), "zero_motion_vector_mae_pixels": float(zero_error.mean()), "phase_correlation_better_than_zero_ratio": float((error < zero_error).float().mean()), "within_1_pixel_ratio": float((error <= 1.0).float().mean())})
    with (output_dir / "phase_correlation_detail.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail[0].keys())
        writer.writeheader()
        writer.writerows(detail)
    with (output_dir / "phase_correlation_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    result = {"config": {"seed": SEED, "test_sequences": 100, "frame_pairs_per_sequence": 7, "estimator": "Hann-windowed phase correlation with integer-pixel peak", "max_search_shift_pixels": MAX_SHIFT_PIXELS, "uses_ground_truth_for_estimation": False}, "summary": summary}
    (output_dir / "phase_correlation_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
