"""同架构 Oracle Motion Graph 的零位移消融实验。

训练时仅把已知位移替换为 (0, 0)，保留数据、网络、损失、采样率轮换、
随机种子和训练轮数。该模型代表“同位置时序边”，用于分离正确运动边带来的收益。
"""

from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / 'oracle_ablation_outputs')
    return parser.parse_args()


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"项目目录不存在：{project_root}")
    sys.path.insert(0, str(project_root))

    from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
    from train_frame_cnn import reconstruction_loss
    from train_mixed_sampling import RATES, coarse_sequence, stats, temporal_error
    from train_oracle_motion_tgnn import (
        BATCH, LR, TEST_N, TRAIN_N, VAL_N, MovingMNISTWithMotion,
        OracleMotionTGNN, evaluate_oracle, measure_oracle_efficiency, set_seed,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("ablation=zero_motion (same-position temporal edges)")
    bank = {
        count: make_patterns(count, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
        for count in RATES
    }
    train_loader = DataLoader(
        MovingMNISTWithMotion(project_root / "data", True, TRAIN_N, SEED + 200),
        batch_size=BATCH, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        MovingMNISTWithMotion(project_root / "data", False, VAL_N, SEED + 201),
        batch_size=BATCH, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        MovingMNISTWithMotion(project_root / "data", False, TEST_N, SEED + 300),
        batch_size=4, shuffle=False, num_workers=0,
    )
    model = OracleMotionTGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    checkpoint = output_dir / "best_zero_motion_tgnn.pt"
    history, best = [], float("inf")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch_index, (target, _, motion) in enumerate(train_loader):
            target = target.to(device)
            zero_motion = torch.zeros_like(motion, device=device)
            pattern_count = RATES[(batch_index + epoch - 1) % len(RATES)]
            with torch.no_grad():
                coarse = coarse_sequence(target, bank[pattern_count])
            prediction = model(coarse, zero_motion)
            flat_target = target.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            loss = reconstruction_loss(prediction.reshape_as(flat_target), flat_target)
            loss = loss + 0.15 * temporal_error(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * target.shape[0]
            sample_count += target.shape[0]

        values = []
        for pattern_count in RATES:
            for target, labels, motion in val_loader:
                target = target.to(device)
                zero_motion = torch.zeros_like(motion, device=device)
                coarse = coarse_sequence(target, bank[pattern_count])
                prediction = model(coarse, zero_motion)
                values.extend((prediction - target).square().mean((1, 2, 3, 4)).tolist())
        val_mse = float(np.mean(values))
        row = {"epoch": epoch, "train_loss": total_loss / sample_count, "val_mse": val_mse}
        history.append(row)
        print(f"epoch={epoch:02d}/{args.epochs} train_loss={row['train_loss']:.6f} val_mse={val_mse:.6f}")
        if val_mse < best:
            best = val_mse
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    detail, summary = [], []
    for pattern_count in RATES:
        rows = []
        model.eval()
        with torch.no_grad():
            for target, labels, motion in test_loader:
                target = target.to(device)
                zero_motion = torch.zeros_like(motion, device=device)
                coarse = coarse_sequence(target, bank[pattern_count])
                prediction = model(coarse, zero_motion)
                batch, frames, channels, height, width = target.shape
                mse = (prediction - target).square().mean((1, 2, 3, 4))
                psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
                from pytorch_msssim import ssim
                frame_ssim = ssim(prediction.reshape(batch * frames, channels, height, width), target.reshape(batch * frames, channels, height, width), data_range=1.0, size_average=False).reshape(batch, frames).mean(1)
                difference_error = ((prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean((1, 2, 3, 4))
                for index in range(batch):
                    rows.append({"label": int(labels[index]), "mse": float(mse[index]), "psnr_db": float(psnr[index]), "ssim": float(frame_ssim[index]), "temporal_difference_error": float(difference_error[index])})
        for sequence_id, row in enumerate(rows):
            detail.append({"pattern_count": pattern_count, "sampling_ratio": pattern_count / IMAGE_SIZE**2, "method": "ZeroMotionTGNN", "sequence_id": sequence_id, **row})
        mse_mean, mse_std = stats(rows, "mse")
        psnr_mean, psnr_std = stats(rows, "psnr_db")
        ssim_mean, ssim_std = stats(rows, "ssim")
        temporal_mean, temporal_std = stats(rows, "temporal_difference_error")
        result = {"pattern_count": pattern_count, "sampling_ratio": pattern_count / IMAGE_SIZE**2, "method": "ZeroMotionTGNN", "mse_mean": mse_mean, "mse_std": mse_std, "psnr_mean_db": psnr_mean, "psnr_std_db": psnr_std, "ssim_mean": ssim_mean, "ssim_std": ssim_std, "temporal_difference_error_mean": temporal_mean, "temporal_difference_error_std": temporal_std}
        summary.append(result)
        print(f"patterns={pattern_count:4d} ZeroMotionTGNN PSNR={psnr_mean:.2f}+/-{psnr_std:.2f} SSIM={ssim_mean:.3f} temp_err={temporal_mean:.4f}")

    sample_target, _, sample_motion = next(iter(test_loader))
    sample = coarse_sequence(sample_target[:1].to(device), bank[256])
    efficiency = measure_oracle_efficiency(model, sample, torch.zeros_like(sample_motion[:1], device=device), device)
    oracle_results = json.loads((project_root / str(PROJECT_ROOT / 'oracle_motion_tgnn_outputs') / "results.json").read_text(encoding="utf-8"))
    oracle_summary = oracle_results["summary"]
    comparison = []
    for baseline, oracle in zip(summary, oracle_summary):
        comparison.append({"pattern_count": baseline["pattern_count"], "sampling_ratio": baseline["sampling_ratio"], "zero_motion_psnr_db": baseline["psnr_mean_db"], "oracle_motion_psnr_db": oracle["psnr_mean_db"], "oracle_minus_zero_psnr_db": oracle["psnr_mean_db"] - baseline["psnr_mean_db"], "zero_motion_ssim": baseline["ssim_mean"], "oracle_motion_ssim": oracle["ssim_mean"], "oracle_minus_zero_ssim": oracle["ssim_mean"] - baseline["ssim_mean"], "zero_motion_temporal_error": baseline["temporal_difference_error_mean"], "oracle_motion_temporal_error": oracle["temporal_difference_error_mean"], "oracle_minus_zero_temporal_error": oracle["temporal_difference_error_mean"] - baseline["temporal_difference_error_mean"]})
    save_csv(output_dir / "training_history.csv", history)
    save_csv(output_dir / "benchmark_detail.csv", detail)
    save_csv(output_dir / "benchmark_summary.csv", summary)
    save_csv(output_dir / "oracle_vs_zero_motion.csv", comparison)
    results = {"config": {"seed": SEED, "rates": list(RATES), "train_sequences": TRAIN_N, "val_sequences": VAL_N, "test_sequences": TEST_N, "epochs": args.epochs, "feature_grid": "16x16", "graph": "four-neighbor spatial edges plus zero-displacement temporal edges", "ablation": "same OracleMotionTGNN architecture with simulator motion replaced by zero"}, "history": history, "summary": summary, "oracle_comparison": comparison, "parameters": sum(parameter.numel() for parameter in model.parameters()), "efficiency": efficiency, "elapsed_seconds": time.time() - start}
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
