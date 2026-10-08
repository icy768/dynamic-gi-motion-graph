"""从相邻 DGI 帧估计全局平移的监督诊断实验。

训练标签来自模拟器，推理输入仅为两帧 DGI。该实验先验证可学习边估计是否可行，
不将真值位移提供给重建网络。
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path(r"D:\download\DynGhost-mini-repro")
RATES = (128, 256, 512, 1024)
SEED = 20260902

class DgiMotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 2))
    def forward(self, pairs):
        return self.head(self.features(pairs))

def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def evaluate(model, loader, bank, device):
    model.eval(); rows = []
    with torch.no_grad():
        for target, labels, motion in loader:
            target = target.to(device)
            for count, patterns in bank.items():
                from train_mixed_sampling import coarse_sequence
                dgi = coarse_sequence(target, patterns)
                batch, frames = dgi.shape[:2]
                pairs = torch.cat((dgi[:, :-1], dgi[:, 1:]), dim=2).reshape(batch * (frames - 1), 2, 64, 64)
                estimate = model(pairs).reshape(batch, frames - 1, 2).cpu()
                for i in range(batch):
                    error = torch.linalg.vector_norm(estimate[i] - motion[i], dim=-1)
                    for frame, value in enumerate(error):
                        rows.append({"pattern_count": count, "sampling_ratio": count / 4096, "sequence_id": len(rows) // (len(bank) * (frames - 1)), "frame_pair": frame, "true_dx": float(motion[i, 0]), "true_dy": float(motion[i, 1]), "estimated_dx": float(estimate[i, frame, 0]), "estimated_dy": float(estimate[i, frame, 1]), "vector_error_pixels": float(value)})
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=HERE / "estimated_motion_outputs")
    args = parser.parse_args(); project = args.project_root.resolve(); sys.path.insert(0, str(project))
    from mini_repro_dynghost import IMAGE_SIZE, make_patterns
    from train_mixed_sampling import coarse_sequence
    from train_oracle_motion_tgnn import MovingMNISTWithMotion
    set_seed(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    bank = {n: make_patterns(n, IMAGE_SIZE, IMAGE_SIZE, SEED).to(device) for n in RATES}
    train = DataLoader(MovingMNISTWithMotion(project / "data", True, 800, SEED + 200), batch_size=8, shuffle=True)
    val = DataLoader(MovingMNISTWithMotion(project / "data", False, 120, SEED + 201), batch_size=8)
    test = DataLoader(MovingMNISTWithMotion(project / "data", False, 100, SEED + 300), batch_size=8)
    model = DgiMotionNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, history, checkpoint = float("inf"), [], output / "best_dgi_motion_net.pt"
    for epoch in range(1, args.epochs + 1):
        model.train(); total = count = 0
        for batch_index, (target, _, motion) in enumerate(train):
            target, motion = target.to(device), motion.to(device)
            dgi = coarse_sequence(target, bank[RATES[(batch_index + epoch - 1) % len(RATES)]])
            batch, frames = dgi.shape[:2]
            pairs = torch.cat((dgi[:, :-1], dgi[:, 1:]), dim=2).reshape(batch * (frames - 1), 2, IMAGE_SIZE, IMAGE_SIZE)
            prediction = model(pairs).reshape(batch, frames - 1, 2)
            loss = F.smooth_l1_loss(prediction, motion[:, None].expand_as(prediction))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += loss.item() * batch; count += batch
        rows = evaluate(model, val, bank, device)
        value = float(np.mean([r["vector_error_pixels"] for r in rows])); history.append({"epoch": epoch, "train_loss": total / count, "val_vector_mae_pixels": value})
        print(f"epoch={epoch:02d}/{args.epochs} train_loss={total/count:.6f} val_motion_mae={value:.3f}")
        if value < best: best = value; torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device)); detail = evaluate(model, test, bank, device)
    summary=[]
    for n in RATES:
        values=np.array([r['vector_error_pixels'] for r in detail if r['pattern_count']==n])
        summary.append({"pattern_count":n,"sampling_ratio":n/4096,"pairs":len(values),"learned_motion_vector_mae_pixels":float(values.mean()),"learned_motion_vector_rmse_pixels":float(np.sqrt(np.mean(values**2))),"within_1_pixel_ratio":float(np.mean(values<=1.0)),"zero_motion_vector_mae_pixels":2.1405608654})
    for row in summary: print(row)
    with (output / "learned_motion_detail.csv").open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=detail[0].keys()); w.writeheader(); w.writerows(detail)
    result={"config":{"seed":SEED,"train_sequences":800,"val_sequences":120,"test_sequences":100,"epochs":args.epochs,"input":"two adjacent DGI frames","label":"simulator global translation; unavailable at inference","purpose":"motion-estimation feasibility diagnostic"},"history":history,"summary":summary,"parameters":sum(p.numel() for p in model.parameters())}
    (output / "learned_motion_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__ == '__main__': main()
