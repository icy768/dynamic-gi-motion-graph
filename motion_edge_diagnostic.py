"""诊断现有 8x8 局部匹配图的跨帧边是否反映真实平移。"""

from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = PROJECT_ROOT


def selected_offsets(model, pooled, batch, frames, dim):
    """复现 MotionAwarePatchTGNN 的最近邻选择，返回每个节点的 patch 位移。"""
    nodes_per_frame = model.grid * model.grid
    features = pooled.reshape(batch, frames, dim, nodes_per_frame).permute(0, 1, 3, 2)
    offsets = []
    for frame in range(frames - 1):
        current, following = features[:, frame], features[:, frame + 1]
        frame_offsets = []
        for node, candidates in enumerate(model.candidates):
            candidate = following[:, candidates]
            distance = (candidate - current[:, node:node + 1]).square().mean(dim=-1)
            selected = torch.as_tensor(candidates, device=pooled.device)[distance.argmin(dim=-1)]
            row, col = divmod(node, model.grid)
            selected_row = torch.div(selected, model.grid, rounding_mode="floor")
            selected_col = selected.remainder(model.grid)
            # 通道顺序是 (dx, dy)，单位为 patch。
            frame_offsets.append(torch.stack((selected_col - col, selected_row - row), dim=-1))
        offsets.append(torch.stack(frame_offsets, dim=1))
    return torch.stack(offsets, dim=1).to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / 'oracle_ablation_outputs' / "motion_edge_diagnostic.json")
    args = parser.parse_args()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))

    from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
    from train_mixed_sampling import coarse_sequence
    from train_motion_tgnn import MotionAwarePatchTGNN
    from train_oracle_motion_tgnn import MovingMNISTWithMotion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionAwarePatchTGNN().to(device)
    model.load_state_dict(torch.load(project / str(PROJECT_ROOT / 'motion_tgnn_outputs') / "best_motion_tgnn.pt", map_location=device))
    model.eval()
    loader = DataLoader(MovingMNISTWithMotion(project / "data", False, 100, SEED + 300), batch_size=4, shuffle=False)
    results = []
    for pattern_count in (128, 256, 512, 1024):
        patterns = make_patterns(pattern_count, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
        all_offsets, all_motion = [], []
        with torch.no_grad():
            for target, _, motion in loader:
                target = target.to(device)
                coarse = coarse_sequence(target, patterns)
                batch, frames, channels, height, width = coarse.shape
                encoded = model.enc2(model.enc1(coarse.reshape(batch * frames, channels, height, width)))
                pooled = F.adaptive_avg_pool2d(encoded, (model.grid, model.grid))
                all_offsets.append(selected_offsets(model, pooled, batch, frames, encoded.shape[1]).cpu())
                all_motion.append(motion)
        offsets = torch.cat(all_offsets)
        motion = torch.cat(all_motion).to(torch.float32)
        estimated_pixels = offsets * (IMAGE_SIZE / model.grid)
        target_pixels = motion[:, None, None, :]
        vector_error = torch.linalg.vector_norm(estimated_pixels - target_pixels, dim=-1)
        zero_error = torch.linalg.vector_norm(target_pixels.expand_as(estimated_pixels), dim=-1)
        results.append({
            "pattern_count": pattern_count,
            "patch_grid": f"{model.grid}x{model.grid}",
            "patch_width_pixels": IMAGE_SIZE / model.grid,
            "true_motion_mean_pixels": motion.mean(dim=0).tolist(),
            "true_motion_speed_mean_pixels": float(torch.linalg.vector_norm(motion, dim=-1).mean()),
            "matched_edge_mean_vector_pixels": estimated_pixels.mean(dim=(0, 1, 2)).tolist(),
            "matched_edge_self_ratio": float((offsets == 0).all(dim=-1).float().mean()),
            "matched_edge_vector_mae_pixels": float(vector_error.mean()),
            "same_position_edge_vector_mae_pixels": float(zero_error.mean()),
        })
    payload = {"description": "在 DGI 输入特征上复现现有 8x8 patch 局部匹配规则；误差越小越好。", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
