import json, sys
import numpy as np

LABELS = {
    "dgi": "DGI（相关成像，原项目的粗输入）",
    "single": "单帧 TV（只用本帧测量）",
    "win3_oracle": "相邻 3 帧凑测量（真实运动）",
    "all_oracle": "8 帧凑测量（真实运动，上限）",
    "all_motionerr0.1": "8 帧（运动误差 0.1 px）",
    "all_motionerr0.25": "8 帧（运动误差 0.25 px）",
    "all_motionerr0.5": "8 帧（运动误差 0.5 px）",
    "mdr_r0": "8 帧 + 估计运动（初始）",
    "mdr_final": "8 帧 + 估计运动（交替 8 轮）",
}


def boot_ci(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def summarize(path):
    rows = json.loads(open(path).read())
    keys = [k for k in LABELS if k in rows[0]]
    out = {"file": path, "n_sequences": len(rows), "methods": {}}
    for k in keys:
        ent = {}
        for b, name in ((0, "1-2"), (1, "2-4"), (2, "4-6"), (None, "all")):
            sel = [r for r in rows if b is None or r["speed_bin"] == b]
            ps = np.array([np.mean(r[k]["psnr"]) for r in sel])
            ent[name] = float(ps.mean())
        res = np.array([np.mean(r[k]["resid"]) for r in rows])
        gain = np.array([np.mean(r[k]["psnr"]) - np.mean(r["single"]["psnr"]) for r in rows])
        ent["gain_vs_single"] = float(gain.mean())
        ent["gain_ci95"] = [float(x) for x in boot_ci(gain)]
        ent["win_rate_vs_single"] = float((gain > 0).mean())
        ent["meas_residual"] = float(res.mean())
        out["methods"][k] = ent
    if "mdr_epe_rounds" in rows[0]:
        e = np.array([r["mdr_epe_rounds"] for r in rows])
        out["motion_error_px_by_round"] = [float(x) for x in e.mean(0)]
        out["motion_error_px_final_by_speed"] = {
            n: float(np.mean([r["mdr_epe_rounds"][-1] for r in rows if r["speed_bin"] == b]))
            for b, n in ((0, "1-2"), (1, "2-4"), (2, "4-6"))}
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        s = summarize(p)
        print(f"\n### {p}  ({s['n_sequences']} 条序列)")
        print(f"{'方法':<30}{'慢1-2':>8}{'中2-4':>8}{'快4-6':>8}{'全部':>8}{'比单帧':>9}  95%区间        胜率  测量残差")
        for k, e in s["methods"].items():
            print(f"{LABELS[k]:<30}{e['1-2']:8.2f}{e['2-4']:8.2f}{e['4-6']:8.2f}{e['all']:8.2f}"
                  f"{e['gain_vs_single']:+9.2f}  [{e['gain_ci95'][0]:+.2f},{e['gain_ci95'][1]:+.2f}]  "
                  f"{e['win_rate_vs_single']:.2f}  {e['meas_residual']:.3f}")
        if "motion_error_px_by_round" in s:
            print("运动误差(px)按轮:", np.round(s["motion_error_px_by_round"], 3),
                  "最终按速度:", {k: round(v, 3) for k, v in s["motion_error_px_final_by_speed"].items()})
        json.dump(s, open(p.replace(".json", "_summary.json"), "w"), ensure_ascii=False, indent=1)
