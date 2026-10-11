
from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = [PROJECT_ROOT / 'joint_motion_tgnn_outputs', ROOT / 'joint_motion_tgnn_seed_20260903', ROOT / 'joint_motion_tgnn_seed_20260904']
rows = []
for run in RUNS:
    with (run / 'benchmark_summary.csv').open(encoding='utf-8-sig') as f:
        for item in csv.DictReader(f):
            rows.append({'run': run.name, **item})
summary = []
for count in (128, 256, 512, 1024):
    group = [r for r in rows if int(r['pattern_count']) == count]
    result = {'pattern_count': count, 'sampling_ratio': count / 4096, 'runs': len(group)}
    for key in ('psnr_mean_db', 'ssim_mean', 'temporal_difference_error_mean', 'motion_vector_error_pixels_mean'):
        values = np.array([float(r[key]) for r in group])
        result[f'{key}_across_runs_mean'] = float(values.mean())
        result[f'{key}_across_runs_std'] = float(values.std(ddof=1))
    summary.append(result)
out = ROOT / 'joint_motion_tgnn_repeat_summary.json'
out.write_text(json.dumps({'runs': [str(r) for r in RUNS], 'summary': summary}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
