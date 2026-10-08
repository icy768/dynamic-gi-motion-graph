import csv, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sources={'Transformer':[Path(r'D:\download\DynGhost-mini-repro\spatiotemporal_transformer_outputs\benchmark_summary.csv'),ROOT/'transformer_seed_20260903'/'benchmark_summary.csv',ROOT/'transformer_seed_20260904'/'benchmark_summary.csv'],'JointMotionTGNN':[ROOT/'joint_motion_tgnn_outputs'/'benchmark_summary.csv',ROOT/'joint_motion_tgnn_seed_20260903'/'benchmark_summary.csv',ROOT/'joint_motion_tgnn_seed_20260904'/'benchmark_summary.csv']}
data={}
for name,paths in sources.items():
 data[name]={}
 for p in paths:
  with p.open(encoding='utf-8-sig') as f:
   for r in csv.DictReader(f):
    if name=='Transformer' and r['method']!='SpatioTemporalTransformer':continue
    data[name].setdefault(int(r['pattern_count']),[]).append(float(r['psnr_mean_db']))
out=[]
for n in (128,256,512,1024):
 t=np.array(data['Transformer'][n]);j=np.array(data['JointMotionTGNN'][n]);out.append({'pattern_count':n,'sampling_ratio':n/4096,'transformer_psnr_mean':float(t.mean()),'transformer_psnr_std':float(t.std(ddof=1)),'joint_tgnn_psnr_mean':float(j.mean()),'joint_tgnn_psnr_std':float(j.std(ddof=1)),'joint_minus_transformer_db':float(j.mean()-t.mean())})
(ROOT/'transformer_vs_joint_repeat_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
