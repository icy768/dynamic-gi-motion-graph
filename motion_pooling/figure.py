import json, numpy as np, matplotlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_feasibility import *
from mcgi import mdr_pair, refine_against_canvas
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","Noto Sans CJK SC","WenQuanYi Zen Hei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"]=False
lam=json.load(open(HERE / "lam_128_diff.json"))["best"]
seed,b=200+1,2   # a fast test sequence
base, frames, pos, speed, ch = make_sequence(seed,b)
rng=np.random.default_rng(seed+99991)
As,ys=measure(frames,pos,base,128,"diff",0.0,rng)
single=per_frame_tv(As,ys,lam["single"],300)
dg=np.stack([dgi(As[t]*64,ys[t]) for t in range(T)])
rel=[np.zeros(2)]
for t in range(1,T): rel.append(rel[-1]+mdr_pair(As[t],ys[t],single[t-1]))
p=np.array(rel); p-=p.mean(0)
est,canvas,_=pooled_recon(As,ys,p,lam["all"],300)
for r in range(8):
    shape,origin=canvas_for(p); p=refine_against_canvas(As,ys,canvas,shape,origin,p); p-=p.mean(0)
    est,canvas,_=pooled_recon(As,ys,p,lam["all"],300)
fig,ax=plt.subplots(4,T,figsize=(T*1.5,7.2))
rowsd=[("真值",frames),("DGI",dg),("单帧 TV",single),("8帧凑测量\n(估计运动)",est)]
for i,(name,st) in enumerate(rowsd):
    for t in range(T):
        a=ax[i,t]; a.imshow(np.clip(st[t],0,1),cmap="gray",vmin=0,vmax=1); a.set_xticks([]); a.set_yticks([])
        if i>0: a.set_xlabel(f"{psnr(st[t],frames[t]):.1f} dB",fontsize=8)
        if t==0: a.set_ylabel(name,fontsize=9)
fig.suptitle(f"128 次测量/帧（3.1%），速度 {speed:.1f} 像素/帧",fontsize=11)
plt.tight_layout(); plt.savefig(HERE / "example_reconstruction.png",dpi=130)
print("ok")
