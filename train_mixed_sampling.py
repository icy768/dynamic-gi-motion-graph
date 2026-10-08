import csv, json, math, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
from train_frame_cnn import FrameCNN, reconstruction_loss
from train_temporal_convgru import BidirectionalConvGRU, MovingMNISTSequences
from benchmark_models import dgi_dynamic

RATES = (128, 256, 512, 1024)
TRAIN_N, VAL_N, TEST_N = 800, 120, 100
BATCH, EPOCHS, LR = 8, 10, 1e-3
OUT = "mixed_sampling_outputs"

def coarse_sequence(x, patterns):
    b, t, c, h, w = x.shape
    return dgi_dynamic(x.reshape(b*t, c, h, w), patterns).reshape_as(x)

def frame_forward(model, x):
    b, t, c, h, w = x.shape
    return model(x.reshape(b*t, c, h, w)).reshape_as(x)

def temporal_error(pred, target):
    return ((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean()

@torch.no_grad()
def evaluate(model, kind, loader, patterns, device):
    model.eval(); rows = []
    for target, labels in loader:
        target = target.to(device); coarse = coarse_sequence(target, patterns)
        pred = frame_forward(model, coarse) if kind == "FrameCNN" else model(coarse)
        b,t,c,h,w = target.shape
        mse = (pred-target).square().mean((1,2,3,4))
        psnr = 10*torch.log10(1/mse.clamp_min(1e-12))
        sf = ssim(pred.reshape(b*t,c,h,w), target.reshape(b*t,c,h,w),
                  data_range=1.0, size_average=False).reshape(b,t).mean(1)
        te = ((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean((1,2,3,4))
        for i in range(b):
            rows.append(dict(label=int(labels[i]), mse=float(mse[i]), psnr_db=float(psnr[i]),
                             ssim=float(sf[i]), temporal_difference_error=float(te[i])))
    return rows

def stats(rows, key):
    v=np.array([r[key] for r in rows]); return float(v.mean()), float(v.std(ddof=1))

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    os.makedirs(OUT, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"device={device}")
    bank={m:make_patterns(m,IMAGE_SIZE,IMAGE_SIZE,seed=SEED).to(device) for m in RATES}
    train=DataLoader(MovingMNISTSequences("./data",True,TRAIN_N,SEED+200),batch_size=BATCH,shuffle=True,num_workers=0)
    val=DataLoader(MovingMNISTSequences("./data",False,VAL_N,SEED+201),batch_size=BATCH,shuffle=False,num_workers=0)
    test=DataLoader(MovingMNISTSequences("./data",False,TEST_N,SEED+300),batch_size=4,shuffle=False,num_workers=0)
    fm,tm=FrameCNN().to(device),BidirectionalConvGRU().to(device)
    fo,to=torch.optim.Adam(fm.parameters(),lr=LR),torch.optim.Adam(tm.parameters(),lr=LR)
    history=[]; best_f=best_t=float("inf"); start=time.time()
    for epoch in range(1,EPOCHS+1):
        fm.train(); tm.train(); fs=ts=n=0
        for bi,(target,_) in enumerate(train):
            target=target.to(device); m=RATES[(bi+epoch-1)%len(RATES)]
            with torch.no_grad(): coarse=coarse_sequence(target,bank[m])
            flat=target.reshape(-1,1,IMAGE_SIZE,IMAGE_SIZE)
            fp=frame_forward(fm,coarse); fl=reconstruction_loss(fp.reshape_as(flat),flat)
            fo.zero_grad(set_to_none=True); fl.backward(); torch.nn.utils.clip_grad_norm_(fm.parameters(),1); fo.step()
            tp=tm(coarse); tl=reconstruction_loss(tp.reshape_as(flat),flat)+.15*temporal_error(tp,target)
            to.zero_grad(set_to_none=True); tl.backward(); torch.nn.utils.clip_grad_norm_(tm.parameters(),1); to.step()
            fs+=fl.item()*target.shape[0]; ts+=tl.item()*target.shape[0]; n+=target.shape[0]
        fv=[]; tv=[]
        for m in RATES:
            fv += [r["mse"] for r in evaluate(fm,"FrameCNN",val,bank[m],device)]
            tv += [r["mse"] for r in evaluate(tm,"BiConvGRU",val,bank[m],device)]
        fvm,tvm=float(np.mean(fv)),float(np.mean(tv))
        history.append(dict(epoch=epoch,frame_train_loss=fs/n,temporal_train_loss=ts/n,
                            frame_val_mse=fvm,temporal_val_mse=tvm))
        print(f"epoch={epoch:02d}/{EPOCHS} frame_val_mse={fvm:.6f} temporal_val_mse={tvm:.6f}")
        if fvm<best_f: best_f=fvm; torch.save(fm.state_dict(),os.path.join(OUT,"best_mixed_frame_cnn.pt"))
        if tvm<best_t: best_t=tvm; torch.save(tm.state_dict(),os.path.join(OUT,"best_mixed_temporal_convgru.pt"))
    fm.load_state_dict(torch.load(os.path.join(OUT,"best_mixed_frame_cnn.pt"),map_location=device))
    tm.load_state_dict(torch.load(os.path.join(OUT,"best_mixed_temporal_convgru.pt"),map_location=device))
    detail=[]; summary=[]
    for m in RATES:
        for model,kind in ((fm,"FrameCNN"),(tm,"BiConvGRU")):
            rows=evaluate(model,kind,test,bank[m],device)
            for i,r in enumerate(rows): detail.append(dict(pattern_count=m,sampling_ratio=m/IMAGE_SIZE**2,method=kind,sequence_id=i,**r))
            pm,ps=stats(rows,"psnr_db"); sm,ss=stats(rows,"ssim"); em,es=stats(rows,"temporal_difference_error"); mm,ms=stats(rows,"mse")
            summary.append(dict(pattern_count=m,sampling_ratio=m/IMAGE_SIZE**2,method=kind,mse_mean=mm,mse_std=ms,
                psnr_mean_db=pm,psnr_std_db=ps,ssim_mean=sm,ssim_std=ss,
                temporal_difference_error_mean=em,temporal_difference_error_std=es))
            print(f"patterns={m:4d} {kind:10s} PSNR={pm:.2f}+/-{ps:.2f} SSIM={sm:.3f} temp_err={em:.4f}")
    def save_csv(name,rows):
        with open(os.path.join(OUT,name),"w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    save_csv("training_history.csv",history); save_csv("benchmark_detail.csv",detail); save_csv("benchmark_summary.csv",summary)
    elapsed=time.time()-start
    with open(os.path.join(OUT,"results.json"),"w",encoding="utf-8") as f:
        json.dump(dict(config=dict(seed=SEED,rates=list(RATES),train_sequences=TRAIN_N,val_sequences=VAL_N,
            test_sequences=TEST_N,epochs=EPOCHS,balanced_mixed_sampling=True),history=history,summary=summary,
            elapsed_seconds=elapsed),f,ensure_ascii=False,indent=2)
    x=np.array(RATES)/IMAGE_SIZE**2*100; fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for kind,mark in (("FrameCNN","s"),("BiConvGRU","^")):
        q=[r for r in summary if r["method"]==kind]
        for ax,key,err in zip(axes,("psnr_mean_db","ssim_mean","temporal_difference_error_mean"),
                             ("psnr_std_db","ssim_std","temporal_difference_error_std")):
            ax.errorbar(x,[r[key] for r in q],yerr=[r[err] for r in q],marker=mark,capsize=3,label=kind)
    for ax,title in zip(axes,("PSNR ↑","SSIM ↑","Temporal difference error ↓")):
        ax.set_xlabel("Sampling ratio (%)"); ax.set_title(title); ax.grid(alpha=.25); ax.legend()
    fig.suptitle("Balanced mixed-sampling training: 100 unseen sequences"); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"mixed_sampling_metrics.png"),dpi=180); plt.close(fig)
    print(f"elapsed_seconds={elapsed:.1f}"); print(f"saved={OUT}")

if __name__=="__main__": main()
