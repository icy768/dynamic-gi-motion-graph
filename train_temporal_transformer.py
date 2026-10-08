import csv, json, os, random, time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
from train_frame_cnn import FrameCNN, reconstruction_loss
from train_temporal_convgru import BidirectionalConvGRU, MovingMNISTSequences
from train_mixed_sampling import RATES, coarse_sequence, frame_forward, temporal_error

TRAIN_N, VAL_N, TEST_N = 800, 120, 100
BATCH, EPOCHS, LR = 8, 10, 5e-4
OUT = "temporal_transformer_outputs"
MIXED_DIR = "mixed_sampling_outputs"

class MiniTemporalTransformer(nn.Module):
    def __init__(self, dim=32, heads=4, layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16,dim,3,stride=2,padding=1), nn.ReLU(inplace=True),
        )
        block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=64, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(block, num_layers=layers)
        self.position = nn.Parameter(torch.zeros(1, 8, dim))
        nn.init.normal_(self.position, std=0.02)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim,16,4,stride=2,padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,1,4,stride=2,padding=1),
        )
    def forward(self, x):
        b,t,c,h,w=x.shape
        z=self.encoder(x.reshape(b*t,c,h,w)); _,d,hh,ww=z.shape
        z=z.reshape(b,t,d,hh,ww).permute(0,3,4,1,2).reshape(b*hh*ww,t,d)
        z=self.temporal(z+self.position[:,:t])
        z=z.reshape(b,hh,ww,t,d).permute(0,3,4,1,2).reshape(b*t,d,hh,ww)
        return torch.sigmoid(self.decoder(z)).reshape(b,t,1,h,w)

@torch.no_grad()
def evaluate(model, kind, loader, patterns, device):
    model.eval(); rows=[]
    for target,labels in loader:
        target=target.to(device); coarse=coarse_sequence(target,patterns)
        if kind=="FrameCNN": pred=frame_forward(model,coarse)
        else: pred=model(coarse)
        b,t,c,h,w=target.shape
        mse=(pred-target).square().mean((1,2,3,4)); psnr=10*torch.log10(1/mse.clamp_min(1e-12))
        sv=ssim(pred.reshape(b*t,c,h,w),target.reshape(b*t,c,h,w),data_range=1.0,
                size_average=False).reshape(b,t).mean(1)
        te=((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean((1,2,3,4))
        for i in range(b): rows.append(dict(label=int(labels[i]),mse=float(mse[i]),psnr_db=float(psnr[i]),
            ssim=float(sv[i]),temporal_difference_error=float(te[i])))
    return rows

def stats(rows,key):
    v=np.array([r[key] for r in rows]); return float(v.mean()),float(v.std(ddof=1))

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    os.makedirs(OUT,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    bank={m:make_patterns(m,IMAGE_SIZE,IMAGE_SIZE,seed=SEED).to(device) for m in RATES}
    train=DataLoader(MovingMNISTSequences("./data",True,TRAIN_N,SEED+200),batch_size=BATCH,shuffle=True,num_workers=0)
    val=DataLoader(MovingMNISTSequences("./data",False,VAL_N,SEED+201),batch_size=BATCH,shuffle=False,num_workers=0)
    test=DataLoader(MovingMNISTSequences("./data",False,TEST_N,SEED+300),batch_size=4,shuffle=False,num_workers=0)
    model=MiniTemporalTransformer().to(device); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    history=[]; best=float("inf"); start=time.time()
    for epoch in range(1,EPOCHS+1):
        model.train(); total=n=0
        for bi,(target,_) in enumerate(train):
            target=target.to(device); m=RATES[(bi+epoch-1)%len(RATES)]
            with torch.no_grad(): coarse=coarse_sequence(target,bank[m])
            pred=model(coarse); flat_t=target.reshape(-1,1,IMAGE_SIZE,IMAGE_SIZE)
            loss=reconstruction_loss(pred.reshape_as(flat_t),flat_t)+.15*temporal_error(pred,target)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            total+=loss.item()*target.shape[0]; n+=target.shape[0]
        vals=[]
        for m in RATES: vals += [r["mse"] for r in evaluate(model,"TemporalTransformer",val,bank[m],device)]
        vm=float(np.mean(vals)); history.append(dict(epoch=epoch,train_loss=total/n,val_mse=vm))
        print(f"epoch={epoch:02d}/{EPOCHS} train_loss={total/n:.6f} val_mse={vm:.6f}")
        if vm<best: best=vm; torch.save(model.state_dict(),os.path.join(OUT,"best_temporal_transformer.pt"))
    model.load_state_dict(torch.load(os.path.join(OUT,"best_temporal_transformer.pt"),map_location=device))
    frame=FrameCNN().to(device); frame.load_state_dict(torch.load(os.path.join(MIXED_DIR,"best_mixed_frame_cnn.pt"),map_location=device))
    gru=BidirectionalConvGRU().to(device); gru.load_state_dict(torch.load(os.path.join(MIXED_DIR,"best_mixed_temporal_convgru.pt"),map_location=device))
    detail=[]; summary=[]
    for m in RATES:
        for net,kind in ((frame,"FrameCNN"),(gru,"BiConvGRU"),(model,"TemporalTransformer")):
            rows=evaluate(net,kind,test,bank[m],device)
            for i,r in enumerate(rows): detail.append(dict(pattern_count=m,sampling_ratio=m/IMAGE_SIZE**2,method=kind,sequence_id=i,**r))
            pm,ps=stats(rows,"psnr_db"); sm,ss=stats(rows,"ssim"); em,es=stats(rows,"temporal_difference_error"); mm,ms=stats(rows,"mse")
            summary.append(dict(pattern_count=m,sampling_ratio=m/IMAGE_SIZE**2,method=kind,mse_mean=mm,mse_std=ms,
                psnr_mean_db=pm,psnr_std_db=ps,ssim_mean=sm,ssim_std=ss,
                temporal_difference_error_mean=em,temporal_difference_error_std=es))
            print(f"patterns={m:4d} {kind:19s} PSNR={pm:.2f}+/-{ps:.2f} SSIM={sm:.3f} temp_err={em:.4f}")
    def save_csv(name,rows):
        with open(os.path.join(OUT,name),"w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    save_csv("training_history.csv",history);save_csv("benchmark_detail.csv",detail);save_csv("benchmark_summary.csv",summary)
    elapsed=time.time()-start
    params=sum(p.numel() for p in model.parameters())
    with open(os.path.join(OUT,"results.json"),"w",encoding="utf-8") as f:
        json.dump(dict(config=dict(seed=SEED,rates=list(RATES),train_sequences=TRAIN_N,val_sequences=VAL_N,
            test_sequences=TEST_N,epochs=EPOCHS,parameters=params),history=history,summary=summary,
            elapsed_seconds=elapsed),f,ensure_ascii=False,indent=2)
    x=np.array(RATES)/IMAGE_SIZE**2*100;fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for kind,mark in (("FrameCNN","s"),("BiConvGRU","^"),("TemporalTransformer","o")):
        q=[r for r in summary if r["method"]==kind]
        for ax,key,err in zip(axes,("psnr_mean_db","ssim_mean","temporal_difference_error_mean"),
            ("psnr_std_db","ssim_std","temporal_difference_error_std")):
            ax.errorbar(x,[r[key] for r in q],yerr=[r[err] for r in q],marker=mark,capsize=3,label=kind)
    for ax,title in zip(axes,("PSNR ↑","SSIM ↑","Temporal difference error ↓")):
        ax.set_xlabel("Sampling ratio (%)");ax.set_title(title);ax.grid(alpha=.25);ax.legend()
    fig.suptitle("Mixed-sampling models: 100 unseen sequences");plt.tight_layout()
    plt.savefig(os.path.join(OUT,"transformer_comparison.png"),dpi=180);plt.close(fig)
    print(f"parameters={params}");print(f"elapsed_seconds={elapsed:.1f}");print(f"saved={OUT}")

if __name__=="__main__":main()
