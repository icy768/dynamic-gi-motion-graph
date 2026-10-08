import csv, json, os, random, time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
from train_frame_cnn import FrameCNN, reconstruction_loss
from train_temporal_convgru import BidirectionalConvGRU, MovingMNISTSequences
from train_mixed_sampling import RATES, coarse_sequence, frame_forward, temporal_error
from train_temporal_transformer import MiniTemporalTransformer, evaluate, stats

TRAIN_N, VAL_N, TEST_N = 800, 120, 100
BATCH, EPOCHS, LR = 8, 10, 5e-4
OUT = "spatiotemporal_transformer_outputs"
MIXED_DIR = "mixed_sampling_outputs"
TEMP_DIR = "temporal_transformer_outputs"

class AttentionUnit(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.spatial = nn.TransformerEncoderLayer(
            dim, heads, 64, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoderLayer(
            dim, heads, 64, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    def forward(self, z):
        # z [B,T,C,H,W]: first attend within each frame, then along time per location.
        b,t,c,h,w=z.shape
        s=z.permute(0,1,3,4,2).reshape(b*t,h*w,c)
        s=self.spatial(s).reshape(b,t,h,w,c)
        q=s.permute(0,2,3,1,4).reshape(b*h*w,t,c)
        q=self.temporal(q).reshape(b,h,w,t,c)
        return q.permute(0,3,4,1,2)

class MiniSpatioTemporalTransformer(nn.Module):
    def __init__(self, dim=32, heads=4, blocks=2):
        super().__init__()
        self.enc1=nn.Sequential(nn.Conv2d(1,16,3,stride=2,padding=1),nn.ReLU(inplace=True))
        self.enc2=nn.Sequential(nn.Conv2d(16,dim,3,stride=2,padding=1),nn.ReLU(inplace=True))
        self.time_pos=nn.Parameter(torch.zeros(1,8,dim,1,1))
        self.units=nn.ModuleList([AttentionUnit(dim,heads) for _ in range(blocks)])
        self.up1=nn.ConvTranspose2d(dim,16,4,stride=2,padding=1)
        self.fuse1=nn.Sequential(nn.Conv2d(32,24,3,padding=1),nn.ReLU(inplace=True),
                                 nn.Conv2d(24,16,3,padding=1),nn.ReLU(inplace=True))
        self.up2=nn.ConvTranspose2d(16,8,4,stride=2,padding=1)
        self.out=nn.Sequential(nn.Conv2d(9,16,3,padding=1),nn.ReLU(inplace=True),
                               nn.Conv2d(16,1,3,padding=1))
        nn.init.normal_(self.time_pos,std=.02)
    def forward(self,x):
        b,t,c,h,w=x.shape; flat=x.reshape(b*t,c,h,w)
        e1=self.enc1(flat); e2=self.enc2(e1); _,d,hh,ww=e2.shape
        z=e2.reshape(b,t,d,hh,ww)+self.time_pos[:,:t]
        for unit in self.units: z=unit(z)
        z=z.reshape(b*t,d,hh,ww)
        z=F.relu(self.up1(z)); z=self.fuse1(torch.cat([z,e1],dim=1))
        z=F.relu(self.up2(z)); logits=self.out(torch.cat([z,flat],dim=1))
        return torch.sigmoid(logits).reshape(b,t,1,h,w)

def main():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    os.makedirs(OUT,exist_ok=True);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    bank={m:make_patterns(m,IMAGE_SIZE,IMAGE_SIZE,seed=SEED).to(device) for m in RATES}
    train=DataLoader(MovingMNISTSequences("./data",True,TRAIN_N,SEED+200),batch_size=BATCH,shuffle=True,num_workers=0)
    val=DataLoader(MovingMNISTSequences("./data",False,VAL_N,SEED+201),batch_size=BATCH,shuffle=False,num_workers=0)
    test=DataLoader(MovingMNISTSequences("./data",False,TEST_N,SEED+300),batch_size=4,shuffle=False,num_workers=0)
    model=MiniSpatioTemporalTransformer().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    history=[];best=float("inf");start=time.time()
    for epoch in range(1,EPOCHS+1):
        model.train();total=n=0
        for bi,(target,_) in enumerate(train):
            target=target.to(device);m=RATES[(bi+epoch-1)%len(RATES)]
            with torch.no_grad():coarse=coarse_sequence(target,bank[m])
            pred=model(coarse);flat=target.reshape(-1,1,IMAGE_SIZE,IMAGE_SIZE)
            loss=reconstruction_loss(pred.reshape_as(flat),flat)+.15*temporal_error(pred,target)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            total+=loss.item()*target.shape[0];n+=target.shape[0]
        vals=[]
        for m in RATES:vals += [r["mse"] for r in evaluate(model,"SpatioTemporalTransformer",val,bank[m],device)]
        vm=float(np.mean(vals));history.append(dict(epoch=epoch,train_loss=total/n,val_mse=vm))
        print(f"epoch={epoch:02d}/{EPOCHS} train_loss={total/n:.6f} val_mse={vm:.6f}")
        if vm<best:best=vm;torch.save(model.state_dict(),os.path.join(OUT,"best_spatiotemporal_transformer.pt"))
    model.load_state_dict(torch.load(os.path.join(OUT,"best_spatiotemporal_transformer.pt"),map_location=device))
    frame=FrameCNN().to(device);frame.load_state_dict(torch.load(os.path.join(MIXED_DIR,"best_mixed_frame_cnn.pt"),map_location=device))
    gru=BidirectionalConvGRU().to(device);gru.load_state_dict(torch.load(os.path.join(MIXED_DIR,"best_mixed_temporal_convgru.pt"),map_location=device))
    temp=MiniTemporalTransformer().to(device);temp.load_state_dict(torch.load(os.path.join(TEMP_DIR,"best_temporal_transformer.pt"),map_location=device))
    detail=[];summary=[]
    methods=((frame,"FrameCNN"),(gru,"BiConvGRU"),(temp,"TemporalTransformer"),(model,"SpatioTemporalTransformer"))
    for m in RATES:
        for net,kind in methods:
            rows=evaluate(net,kind,test,bank[m],device)
            for i,r in enumerate(rows):detail.append(dict(pattern_count=m,sampling_ratio=m/IMAGE_SIZE**2,method=kind,sequence_id=i,**r))
            pm,ps=stats(rows,"psnr_db");sm,ss=stats(rows,"ssim");em,es=stats(rows,"temporal_difference_error");mm,ms=stats(rows,"mse")
            summary.append(dict(pattern_count=m,sampling_ratio=m/IMAGE_SIZE**2,method=kind,mse_mean=mm,mse_std=ms,
                psnr_mean_db=pm,psnr_std_db=ps,ssim_mean=sm,ssim_std=ss,
                temporal_difference_error_mean=em,temporal_difference_error_std=es))
            print(f"patterns={m:4d} {kind:26s} PSNR={pm:.2f} SSIM={sm:.3f} temp_err={em:.4f}")
    def save_csv(name,rows):
        with open(os.path.join(OUT,name),"w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    save_csv("training_history.csv",history);save_csv("benchmark_detail.csv",detail);save_csv("benchmark_summary.csv",summary)
    elapsed=time.time()-start;params=sum(p.numel() for p in model.parameters())
    with open(os.path.join(OUT,"results.json"),"w",encoding="utf-8") as f:
        json.dump(dict(config=dict(seed=SEED,rates=list(RATES),train_sequences=TRAIN_N,val_sequences=VAL_N,
            test_sequences=TEST_N,epochs=EPOCHS,parameters=params,alternating_spatial_temporal_attention=True,
            skip_connections=True),history=history,summary=summary,elapsed_seconds=elapsed),f,ensure_ascii=False,indent=2)
    x=np.array(RATES)/IMAGE_SIZE**2*100;fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    marks=("s","^","o","D")
    for (_,kind),mark in zip(methods,marks):
        q=[r for r in summary if r["method"]==kind]
        for ax,key,err in zip(axes,("psnr_mean_db","ssim_mean","temporal_difference_error_mean"),
            ("psnr_std_db","ssim_std","temporal_difference_error_std")):
            ax.errorbar(x,[r[key] for r in q],yerr=[r[err] for r in q],marker=mark,capsize=2,label=kind)
    for ax,title in zip(axes,("PSNR ↑","SSIM ↑","Temporal difference error ↓")):
        ax.set_xlabel("Sampling ratio (%)");ax.set_title(title);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle("Alternating spatial-temporal attention: 100 unseen sequences");plt.tight_layout()
    plt.savefig(os.path.join(OUT,"spatiotemporal_comparison.png"),dpi=180);plt.close(fig)
    print(f"parameters={params}");print(f"elapsed_seconds={elapsed:.1f}");print(f"saved={OUT}")

if __name__=="__main__":main()
