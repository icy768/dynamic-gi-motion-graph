"""冻结 DGI 位移估计器后训练连续运动 TGNN。

估计器仅从 DGI 序列推理全局位移；重建模型和 Oracle / Zero-motion 消融保持同一结构。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path(r"D:\download\DynGhost-mini-repro")

def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,default=DEFAULT_PROJECT)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--output-dir",type=Path,default=HERE / "estimated_motion_tgnn_outputs")
    parser.add_argument("--motion-checkpoint",type=Path,default=HERE / "estimated_motion_outputs" / "best_dgi_motion_net.pt")
    parser.add_argument("--fusion",choices=("mean","median"),default="mean",help="融合七组相邻帧位移的方式")
    args=parser.parse_args(); project=args.project_root.resolve(); sys.path[:0]=[str(project),str(HERE)]
    from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
    from train_frame_cnn import reconstruction_loss
    from train_mixed_sampling import RATES, coarse_sequence, stats, temporal_error
    from train_oracle_motion_tgnn import BATCH, LR, TEST_N, TRAIN_N, VAL_N, MovingMNISTWithMotion, OracleMotionTGNN, set_seed
    from train_dgi_motion_estimator import DgiMotionNet

    class EstimatedMotionTGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.motion_net=DgiMotionNet()
            self.motion_net.load_state_dict(torch.load(args.motion_checkpoint, map_location="cpu"))
            for parameter in self.motion_net.parameters(): parameter.requires_grad_(False)
            self.reconstructor=OracleMotionTGNN()
        def estimate_motion(self, x):
            batch, frames, _, height, width=x.shape
            pairs=torch.cat((x[:,:-1],x[:,1:]),dim=2).reshape(batch*(frames-1),2,height,width)
            self.motion_net.eval()
            with torch.no_grad():
                estimates=self.motion_net(pairs).reshape(batch,frames-1,2)
                # 当前数据的平移速度全局恒定；中位数可抑制少数噪声帧的异常估计。
                if args.fusion == "median":
                    return estimates.median(dim=1).values.clamp(-5.0,5.0)
                return estimates.mean(dim=1).clamp(-5.0,5.0)
        def forward(self, x):
            return self.reconstructor(x,self.estimate_motion(x))

    def evaluate(model,loader,patterns,device):
        model.eval(); rows=[]
        with torch.no_grad():
            for target,labels,_ in loader:
                target=target.to(device); pred=model(coarse_sequence(target,patterns)); b,t,c,h,w=target.shape
                mse=(pred-target).square().mean((1,2,3,4)); psnr=10*torch.log10(1/mse.clamp_min(1e-12))
                score=ssim(pred.reshape(b*t,c,h,w),target.reshape(b*t,c,h,w),data_range=1.,size_average=False).reshape(b,t).mean(1)
                terr=((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean((1,2,3,4))
                motion=model.estimate_motion(coarse_sequence(target,patterns)).cpu()
                for i in range(b): rows.append(dict(label=int(labels[i]),mse=float(mse[i]),psnr_db=float(psnr[i]),ssim=float(score[i]),temporal_difference_error=float(terr[i]),estimated_dx=float(motion[i,0]),estimated_dy=float(motion[i,1])))
        return rows

    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True); set_seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"device={device}")
    bank={n:make_patterns(n,IMAGE_SIZE,IMAGE_SIZE,SEED).to(device) for n in RATES}
    train=DataLoader(MovingMNISTWithMotion(project / "data",True,TRAIN_N,SEED+200),batch_size=BATCH,shuffle=True)
    val=DataLoader(MovingMNISTWithMotion(project / "data",False,VAL_N,SEED+201),batch_size=BATCH)
    test=DataLoader(MovingMNISTWithMotion(project / "data",False,TEST_N,SEED+300),batch_size=4)
    model=EstimatedMotionTGNN().to(device); opt=torch.optim.AdamW(model.reconstructor.parameters(),lr=LR,weight_decay=1e-4)
    best=float("inf"); history=[]; checkpoint=output / "best_estimated_motion_tgnn.pt"; start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); total=count=0
        for bi,(target,_,_) in enumerate(train):
            target=target.to(device); coarse=coarse_sequence(target,bank[RATES[(bi+epoch-1)%len(RATES)]])
            pred=model(coarse); flat=target.reshape(-1,1,IMAGE_SIZE,IMAGE_SIZE)
            loss=reconstruction_loss(pred.reshape_as(flat),flat)+.15*temporal_error(pred,target)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.reconstructor.parameters(),1); opt.step()
            total+=loss.item()*target.shape[0]; count+=target.shape[0]
        values=[]
        for n in RATES: values.extend(row['mse'] for row in evaluate(model,val,bank[n],device))
        vm=float(np.mean(values)); history.append(dict(epoch=epoch,train_loss=total/count,val_mse=vm)); print(f"epoch={epoch:02d}/{args.epochs} train_loss={total/count:.6f} val_mse={vm:.6f}")
        if vm<best: best=vm; torch.save(model.state_dict(),checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location=device)); detail=[]; summary=[]
    for n in RATES:
        rows=evaluate(model,test,bank[n],device)
        for sequence_id,row in enumerate(rows): detail.append(dict(pattern_count=n,sampling_ratio=n/IMAGE_SIZE**2,method=f"EstimatedMotionTGNN-{args.fusion}",sequence_id=sequence_id,**row))
        mm,ms=stats(rows,'mse'); pm,ps=stats(rows,'psnr_db'); sm,ss=stats(rows,'ssim'); tm,ts=stats(rows,'temporal_difference_error')
        summary.append(dict(pattern_count=n,sampling_ratio=n/IMAGE_SIZE**2,method=f"EstimatedMotionTGNN-{args.fusion}",mse_mean=mm,mse_std=ms,psnr_mean_db=pm,psnr_std_db=ps,ssim_mean=sm,ssim_std=ss,temporal_difference_error_mean=tm,temporal_difference_error_std=ts))
        print(f"patterns={n:4d} EstimatedMotionTGNN PSNR={pm:.2f}+/-{ps:.2f} SSIM={sm:.3f} temp_err={tm:.4f}")
    save_csv(output/'training_history.csv',history); save_csv(output/'benchmark_detail.csv',detail); save_csv(output/'benchmark_summary.csv',summary)
    result={"config":{"seed":SEED,"rates":list(RATES),"train_sequences":TRAIN_N,"val_sequences":VAL_N,"test_sequences":TEST_N,"epochs":args.epochs,"graph":"four-neighbor spatial edges plus DGI-estimated global bilinear motion edges","motion_estimator":f"frozen DgiMotionNet; seven pairwise estimates fused by {args.fusion}","uses_ground_truth_motion_at_inference":False},"history":history,"summary":summary,"parameters":{"motion_estimator":sum(p.numel() for p in model.motion_net.parameters()),"reconstructor":sum(p.numel() for p in model.reconstructor.parameters())},"elapsed_seconds":time.time()-start}
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(f"saved={output}")
if __name__=='__main__': main()
