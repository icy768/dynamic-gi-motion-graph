"""多尺度特征相关性 DGI 全局位移估计实验。"""
import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; PROJECT=Path(r"D:\download\DynGhost-mini-repro"); SEED=20260902; RATES=(128,256,512,1024)

class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage32=nn.Sequential(nn.Conv2d(1,24,5,stride=2,padding=2),nn.GELU(),nn.Conv2d(24,24,3,padding=1),nn.GELU())
        self.stage16=nn.Sequential(nn.Conv2d(24,40,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(40,40,3,padding=1),nn.GELU())
    def forward(self,x):
        f32=self.stage32(x); return f32,self.stage16(f32)

def correlation_volume(first, second, radius):
    """每个通道是一个候选位移的归一化局部特征相似度。"""
    first=F.normalize(first,dim=1); second=F.normalize(second,dim=1); h,w=first.shape[-2:]; padded=F.pad(second,(radius,radius,radius,radius))
    values=[]
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            candidate=padded[:,:,radius+dy:radius+dy+h,radius+dx:radius+dx+w]
            values.append((first*candidate).sum(dim=1,keepdim=True))
    return torch.cat(values,dim=1)

class MultiScaleCorrelationMotionNet(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=SharedEncoder()
        # 32x32 半径3（覆盖约6像素）+ 16x16 半径2（覆盖约8像素）。
        self.head=nn.Sequential(nn.Linear(49+25,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,pairs):
        first32,first16=self.encoder(pairs[:,:1]); second32,second16=self.encoder(pairs[:,1:])
        cost32=correlation_volume(first32,second32,3).mean(dim=(-2,-1))
        cost16=correlation_volume(first16,second16,2).mean(dim=(-2,-1))
        return self.head(torch.cat((cost32,cost16),dim=1))

def set_seed(): random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def predict_rows(model,loader,bank,device):
    from train_mixed_sampling import coarse_sequence
    model.eval(); rows=[]; seq=0
    with torch.no_grad():
        for target,_,motion in loader:
            target=target.to(device); b,t=target.shape[:2]
            for count,patterns in bank.items():
                dgi=coarse_sequence(target,patterns); pairs=torch.cat((dgi[:,:-1],dgi[:,1:]),2).reshape(b*(t-1),2,64,64)
                est=model(pairs).reshape(b,t-1,2).cpu()
                for i in range(b):
                    for frame in range(t-1): rows.append(dict(pattern_count=count,sequence_id=seq+i,frame_pair=frame,true_dx=float(motion[i,0]),true_dy=float(motion[i,1]),estimated_dx=float(est[i,frame,0]),estimated_dy=float(est[i,frame,1]),vector_error_pixels=float(torch.linalg.vector_norm(est[i,frame]-motion[i]))))
            seq+=b
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT);ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--output-dir',type=Path,default=HERE/'multiscale_motion_outputs');args=ap.parse_args()
    project=args.project_root.resolve();sys.path.insert(0,str(project));from mini_repro_dynghost import IMAGE_SIZE,make_patterns;from train_mixed_sampling import coarse_sequence;from train_oracle_motion_tgnn import MovingMNISTWithMotion
    set_seed();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    bank={n:make_patterns(n,IMAGE_SIZE,IMAGE_SIZE,SEED).to(device) for n in RATES};train=DataLoader(MovingMNISTWithMotion(project/'data',True,800,SEED+200),batch_size=8,shuffle=True);val=DataLoader(MovingMNISTWithMotion(project/'data',False,120,SEED+201),batch_size=8);test=DataLoader(MovingMNISTWithMotion(project/'data',False,100,SEED+300),batch_size=8)
    model=MultiScaleCorrelationMotionNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);best=float('inf');history=[];ckpt=out/'best_multiscale_correlation_motion.pt'
    for epoch in range(1,args.epochs+1):
        model.train();total=num=0
        for bi,(target,_,motion) in enumerate(train):
            target,motion=target.to(device),motion.to(device);dgi=coarse_sequence(target,bank[RATES[(bi+epoch-1)%4]]);b,t=dgi.shape[:2];pairs=torch.cat((dgi[:,:-1],dgi[:,1:]),2).reshape(b*(t-1),2,64,64);pred=model(pairs).reshape(b,t-1,2);loss=F.smooth_l1_loss(pred,motion[:,None].expand_as(pred));opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=loss.item()*b;num+=b
        rows=predict_rows(model,val,bank,device);mae=float(np.mean([r['vector_error_pixels'] for r in rows]));history.append(dict(epoch=epoch,train_loss=total/num,val_vector_mae_pixels=mae));print(f'epoch={epoch:02d}/{args.epochs} train_loss={total/num:.6f} val_motion_mae={mae:.3f}')
        if mae<best:best=mae;torch.save(model.state_dict(),ckpt)
    model.load_state_dict(torch.load(ckpt,map_location=device));detail=predict_rows(model,test,bank,device);summary=[]
    for n in RATES:
        values=np.array([r['vector_error_pixels'] for r in detail if r['pattern_count']==n]);summary.append(dict(pattern_count=n,sampling_ratio=n/4096,pairs=len(values),multiscale_correlation_vector_mae_pixels=float(values.mean()),multiscale_correlation_vector_rmse_pixels=float(np.sqrt(np.mean(values**2))),within_1_pixel_ratio=float(np.mean(values<=1.)),previous_simple_motion_mae_pixels=({128:1.5716,256:1.5334,512:1.5785,1024:1.5660}[n])))
    for r in summary:print(r)
    with (out/'multiscale_correlation_detail.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=detail[0].keys());w.writeheader();w.writerows(detail)
    (out/'results.json').write_text(json.dumps(dict(config=dict(seed=SEED,input='two adjacent DGI frames',estimator='shared two-scale encoder plus local correlation volumes',uses_ground_truth_motion_at_inference=False),history=history,summary=summary,parameters=sum(p.numel() for p in model.parameters())),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
