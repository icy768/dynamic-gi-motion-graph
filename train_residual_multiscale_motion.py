"""残差多尺度 CNN：从相邻 DGI 帧直接回归全局平移。"""
import argparse,csv,json,random,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent; PROJECT=Path(r"D:\download\DynGhost-mini-repro");SEED=20260902;RATES=(128,256,512,1024)
class ResBlock(nn.Module):
 def __init__(self,c): super().__init__();self.a=nn.Conv2d(c,c,3,padding=1);self.b=nn.Conv2d(c,c,3,padding=1)
 def forward(self,x):return F.gelu(x+self.b(F.gelu(self.a(x))))
class ResidualMultiScaleMotionNet(nn.Module):
 def __init__(self):
  super().__init__();self.s32=nn.Sequential(nn.Conv2d(2,24,5,stride=2,padding=2),nn.GELU(),ResBlock(24));self.s16=nn.Sequential(nn.Conv2d(24,48,3,stride=2,padding=1),nn.GELU(),ResBlock(48));self.s8=nn.Sequential(nn.Conv2d(48,72,3,stride=2,padding=1),nn.GELU(),ResBlock(72));self.head=nn.Sequential(nn.Linear(24+48+72,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,32),nn.GELU(),nn.Linear(32,2))
 def forward(self,x):
  a=self.s32(x);b=self.s16(a);c=self.s8(b);return self.head(torch.cat([a.mean((-2,-1)),b.mean((-2,-1)),c.mean((-2,-1))],1))
def seed():random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def rows(model,loader,bank,device):
 from train_mixed_sampling import coarse_sequence
 model.eval();out=[];sid=0
 with torch.no_grad():
  for target,_,motion in loader:
   target=target.to(device);b,t=target.shape[:2]
   for n,p in bank.items():
    d=coarse_sequence(target,p);pair=torch.cat((d[:,:-1],d[:,1:]),2).reshape(b*(t-1),2,64,64);e=model(pair).reshape(b,t-1,2).cpu()
    for i in range(b):
     for k in range(t-1):out.append(dict(pattern_count=n,sequence_id=sid+i,frame_pair=k,true_dx=float(motion[i,0]),true_dy=float(motion[i,1]),estimated_dx=float(e[i,k,0]),estimated_dy=float(e[i,k,1]),vector_error_pixels=float(torch.linalg.vector_norm(e[i,k]-motion[i]))))
   sid+=b
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT);ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--output-dir',type=Path,default=HERE/'residual_multiscale_motion_outputs');args=ap.parse_args();project=args.project_root.resolve();sys.path.insert(0,str(project));from mini_repro_dynghost import IMAGE_SIZE,make_patterns;from train_mixed_sampling import coarse_sequence;from train_oracle_motion_tgnn import MovingMNISTWithMotion
 seed();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);bank={n:make_patterns(n,IMAGE_SIZE,IMAGE_SIZE,SEED).to(dev) for n in RATES};train=DataLoader(MovingMNISTWithMotion(project/'data',True,800,SEED+200),batch_size=8,shuffle=True);val=DataLoader(MovingMNISTWithMotion(project/'data',False,120,SEED+201),batch_size=8);test=DataLoader(MovingMNISTWithMotion(project/'data',False,100,SEED+300),batch_size=8)
 model=ResidualMultiScaleMotionNet().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4);best=1e9;hist=[];ckpt=out/'best_residual_multiscale_motion.pt'
 for ep in range(1,args.epochs+1):
  model.train();loss_sum=num=0
  for bi,(target,_,motion) in enumerate(train):
   target,motion=target.to(dev),motion.to(dev);d=coarse_sequence(target,bank[RATES[(bi+ep-1)%4]]);b,t=d.shape[:2];pair=torch.cat((d[:,:-1],d[:,1:]),2).reshape(b*(t-1),2,64,64);pred=model(pair).reshape(b,t-1,2);loss=F.smooth_l1_loss(pred,motion[:,None].expand_as(pred));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=loss.item()*b;num+=b
  v=rows(model,val,bank,dev);mae=float(np.mean([r['vector_error_pixels'] for r in v]));hist.append(dict(epoch=ep,train_loss=loss_sum/num,val_vector_mae_pixels=mae));print(f'epoch={ep:02d}/{args.epochs} train_loss={loss_sum/num:.6f} val_motion_mae={mae:.3f}')
  if mae<best:best=mae;torch.save(model.state_dict(),ckpt)
 model.load_state_dict(torch.load(ckpt,map_location=dev));detail=rows(model,test,bank,dev);summary=[]
 for n in RATES:
  x=np.array([r['vector_error_pixels'] for r in detail if r['pattern_count']==n]);summary.append(dict(pattern_count=n,sampling_ratio=n/4096,pairs=len(x),residual_multiscale_vector_mae_pixels=float(x.mean()),residual_multiscale_vector_rmse_pixels=float(np.sqrt(np.mean(x*x))),within_1_pixel_ratio=float(np.mean(x<=1.)),simple_cnn_vector_mae_pixels={128:1.5716,256:1.5334,512:1.5785,1024:1.566}[n]))
 for x in summary:print(x)
 with (out/'detail.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=detail[0].keys());w.writeheader();w.writerows(detail)
 (out/'results.json').write_text(json.dumps(dict(config=dict(seed=SEED,input='two adjacent DGI frames',estimator='direct residual multi-scale global-motion regressor',uses_ground_truth_motion_at_inference=False),history=hist,summary=summary,parameters=sum(p.numel() for p in model.parameters())),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
