"""加入显式像素坐标的 DGI 全局位移回归。"""
import argparse,csv,json,random,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent; PROJECT=Path(r"D:\download\DynGhost-mini-repro"); SEED=20260902; RATES=(128,256,512,1024)
class CoordinateMotionNet(nn.Module):
 def __init__(self):
  super().__init__();grid=torch.linspace(-1,1,64);yy,xx=torch.meshgrid(grid,grid,indexing='ij');self.register_buffer('coords',torch.stack((xx,yy))[None]);self.net=nn.Sequential(nn.Conv2d(4,20,5,stride=2,padding=2),nn.GELU(),nn.Conv2d(20,36,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(36,48,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1));self.head=nn.Sequential(nn.Flatten(),nn.Linear(48,32),nn.GELU(),nn.Linear(32,2))
 def forward(self,pairs):return self.head(self.net(torch.cat((pairs,self.coords.expand(pairs.shape[0],-1,-1,-1)),1)))
def seed():random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def evaluate(model,loader,bank,dev):
 from train_mixed_sampling import coarse_sequence
 model.eval();out=[];sid=0
 with torch.no_grad():
  for tar,_,motion in loader:
   tar=tar.to(dev);b,t=tar.shape[:2]
   for n,p in bank.items():
    d=coarse_sequence(tar,p);e=model(torch.cat((d[:,:-1],d[:,1:]),2).reshape(b*(t-1),2,64,64)).reshape(b,t-1,2).cpu()
    for i in range(b):
     for f in range(t-1):out.append(dict(pattern_count=n,sequence_id=sid+i,frame_pair=f,true_dx=float(motion[i,0]),true_dy=float(motion[i,1]),estimated_dx=float(e[i,f,0]),estimated_dy=float(e[i,f,1]),vector_error_pixels=float(torch.linalg.vector_norm(e[i,f]-motion[i]))))
   sid+=b
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT);ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--output-dir',type=Path,default=HERE/'coordinate_motion_outputs');args=ap.parse_args();project=args.project_root.resolve();sys.path.insert(0,str(project));from mini_repro_dynghost import IMAGE_SIZE,make_patterns;from train_mixed_sampling import coarse_sequence;from train_oracle_motion_tgnn import MovingMNISTWithMotion
 seed();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);bank={n:make_patterns(n,IMAGE_SIZE,IMAGE_SIZE,SEED).to(dev) for n in RATES};train=DataLoader(MovingMNISTWithMotion(project/'data',True,800,SEED+200),batch_size=8,shuffle=True);val=DataLoader(MovingMNISTWithMotion(project/'data',False,120,SEED+201),batch_size=8);test=DataLoader(MovingMNISTWithMotion(project/'data',False,100,SEED+300),batch_size=8)
 m=CoordinateMotionNet().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);best=1e9;hist=[];ckpt=out/'best_coordinate_motion.pt'
 for ep in range(1,args.epochs+1):
  m.train();s=c=0
  for bi,(tar,_,motion) in enumerate(train):
   tar,motion=tar.to(dev),motion.to(dev);d=coarse_sequence(tar,bank[RATES[(bi+ep-1)%4]]);b,t=d.shape[:2];pred=m(torch.cat((d[:,:-1],d[:,1:]),2).reshape(b*(t-1),2,64,64)).reshape(b,t-1,2);loss=F.smooth_l1_loss(pred,motion[:,None].expand_as(pred));opt.zero_grad(set_to_none=True);loss.backward();opt.step();s+=loss.item()*b;c+=b
  v=evaluate(m,val,bank,dev);mae=float(np.mean([x['vector_error_pixels'] for x in v]));hist.append(dict(epoch=ep,train_loss=s/c,val_vector_mae_pixels=mae));print(f'epoch={ep:02d}/{args.epochs} train_loss={s/c:.6f} val_motion_mae={mae:.3f}')
  if mae<best:best=mae;torch.save(m.state_dict(),ckpt)
 m.load_state_dict(torch.load(ckpt,map_location=dev));detail=evaluate(m,test,bank,dev);summary=[]
 for n in RATES:
  x=np.array([r['vector_error_pixels'] for r in detail if r['pattern_count']==n]);summary.append(dict(pattern_count=n,sampling_ratio=n/4096,pairs=len(x),coordinate_motion_vector_mae_pixels=float(x.mean()),coordinate_motion_vector_rmse_pixels=float(np.sqrt(np.mean(x*x))),within_1_pixel_ratio=float(np.mean(x<=1)),simple_cnn_vector_mae_pixels={128:1.5716,256:1.5334,512:1.5785,1024:1.566}[n]))
 for r in summary:print(r)
 with (out/'detail.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=detail[0].keys());w.writeheader();w.writerows(detail)
 (out/'results.json').write_text(json.dumps(dict(config=dict(seed=SEED,estimator='direct DGI motion regressor with x/y coordinate channels',uses_ground_truth_motion_at_inference=False),history=hist,summary=summary,parameters=sum(p.numel() for p in m.parameters())),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
