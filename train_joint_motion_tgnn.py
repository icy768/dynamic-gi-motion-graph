"""端到端联合微调：MotionNet 与连续运动 TGNN 共同优化。"""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;PROJECT=Path(r"D:\download\DynGhost-mini-repro")
def csv_save(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT);ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--motion-weight',type=float,default=.05);ap.add_argument('--output-dir',type=Path,default=HERE/'joint_motion_tgnn_outputs');ap.add_argument('--initial-checkpoint',type=Path,default=HERE/'estimated_motion_tgnn_outputs'/'best_estimated_motion_tgnn.pt');ap.add_argument('--run-seed',type=int,default=None,help='仅改变训练随机性；数据、散斑与测试集保持固定');args=ap.parse_args();project=args.project_root.resolve();sys.path[:0]=[str(project),str(HERE)]
 from mini_repro_dynghost import IMAGE_SIZE,SEED,make_patterns
 from train_frame_cnn import reconstruction_loss
 from train_mixed_sampling import RATES,coarse_sequence,stats,temporal_error
 from train_oracle_motion_tgnn import BATCH,LR,TEST_N,TRAIN_N,VAL_N,MovingMNISTWithMotion,OracleMotionTGNN,set_seed
 from train_dgi_motion_estimator import DgiMotionNet
 class JointModel(nn.Module):
  def __init__(self):super().__init__();self.motion_net=DgiMotionNet();self.reconstructor=OracleMotionTGNN()
  def pair_motion(self,x):
   b,t,_,h,w=x.shape;pair=torch.cat((x[:,:-1],x[:,1:]),2).reshape(b*(t-1),2,h,w);return self.motion_net(pair).reshape(b,t-1,2)
  def forward(self,x):return self.reconstructor(x,self.pair_motion(x).mean(1).clamp(-5,5))
 def evaluate(m,loader,pats,dev):
  m.eval();rows=[]
  with torch.no_grad():
   for target,labels,motion in loader:
    target=target.to(dev);coarse=coarse_sequence(target,pats);pred=m(coarse);b,t,c,h,w=target.shape;mse=(pred-target).square().mean((1,2,3,4));psnr=10*torch.log10(1/mse.clamp_min(1e-12));score=ssim(pred.reshape(b*t,c,h,w),target.reshape(b*t,c,h,w),data_range=1.,size_average=False).reshape(b,t).mean(1);terr=((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean((1,2,3,4));est=m.pair_motion(coarse).mean(1).cpu()
    for i in range(b):rows.append(dict(label=int(labels[i]),mse=float(mse[i]),psnr_db=float(psnr[i]),ssim=float(score[i]),temporal_difference_error=float(terr[i]),motion_vector_error_pixels=float(torch.linalg.vector_norm(est[i]-motion[i]))) )
  return rows
 out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);run_seed=SEED if args.run_seed is None else args.run_seed;set_seed(run_seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');bank={n:make_patterns(n,IMAGE_SIZE,IMAGE_SIZE,SEED).to(dev) for n in RATES};train=DataLoader(MovingMNISTWithMotion(project/'data',True,TRAIN_N,SEED+200),batch_size=BATCH,shuffle=True);val=DataLoader(MovingMNISTWithMotion(project/'data',False,VAL_N,SEED+201),batch_size=BATCH);test=DataLoader(MovingMNISTWithMotion(project/'data',False,TEST_N,SEED+300),batch_size=4)
 model=JointModel().to(dev);model.load_state_dict(torch.load(args.initial_checkpoint,map_location=dev));opt=torch.optim.AdamW(model.parameters(),lr=LR*.5,weight_decay=1e-4);best=1e9;hist=[];ckpt=out/'best_joint_motion_tgnn.pt';start=time.time()
 for ep in range(1,args.epochs+1):
  model.train();rec_sum,motion_sum,num=0.,0.,0
  for bi,(target,_,truth) in enumerate(train):
   target,truth=target.to(dev),truth.to(dev);coarse=coarse_sequence(target,bank[RATES[(bi+ep-1)%len(RATES)]]);pred=model(coarse);flat=target.reshape(-1,1,IMAGE_SIZE,IMAGE_SIZE);rec=reconstruction_loss(pred.reshape_as(flat),flat)+.15*temporal_error(pred,target);pair=model.pair_motion(coarse);mloss=F.smooth_l1_loss(pair,truth[:,None].expand_as(pair));loss=rec+args.motion_weight*mloss;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();rec_sum+=rec.item()*target.shape[0];motion_sum+=mloss.item()*target.shape[0];num+=target.shape[0]
  vals=[]
  for n in RATES:vals.extend(r['mse'] for r in evaluate(model,val,bank[n],dev))
  vm=float(np.mean(vals));hist.append(dict(epoch=ep,reconstruction_loss=rec_sum/num,motion_loss=motion_sum/num,val_mse=vm));print(f'epoch={ep:02d}/{args.epochs} rec_loss={rec_sum/num:.6f} motion_loss={motion_sum/num:.6f} val_mse={vm:.6f}')
  if vm<best:best=vm;torch.save(model.state_dict(),ckpt)
 model.load_state_dict(torch.load(ckpt,map_location=dev));detail=[];summary=[]
 for n in RATES:
  rows=evaluate(model,test,bank[n],dev)
  for i,r in enumerate(rows):detail.append(dict(pattern_count=n,sampling_ratio=n/IMAGE_SIZE**2,method='JointMotionTGNN',sequence_id=i,**r))
  mm,ms=stats(rows,'mse');pm,ps=stats(rows,'psnr_db');sm,ss=stats(rows,'ssim');tm,ts=stats(rows,'temporal_difference_error');me,mestd=stats(rows,'motion_vector_error_pixels');summary.append(dict(pattern_count=n,sampling_ratio=n/IMAGE_SIZE**2,method='JointMotionTGNN',mse_mean=mm,mse_std=ms,psnr_mean_db=pm,psnr_std_db=ps,ssim_mean=sm,ssim_std=ss,temporal_difference_error_mean=tm,temporal_difference_error_std=ts,motion_vector_error_pixels_mean=me,motion_vector_error_pixels_std=mestd));print(f'patterns={n:4d} JointMotionTGNN PSNR={pm:.2f} SSIM={sm:.3f} motion_err={me:.3f}')
 csv_save(out/'training_history.csv',hist);csv_save(out/'benchmark_detail.csv',detail);csv_save(out/'benchmark_summary.csv',summary);(out/'results.json').write_text(json.dumps(dict(config=dict(run_seed=run_seed,fixed_data_and_pattern_seed=SEED,epochs=args.epochs,motion_loss_weight=args.motion_weight,initialization='pretrained EstimatedMotionTGNN',uses_ground_truth_motion_at_inference=False),history=hist,summary=summary,elapsed_seconds=time.time()-start),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
