"""以 Estimated-motion TGNN 的相同重建器初始化，训练真值运动 Oracle 对照。"""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;PROJECT=Path(r"D:\download\DynGhost-mini-repro")
def save(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT);ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--output-dir',type=Path,default=HERE/'fair_oracle_tgnn_outputs');ap.add_argument('--initial-checkpoint',type=Path,default=HERE/'estimated_motion_tgnn_outputs'/'best_estimated_motion_tgnn.pt');args=ap.parse_args();project=args.project_root.resolve();sys.path.insert(0,str(project))
 from mini_repro_dynghost import IMAGE_SIZE,SEED,make_patterns
 from train_frame_cnn import reconstruction_loss
 from train_mixed_sampling import RATES,coarse_sequence,stats,temporal_error
 from train_oracle_motion_tgnn import BATCH,LR,TEST_N,TRAIN_N,VAL_N,MovingMNISTWithMotion,OracleMotionTGNN,set_seed
 def evaluate(model,loader,pat,dev):
  model.eval();rows=[]
  with torch.no_grad():
   for target,labels,motion in loader:
    target,motion=target.to(dev),motion.to(dev);pred=model(coarse_sequence(target,pat),motion);b,t,c,h,w=target.shape;mse=(pred-target).square().mean((1,2,3,4));psnr=10*torch.log10(1/mse.clamp_min(1e-12));score=ssim(pred.reshape(b*t,c,h,w),target.reshape(b*t,c,h,w),data_range=1.,size_average=False).reshape(b,t).mean(1);terr=((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean((1,2,3,4))
    for i in range(b):rows.append(dict(label=int(labels[i]),mse=float(mse[i]),psnr_db=float(psnr[i]),ssim=float(score[i]),temporal_difference_error=float(terr[i])))
  return rows
 out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);set_seed(SEED);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');bank={n:make_patterns(n,IMAGE_SIZE,IMAGE_SIZE,SEED).to(dev) for n in RATES};train=DataLoader(MovingMNISTWithMotion(project/'data',True,TRAIN_N,SEED+200),batch_size=BATCH,shuffle=True);val=DataLoader(MovingMNISTWithMotion(project/'data',False,VAL_N,SEED+201),batch_size=BATCH);test=DataLoader(MovingMNISTWithMotion(project/'data',False,TEST_N,SEED+300),batch_size=4)
 model=OracleMotionTGNN().to(dev);source=torch.load(args.initial_checkpoint,map_location=dev);model.load_state_dict({k.removeprefix('reconstructor.'):v for k,v in source.items() if k.startswith('reconstructor.')});opt=torch.optim.AdamW(model.parameters(),lr=LR*.5,weight_decay=1e-4);best=1e9;hist=[];ckpt=out/'best_fair_oracle_tgnn.pt';start=time.time()
 for ep in range(1,args.epochs+1):
  model.train();total=num=0
  for bi,(target,_,motion) in enumerate(train):
   target,motion=target.to(dev),motion.to(dev);pred=model(coarse_sequence(target,bank[RATES[(bi+ep-1)%len(RATES)]]),motion);flat=target.reshape(-1,1,IMAGE_SIZE,IMAGE_SIZE);loss=reconstruction_loss(pred.reshape_as(flat),flat)+.15*temporal_error(pred,target);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*target.shape[0];num+=target.shape[0]
  vals=[]
  for n in RATES:vals += [r['mse'] for r in evaluate(model,val,bank[n],dev)]
  vm=float(np.mean(vals));hist.append(dict(epoch=ep,train_loss=total/num,val_mse=vm));print(f'epoch={ep:02d}/{args.epochs} train_loss={total/num:.6f} val_mse={vm:.6f}')
  if vm<best:best=vm;torch.save(model.state_dict(),ckpt)
 model.load_state_dict(torch.load(ckpt,map_location=dev));detail=[];summary=[]
 for n in RATES:
  rows=evaluate(model,test,bank[n],dev)
  for i,r in enumerate(rows):detail.append(dict(pattern_count=n,sampling_ratio=n/IMAGE_SIZE**2,method='FairOracleMotionTGNN',sequence_id=i,**r))
  mm,ms=stats(rows,'mse');pm,ps=stats(rows,'psnr_db');sm,ss=stats(rows,'ssim');tm,ts=stats(rows,'temporal_difference_error');summary.append(dict(pattern_count=n,sampling_ratio=n/IMAGE_SIZE**2,method='FairOracleMotionTGNN',mse_mean=mm,mse_std=ms,psnr_mean_db=pm,psnr_std_db=ps,ssim_mean=sm,ssim_std=ss,temporal_difference_error_mean=tm,temporal_difference_error_std=ts));print(f'patterns={n:4d} FairOracleMotionTGNN PSNR={pm:.2f} SSIM={sm:.3f}')
 save(out/'training_history.csv',hist);save(out/'benchmark_detail.csv',detail);save(out/'benchmark_summary.csv',summary);(out/'results.json').write_text(json.dumps(dict(config=dict(seed=SEED,epochs=args.epochs,initialization='same pretrained estimated-motion reconstructor',motion='simulator ground truth; diagnostic only'),history=hist,summary=summary,elapsed_seconds=time.time()-start),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
