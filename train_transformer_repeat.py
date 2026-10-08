"""在固定数据/散斑条件下重复训练交替时空 Transformer。"""
import argparse,csv,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent; PROJECT=Path(r"D:\download\DynGhost-mini-repro")
def save(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT);ap.add_argument('--run-seed',type=int,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--epochs',type=int,default=10);a=ap.parse_args();p=a.project_root.resolve();sys.path.insert(0,str(p))
 from mini_repro_dynghost import IMAGE_SIZE,SEED,make_patterns
 from train_frame_cnn import reconstruction_loss
 from train_mixed_sampling import RATES,coarse_sequence,temporal_error
 from train_temporal_convgru import MovingMNISTSequences
 from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
 from train_temporal_transformer import evaluate,stats
 random.seed(a.run_seed);np.random.seed(a.run_seed);torch.manual_seed(a.run_seed);torch.cuda.manual_seed_all(a.run_seed);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');bank={n:make_patterns(n,IMAGE_SIZE,IMAGE_SIZE,SEED).to(dev) for n in RATES};train=DataLoader(MovingMNISTSequences(p/'data',True,800,SEED+200),batch_size=8,shuffle=True);val=DataLoader(MovingMNISTSequences(p/'data',False,120,SEED+201),batch_size=8);test=DataLoader(MovingMNISTSequences(p/'data',False,100,SEED+300),batch_size=4)
 m=MiniSpatioTemporalTransformer().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-4);best=1e9;hist=[];ckpt=out/'best_transformer.pt';start=time.time()
 for ep in range(1,a.epochs+1):
  m.train();total=num=0
  for bi,(target,_) in enumerate(train):
   target=target.to(dev);coarse=coarse_sequence(target,bank[RATES[(bi+ep-1)%len(RATES)]]);pred=m(coarse);flat=target.reshape(-1,1,IMAGE_SIZE,IMAGE_SIZE);loss=reconstruction_loss(pred.reshape_as(flat),flat)+.15*temporal_error(pred,target);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();total+=loss.item()*target.shape[0];num+=target.shape[0]
  vals=[]
  for n in RATES:vals += [r['mse'] for r in evaluate(m,'SpatioTemporalTransformer',val,bank[n],dev)]
  vm=float(np.mean(vals));hist.append(dict(epoch=ep,train_loss=total/num,val_mse=vm));print(f'epoch={ep:02d}/{a.epochs} train_loss={total/num:.6f} val_mse={vm:.6f}')
  if vm<best:best=vm;torch.save(m.state_dict(),ckpt)
 m.load_state_dict(torch.load(ckpt,map_location=dev));detail=[];summary=[]
 for n in RATES:
  rows=evaluate(m,'SpatioTemporalTransformer',test,bank[n],dev)
  for i,r in enumerate(rows):detail.append(dict(pattern_count=n,sampling_ratio=n/IMAGE_SIZE**2,method='SpatioTemporalTransformer',sequence_id=i,**r))
  mm,ms=stats(rows,'mse');pm,ps=stats(rows,'psnr_db');sm,ss=stats(rows,'ssim');tm,ts=stats(rows,'temporal_difference_error');summary.append(dict(pattern_count=n,sampling_ratio=n/IMAGE_SIZE**2,method='SpatioTemporalTransformer',mse_mean=mm,mse_std=ms,psnr_mean_db=pm,psnr_std_db=ps,ssim_mean=sm,ssim_std=ss,temporal_difference_error_mean=tm,temporal_difference_error_std=ts));print(f'patterns={n:4d} Transformer PSNR={pm:.2f} SSIM={sm:.3f}')
 save(out/'benchmark_summary.csv',summary);save(out/'benchmark_detail.csv',detail);(out/'results.json').write_text(json.dumps(dict(config=dict(run_seed=a.run_seed,fixed_data_and_pattern_seed=SEED,epochs=a.epochs),history=hist,summary=summary,elapsed_seconds=time.time()-start),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
