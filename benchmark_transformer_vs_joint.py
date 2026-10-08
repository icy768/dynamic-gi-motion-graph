"""同一 DGI 输入上的 Transformer / Joint TGNN 推理成本基准。"""
import json,sys,time
from pathlib import Path
import torch
import torch.nn as nn
HERE=Path(__file__).resolve().parent;PROJECT=Path(r"D:\download\DynGhost-mini-repro")
def measure(model,x,device,repeats=50):
 model.eval()
 with torch.no_grad():
  for _ in range(10):model(x)
  if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
  start=time.perf_counter()
  for _ in range(repeats):model(x)
  if device.type=='cuda':torch.cuda.synchronize()
 return {'parameters':sum(p.numel() for p in model.parameters()),'latency_ms':(time.perf_counter()-start)*1000/repeats,'peak_memory_mb':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.}
def main():
 sys.path[:0]=[str(PROJECT),str(HERE)]
 from mini_repro_dynghost import IMAGE_SIZE,SEED,make_patterns
 from train_mixed_sampling import coarse_sequence
 from train_oracle_motion_tgnn import MovingMNISTWithMotion,OracleMotionTGNN
 from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
 from train_dgi_motion_estimator import DgiMotionNet
 from torch.utils.data import DataLoader
 class Joint(nn.Module):
  def __init__(self):super().__init__();self.motion_net=DgiMotionNet();self.reconstructor=OracleMotionTGNN()
  def forward(self,x):
   b,t,_,h,w=x.shape;pair=torch.cat((x[:,:-1],x[:,1:]),2).reshape(b*(t-1),2,h,w);motion=self.motion_net(pair).reshape(b,t-1,2).mean(1).clamp(-5,5);return self.reconstructor(x,motion)
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');target,_,_=next(iter(DataLoader(MovingMNISTWithMotion(PROJECT/'data',False,1,SEED+300),batch_size=1)));x=coarse_sequence(target.to(dev),make_patterns(256,IMAGE_SIZE,IMAGE_SIZE,SEED).to(dev))
 tr=MiniSpatioTemporalTransformer().to(dev);tr.load_state_dict(torch.load(PROJECT/'spatiotemporal_transformer_outputs'/'best_spatiotemporal_transformer.pt',map_location=dev));joint=Joint().to(dev);joint.load_state_dict(torch.load(HERE/'joint_motion_tgnn_outputs'/'best_joint_motion_tgnn.pt',map_location=dev))
 result={'config':{'input':'one 8-frame 64x64 DGI sequence','pattern_count':256,'repeats':50,'dgi_generation_included':False},'SpatioTemporalTransformer':measure(tr,x,dev),'JointMotionTGNN':measure(joint,x,dev)}
 (HERE/'transformer_vs_joint_efficiency.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
