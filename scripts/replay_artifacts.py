"""Replay saved final predictions; this does not train the original models."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'submission_mean_final_v5.csv':'1e4060c84d292eecd81b957c1cf9c1af61321866d9c9bac3b5e534268f12470f',
 'submission_v14_final_decomposed_guided.csv':'1015761f1edefcb0f3614816d717bcc3a41a876fd12fe00debdc89dc0fe1e36b',
}

def load_arrays(path):
 with np.load(path,allow_pickle=False) as data:
  arrays={k:data[k].copy() for k in data.files}
 for k,v in arrays.items():
  if v.shape!=(9408,) or not np.isfinite(v).all():raise ValueError(f'Invalid array: {path.name}:{k}')
 return arrays

def mean_blend():
 comp=load_arrays(ROOT/'mean_research/final_v5_subspace_ensemble/component_models.npz')
 m1,m2,m3,m4,m5,m6,m7,m8,m9=[comp[f'm{i}'] for i in range(1,10)]
 correction=np.load(ROOT/'mean_research/final_v5_subspace_ensemble/d_v5_orth.npy',allow_pickle=False)
 if correction.shape!=(9408,) or not np.isfinite(correction).all():raise ValueError('Invalid correction')
 # Final blend arithmetic from the competition report. The RMSE constants below
 # are public-leaderboard feedback; they are not held-out validation labels.
 D5=np.column_stack([m1-m3,m2-m3,m4-m3,m5-m3,m6-m3])
 G5=D5.T@D5/len(m3)
 R3=.17766
 R5=np.array([.18334,.17865,.17719,.17650,.17774])
 b5=np.array([.5*(R3**2-r**2+G5[i,i]) for i,r in enumerate(R5)])
 s5=m3+D5@np.linalg.solve(G5+1e-6*np.eye(5),b5)
 physical=(m7-s5)/.50
 D6=np.column_stack([D5,physical])
 d8=(m8-m3)-D6@np.linalg.solve(D6.T@D6,D6.T@(m8-m3))
 D7=np.column_stack([D5,physical,d8])
 d9=(m9-m3)-D7@np.linalg.solve(D7.T@D7,D7.T@(m9-m3))
 D8=np.column_stack([m1-m3,m2-m3,m4-m3,m5-m3,m6-m3,m7-m3,m8-m3,m9-m3])
 G8=D8.T@D8/len(m3)
 R8=np.array([.18334,.17865,.17719,.17650,.17774,.17331,.17295,.17269])
 b8=np.array([.5*(R3**2-r**2+G8[i,i]) for i,r in enumerate(R8)])
 values,vectors=np.linalg.eigh(G8)
 order=np.argsort(values)[::-1]
 values,vectors=values[order],vectors[:,order]
 coeff=np.zeros(8)
 for k in range(5):coeff[k]=(vectors[:,k]@b8)/(values[k]+1.2e-5)
 spectral=m3+D8@(vectors@coeff)
 step=s5+.86*physical+2.12*d8+1.95*d9
 return np.clip(.75*step+.25*spectral+.55*correction,0,1)

def reference(name,columns):
 path=ROOT/'submissions'/name
 if hashlib.sha256(path.read_bytes()).hexdigest()!=FILES[name]:raise ValueError(f'Checksum mismatch: {name}')
 frame=pd.read_csv(path)
 if list(frame.columns)!=columns or len(frame)!=9408 or not frame.id.is_unique:raise ValueError(f'Invalid schema: {name}')
 if not np.isfinite(frame.to_numpy()).all() or not frame.mu.between(0,1).all():raise ValueError(f'Invalid values: {name}')
 if 'sigma' in columns and not (frame.sigma>0).all():raise ValueError('Nonpositive sigma')
 return frame

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=ROOT/'reproduced')
 args=parser.parse_args()
 mean=reference('submission_mean_final_v5.csv',['id','mu'])
 unc=reference('submission_v14_final_decomposed_guided.csv',['id','mu','sigma'])
 if not mean.id.equals(unc.id):raise ValueError('Submission ID order differs')
 cached=load_arrays(ROOT/'uncertainty_research/final_v14_components.npz')
 # Cached arrays have no embedded IDs. Alignment to this checksum-verified
 # submission snapshot assumes the original competition row order.
 replays={
  'submission_mean_final_v5.csv':pd.DataFrame({'id':mean.id,'mu':mean_blend()}),
  'submission_v14_final_decomposed_guided.csv':pd.DataFrame({'id':unc.id,'mu':np.clip(cached['mu_final'],0,1),'sigma':np.clip(cached['sigma_final'],.01,.50)}),
 }
 args.output_dir.mkdir(parents=True,exist_ok=True)
 for name,frame in replays.items():
  ref=mean if len(frame.columns)==2 else unc
  error=float(np.max(np.abs(frame.iloc[:,1:].to_numpy()-ref.iloc[:,1:].to_numpy())))
  if error>1e-7:raise ValueError(f'Replay differs: {name}: {error}')
  path=args.output_dir/name
  frame.to_csv(path,index=False,float_format='%.8f',lineterminator='\n')
  exact=hashlib.sha256(path.read_bytes()).hexdigest()==FILES[name]
  print(f'{name}: 9408 rows; max absolute error={error:.3g}; exact CSV hash={exact}')
 print('PASS: cached artifact replay. Model training and leaderboard scores were not recomputed.')
if __name__=='__main__':main()
