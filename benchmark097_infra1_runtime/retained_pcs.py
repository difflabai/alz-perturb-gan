"""Keep exact generated stage PCs without consuming RNG or altering samples."""
from pathlib import Path
import numpy as np

def retain(factory,folder):
 def iterator(schedule,condition,seed):
  stage=int(round(float(condition[3])*3))
  if stage not in range(4) or not np.isclose(float(condition[3]),stage/3,rtol=0,atol=1e-6):raise ValueError('Exact native query stage required')
  path=Path(folder)/f'stage{stage}_all_generated_PCs.npy'
  if path.exists():raise FileExistsError('Retained stage output is one-use')
  target=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(200,400,50))
  next_start=0
  try:
   for item in factory(schedule,condition,seed):
    start,pcs,exemplar,totals=item
    if start!=next_start or pcs.dtype!=np.float32 or pcs.ndim!=3 or pcs.shape[1:]!=(400,50) or not len(pcs) or start+len(pcs)>200 or not np.isfinite(pcs).all():raise ValueError('Complete ordered native float32 PC bags required')
    target[start:start+len(pcs)]=pcs;next_start+=len(pcs)
    yield item
   if next_start!=200:raise ValueError('Incomplete native stage output retained, never accepted as complete')
   target.flush()
  finally:
   del target
 return iterator
