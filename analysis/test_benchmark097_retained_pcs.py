"""Archiving preserves samples/RNG and refuses incomplete stage output."""
import ast,sys,tempfile,unittest
from pathlib import Path
R=Path('/ml2/alz');P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(R/'benchmark097_runtime'),str(P/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(P/'.venv312/lib/python3.12/site-packages')]
import numpy as np
from retained_pcs import retain
class Retention(unittest.TestCase):
 def test_exact_samples_counts_totals_and_global_rng_unchanged(self):
  pcs=np.arange(200*400*50,dtype=np.float32).reshape(200,400,50)/1000;counts=np.ones((1,400,2000),dtype=np.float32);totals=np.ones(2000)
  pieces=[(i,pcs[i:i+5],counts,totals) for i in range(0,200,5)]
  state=np.random.get_state()
  with tempfile.TemporaryDirectory() as folder:
   values=list(retain(lambda *args:iter(pieces),folder)(None,np.array([0,0,0,1/3]),72))
   for expected,actual in zip(pieces,values):
    self.assertIs(expected,actual)
   np.testing.assert_array_equal(np.load(Path(folder)/'stage1_all_generated_PCs.npy'),pcs)
   after=np.random.get_state();self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
 def test_partial_stage_refused(self):
  with tempfile.TemporaryDirectory() as folder:
   factory=lambda *args:iter([(0,np.zeros((5,400,50),np.float32),None,np.zeros(2000))])
   with self.assertRaisesRegex(ValueError,'Incomplete'):list(retain(factory,folder)(None,np.array([0,0,0,0]),1))
 def test_duplicate_or_misordered_bags_and_wrong_precision_refused(self):
  for start,dtype in ((5,np.float32),(0,np.float64)):
   with tempfile.TemporaryDirectory() as folder:
    factory=lambda *args:iter([(start,np.zeros((5,400,50),dtype),None,np.zeros(2000))])
    with self.assertRaises(ValueError):list(retain(factory,folder)(None,np.array([0,0,0,0]),1))
 def test_invalid_stage_and_existing_archive_refused(self):
  with tempfile.TemporaryDirectory() as folder:
   with self.assertRaises(ValueError):list(retain(lambda *a:iter([]),folder)(None,np.array([0,0,0,.5]),1))
   (Path(folder)/'stage0_all_generated_PCs.npy').touch()
   with self.assertRaises(FileExistsError):list(retain(lambda *a:iter([]),folder)(None,np.array([0,0,0,0]),1))
 def test_original_pooled_reservoir_reconstructed_exactly_for_unequal_stage_counts(self):
  # Execute only the frozen original reservoir class, with no model imports.
  tree=ast.parse((P/'evaluation.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Reservoir')
  namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'original_Reservoir','exec'),namespace)
  Reservoir=namespace['Reservoir'];original=Reservoir();replay=Reservoir()
  with tempfile.TemporaryDirectory() as folder:
   for stage,donors in enumerate((1,3,2,4)):
    pcs=np.full((200,400,50),stage,dtype=np.float32)+np.arange(200,dtype=np.float32)[:,None,None]/1000
    pieces=[(i,pcs[i:i+5],None,np.ones(2000)) for i in range(0,200,5)]
    list(retain(lambda *args:iter(pieces),folder)(None,np.array([0,0,0,stage/3]),stage))
    saved=np.load(Path(folder)/f'stage{stage}_all_generated_PCs.npy',mmap_mode='r')
    for _ in range(donors):original.add(pcs);replay.add(saved)
   np.testing.assert_array_equal(original.cells,replay.cells)
if __name__=='__main__':unittest.main(verbosity=2)
