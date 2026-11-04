"""Check the three-way forecast algebra by direct finite enumeration."""
import itertools,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0,'/ml2/alz/analysis')
from estimate_uniform_training_seed_ensemble import expected_multimask_score,empirical_cell_mixture,fixed_counts,require_complete_set

class Identities(unittest.TestCase):
    def test_three_way_fixed_counts_match_all_assignments(self):
        rng=np.random.default_rng(81);forecasts=[rng.normal(size=(4,3)) for _ in range(3)];target=rng.normal(size=(1,3))
        def es(x):return cdist(x,target).mean()-.5*cdist(x,x).sum()/(len(x)*(len(x)-1))
        for counts in ([2,1,1],[0,2,2],[4,0,0]):
            observed=[]
            for mask in itertools.product(range(3),repeat=4):
                if list(np.bincount(mask,minlength=3))==counts:observed.append(es(np.stack([forecasts[m][i] for i,m in enumerate(mask)])))
            expected,_=expected_multimask_score([es(x) for x in forecasts],forecasts,counts)
            self.assertAlmostEqual(expected,np.mean(observed),places=12)
    def test_three_way_cell_energy_matches_direct_weighted_cloud(self):
        clouds=[np.array([[0.],[2.]]),np.array([[-1.],[4.],[5.]]),np.array([[1.],[3.],[7.],[8.]])];target=np.array([[1.],[3.]])
        def ed(x):return 2*cdist(x,target).mean()-cdist(x,x).mean()-cdist(target,target).mean()
        for weights in ([1/3]*3,[.2,.3,.5],[1.,0.,0.]):
            all_points=np.concatenate(clouds);w=np.concatenate([np.full(len(x),a/len(x)) for a,x in zip(weights,clouds)])
            direct=2*(w*cdist(all_points,target).mean(1)).sum()-w@cdist(all_points,all_points)@w-cdist(target,target).mean()
            predicted,_=empirical_cell_mixture([ed(x) for x in clouds],clouds,weights)
            self.assertAlmostEqual(predicted,direct,places=12)
    def test_registered_rounding_keeps_every_model_and_exact200_without_choice(self):
        counts=[fixed_counts(rep,stage) for rep in (0,1) for stage in range(4)]
        self.assertTrue(all(sum(x)==200 and sorted(x)==[66,67,67] for x in counts))
        self.assertEqual(np.sum(counts,axis=0).tolist(),[533,533,534])
    def test_incomplete_original_confirmation_cannot_be_screened(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'set.json';h={'attempts':['zero','one','two']}
            for rows in ([],[{'seed':0,'attempt':'zero'},{'seed':1,'attempt':'one'}],[{'seed':0,'attempt':'zero'},{'seed':2,'attempt':'two'},{'seed':1,'attempt':'one'}]):
                p.write_text(json.dumps({'records':rows}))
                with self.assertRaises(ValueError):require_complete_set(p,h)
    def test_invalid_counts_and_weights_refuse(self):
        clouds=[np.zeros((4,2))]*3
        for counts in ([1,1,1],[-1,2,3],[2.,1.,1.]):
            with self.assertRaises(ValueError):expected_multimask_score([0,0,0],clouds,counts)
        for w in ([1,1,1],[-1,1,1],[float('nan'),0,1]):
            with self.assertRaises(ValueError):empirical_cell_mixture([0,0,0],clouds,w)

if __name__=='__main__':unittest.main()
