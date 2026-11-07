"""Reject missing/relabelled old endpoints and partial new follow-up pairs."""
import copy, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path('/ml2/alz/analysis')))
from released097_followup_design import neighbors, NOMINAL
from test_released097_selection import fixture
from selection import summarize_local

def rows():
    _, source=fixture();controls=[];new=[]
    for cid in ('f000','f001',*[f't{i:03}' for i in range(6)]):
        for raw in source[:2]:
            r=copy.deepcopy(raw);r.update(candidate_id=cid,hyperparameters=dict(NOMINAL),
                charged_GPU_seconds=100.,read_only_previously_measured_reference=True,
                new_model_fitted=False,new_GPU_charge=0,source_result_path=f'/synthetic/{cid}_{raw["seed"]}.json',
                source_result_SHA256='synthetic');controls.append(r)
    for c in neighbors():
        for raw in source[:2]:
            r=copy.deepcopy(raw);r.update(candidate_id=c['id'],hyperparameters=c['hyperparameters'],charged_GPU_seconds=100.)
            new.append(r)
    return neighbors(),new,controls

class CombinedSelection(unittest.TestCase):
    def test_all20_declared_endpoints_and_reference_roles_retained(self):
        c,n,p=rows();q=summarize_local(c,n,p)
        self.assertEqual(len(q['candidates']),10);self.assertEqual(q['required_new_fits'],4)
        self.assertEqual(q['previously_measured_control_fits'],16);self.assertFalse(q['controls_refitted'])
        self.assertEqual(q['reference_id'],'f000')
        self.assertEqual(sum(len(x['per_seed']) for x in q['candidates']),20)
    def test_missing_or_duplicate_old_control_refused(self):
        c,n,p=rows()
        for changed in (p[:-1],p[:-1]+[p[0]]):
            with self.assertRaises(ValueError):summarize_local(c,n,changed)
    def test_partial_or_duplicate_new_pair_refused(self):
        c,n,p=rows()
        for changed in (n[:-1],n[:-1]+[n[0]]):
            with self.assertRaises(ValueError):summarize_local(c,changed,p)
    def test_old_controls_cannot_be_relabelled_or_recharged(self):
        c,n,p=rows();p[0]['new_GPU_charge']=100
        with self.assertRaises(ValueError):summarize_local(c,n,p)
        c,n,p=rows();n[0]['read_only_previously_measured_reference']=True
        with self.assertRaises(ValueError):summarize_local(c,n,p)
if __name__=='__main__':unittest.main(verbosity=2)
