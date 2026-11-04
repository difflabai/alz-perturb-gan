"""Real source closure for control scopes; no allocation or model imports."""
import copy
from pathlib import Path
import sys
import unittest
import binding as b

ROOT=b.HERE.parent;sys.path.insert(0,str(ROOT))
import ensemble_scope as scope

class EnsembleBinding(unittest.TestCase):
    def setUp(self):
        prior=b.read(ROOT/'calibration_deployment.json');self.auth=copy.deepcopy(b.read(prior['authorization_path']))
        research={str(p):b.sha(p) for p in [*ROOT.glob('*.py'),*b.HERE.glob('*.py'),*(ROOT/'model').glob('*.py')]}
        self.auth.update(operation='qualify_pure_ensemble',fixed_endpoint=40000,cap_seconds=900,qualification_only=True,hybrid_config=None,
                         variant_config=scope.CFG,pure_formulation='pg8_learned_ensemble75',ensemble_weight=0.,
                         ensemble_manifest_SHA256=b.sha(ROOT/'ensemble_parents.json'),research_source_SHA256=research,
                         cell={'fold':0,'method':'pg','seed':0,'experiment':'ensemble_source_cpu_only'})
        pins=self.auth['files_SHA256'];pins.update(research)
        for path in [ROOT/'ensemble_parents.json',ROOT/'ensemble_protocol.json',ROOT/'ensemble_plan.json',ROOT/'analysis/ensemble_cpu_qualification.json',ROOT/'analysis/ensemble_parent_checkpoint_audit.json']:
            pins[str(path)]=b.sha(path)
        for proof in [scope.read(ROOT/'analysis/ensemble_cpu_qualification.json'),scope.read(ROOT/'analysis/ensemble_parent_checkpoint_audit.json')]:
            for path in proof['source_SHA256']:pins[path]=b.sha(path)
        for parent in scope.manifest()['parents']:
            p=Path(parent['checkpoint_path'])
            for path in [p,p.with_name('checkpoint_receipt.json')]:pins[str(path)]=b.sha(path)

    def test_complete_control_source_closure_for_both_parents(self):
        for weight in (0.,1.):b.source_closure({**self.auth,'ensemble_weight':weight},b.NATIVE_LAYOUT)

    def test_wrong_weight_seed_endpoint_parent_or_qualification_refuses(self):
        for change in [{'ensemble_weight':.5},{'ensemble_weight':True},{'fixed_endpoint':20000},{'cap_seconds':21600},
                       {'cell':{**self.auth['cell'],'seed':1}},{'hybrid_config':{'pg_bag_fraction':.75}},
                       {'qualification_only':False},{'ensemble_manifest_SHA256':'0'*64}]:
            with self.subTest(change=change),self.assertRaises(b.Refusal):b.source_closure({**self.auth,**change},b.NATIVE_LAYOUT)

    def test_candidate_cannot_use_control_scope_or_disabled_plan(self):
        with self.assertRaises(b.Refusal):
            b.source_closure({**self.auth,'operation':'evaluate_pure_ensemble','qualification_only':False,'ensemble_weight':.25},b.NATIVE_LAYOUT)
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
