"""Actual native full-width replay and fixed-formulation/source checks."""
import ast,copy,datetime,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[HERE/'model',ROOT/'hpo500_runtime',HERE,ROOT/'analysis/duration096',ROOT/'model',ROOT,ROOT/'routed_resume_runtime',SCIENCE,PROJECT,PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import torch,numpy as np
import released_adapter as model,released_update as update
from native_fixture import context
from state_fingerprint import fingerprint
import preset_scope as scope,search
def node(p,name):return ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
class Model(unittest.TestCase):
    def setUp(self):update.configure(scope.CFG);update.configure_hyperparameters(scope.HYPER)
    def tearDown(self):self.assertFalse(torch.cuda.is_initialized())
    def test_formulation_native_update_initialization_and_model_context_unchanged(self):
        old=ROOT/'analysis/released_tuning_v096'
        self.assertEqual((HERE/'model/released_update.py').read_bytes(),(old/'released_update.py').read_bytes())
        self.assertEqual((HERE/'model/release_env.py').read_bytes(),(old/'release_env.py').read_bytes())
        for name in ('Context','build','training_streams','passive_particle_diagnostics','snapshot'):
            self.assertEqual(node(HERE/'model/released_adapter.py',name),node(old/'released_adapter.py',name))
        self.assertEqual(model.PG_UPDATES,500)
        r=update.recipe();self.assertEqual(r.total_steps,20000);self.assertEqual(r.ema_decay,0)
        self.assertEqual(r.generator_gradient_response,'none')
    def test_native_scalar_public_recipe_domains_and_no_other_fields_change(self):
        baseline=update.resolved_recipe(scope.HYPER).to_dict()
        for k in scope.HYPER:
            for factor in (.25,.5,2,4):
                h={**scope.HYPER,k:scope.HYPER[k]*factor};r=update.resolved_recipe(h);q=r.to_dict()
                changed={key for key in baseline if q[key]!=baseline[key]}
                self.assertFalse(changed-{'lr','d_lr_mult','prior_lr_mult','reg_coeff','reg_kappa','prior_reg'})
                self.assertEqual(r.total_steps,20000);self.assertEqual(r.ema_decay,0)
                self.assertAlmostEqual(r.lr,h['generator_lr']);self.assertAlmostEqual(r.lr*r.d_lr_mult,h['critic_lr']);self.assertAlmostEqual(r.lr*r.prior_lr_mult,h['particle_lr'])
    def test_actual_full2000gene_native_complete_state_and_next_pair_replay(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache,fullwidth=True)
            b,g,r,u,s=model.build(ctx.mu,ctx.theta,'cpu',0,0)
            model.advance(ctx,b,g,r,u,execution_limit=500)
            q=model.snapshot(ctx,b,g,r,u,0,s,'synthetic');self.assertEqual(q['fixed_endpoint'],500)
            self.assertEqual(q['recipe']['total_steps'],20000)
            model.advance(ctx,b,g,r,u,execution_limit=500);expected=model.snapshot(ctx,b,g,r,u,0,s,'synthetic')
            rb,rg,rr,ru,rs=model.restore(ctx,q)
            self.assertEqual(fingerprint(q),fingerprint(model.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic')))
            model.advance(ctx,rb,rg,rr,ru,execution_limit=500)
            self.assertEqual(fingerprint(expected),fingerprint(model.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic')))
            for key,value in [('fixed_endpoint',20000),('completed_updates',501),('status','completed'),('particlegan_version','0.9.5')]:
                bad=copy.deepcopy(q);bad[key]=value
                with self.assertRaises(ValueError):model.restore(ctx,bad)
            with self.assertRaises(RuntimeError):model.fit(ctx,cache/'forbidden',0,end_monotonic=0)
            self.assertFalse((cache/'forbidden').exists())
    def test_structured_then_surrogate_search_deterministic_and_bounded(self):
        records=[]
        for index in range(13):
            p=search.make_proposal(index,records);x=np.asarray(p['log2_coordinates'])
            self.assertEqual(np.count_nonzero(x),0 if index==0 else 1)
            metrics=dict(energy_score=2+float(((x-.4)**2).sum())/10,cell_energy_distance=.2+float((x*x).sum())/100,spread_ratio=1.)
            records.append(dict(candidate_id=p['id'],attempt=p['id'],hyperparameters=p['hyperparameters'],finite=True,metrics=metrics))
        p=search.make_proposal(13,records);self.assertEqual(p,search.make_proposal(13,records))
        x=np.asarray(p['log2_coordinates']);self.assertTrue((x>=-2).all() and (x<=2).all())
        self.assertGreater(np.min(((x-np.stack([search.coordinates(r['hyperparameters']) for r in records]))**2).sum(-1)),1e-6)
        scope.validate_proposal(p)
    def test_posterior_interpolates_and_uncertainty_is_larger_away_from_observations(self):
        x=np.stack([np.zeros(6),np.ones(6)]);y=np.array([1.,2.]);pool=np.stack([np.zeros(6),np.full(6,-2)])
        mean,sd=search.posterior(x,y,pool)
        self.assertAlmostEqual(mean[0],1.,places=4);self.assertGreater(sd[1],sd[0])
    def test_only500_endpoint_and_same_configuration_allowed_in_saved_model_gate(self):
        from worker_gpu0 import validate_pg
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');recipe=update.recipe().to_dict()
            a=dict(cell={'seed':0},cache_dir=str(cache),released_hyperparameters=scope.HYPER)
            q=dict(schema=1,method='pg',native_recipe='bispf_optimized',fold=0,seed=0,status='completed',completed_updates=500,fixed_endpoint=500,particlegan_version='0.9.6',penalty_options={'input_domain':'continuous'},released_hyperparameters=scope.HYPER,variant_config=scope.CFG,recipe=recipe,policy={'completed_steps':500,'recipe':recipe},cache_manifest_SHA256=scope.sha(cache/'manifest.json'))
            validate_pg(q,a)
            for key,value in [('completed_updates',499),('fixed_endpoint',20000),('status','diagnostic'),('particlegan_version','0.9.5')]:
                changed=copy.deepcopy(q);changed[key]=value
                with self.assertRaises(Exception):validate_pg(changed,a)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Model))
    assert result.wasSuccessful() and result.testsRun==6
    paths=[*HERE.glob('*.py'),*scope.MODEL_DIR.glob('*.py'),ROOT/'analysis/released_tuning_v096/released_adapter.py',ROOT/'analysis/released_tuning_v096/released_update.py',ROOT/'analysis/released_tuning_v096/release_env.py',ROOT/'analysis/duration096/native_fixture.py',ROOT/'test_variant.py',scope.NATIVE_PROOF]
    q=dict(status='NATIVE096_FIXED_FORMULATION_500STEP_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tests_passed=6,native_update_byte_identical=True,full2000gene_complete_native_state_next_pair_replay=True,CUDA_initialized=False,GPU_allocation_created=False,source_SHA256={str(p):scope.sha(p) for p in paths})
    scope.MODEL_PROOF.write_text(json.dumps(q,indent=2)+'\n')
