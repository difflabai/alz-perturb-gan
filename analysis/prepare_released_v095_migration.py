"""Prepare a versioned latest-release adapter; never edit historical environments."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/ml2/alz')
OLD = ROOT / 'analysis/released_tuning'
NEW = ROOT / 'analysis/released_tuning_v095'
RELEASE = ROOT / 'artifacts/native-releases/particlegan-0.9.5'
SITE = RELEASE / 'site-packages'
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not NEW.exists()
    NEW.mkdir()
    installation = read(RELEASE / 'installation_and_source_diff.json')
    pins = installation['all0.9.5_Python_sources_SHA256']
    metadata = SITE / 'particlegan-0.9.5.dist-info/METADATA'
    native = dict(status='OFFICIAL_PYPI095_WHEEL_EXTRACTED_SOURCE_PINNED_UNMODIFIED',
                  version='0.9.5', official_wheel_SHA256=installation['official_wheel_SHA256'],
                  official_wheel_source_SHA256=pins, package_metadata_path=str(metadata),
                  package_metadata_SHA256=sha(metadata), site_packages=str(SITE),
                  old_environment_modified=False)
    (NEW / 'native_release.json').write_text(json.dumps(native, indent=2) + '\n')
    env = '''"""Pin the actual imported official release, independently of historical Python sites."""
from pathlib import Path
import importlib.metadata
import sys
SITE=Path('/ml2/alz/artifacts/native-releases/particlegan-0.9.5/site-packages')
VERSION='0.9.5'
PENALTY_OPTIONS={'input_domain':'continuous'}
def activate():
    if 'particlegan' not in sys.modules:
        sys.path.insert(0,str(SITE))
    import particlegan
    if not Path(particlegan.__file__).resolve().is_relative_to(SITE/'particlegan'):
        raise RuntimeError('Historical or unqualified ParticleGAN import origin')
    if importlib.metadata.version('particlegan')!=VERSION:
        raise RuntimeError('ParticleGAN distribution version differs')
    return particlegan
'''
    (NEW / 'release_env.py').write_text(env)
    for name in ('released_update.py', 'released_adapter.py'):
        source = (OLD / name).read_text()
        source = source.replace('sys.path.insert(0,str(B))', 'sys.path.insert(0,str(B))\nfrom release_env import activate,VERSION,PENALTY_OPTIONS\nactivate()')
        if name == 'released_adapter.py':
            source = source.replace('pen=rec.make_critic_penalty(od)', 'pen=rec.make_critic_penalty(od,**PENALTY_OPTIONS)')
            source = source.replace("q['variant_config']=dict(CONFIG);", "q['particlegan_version']=VERSION;q['penalty_options']=dict(PENALTY_OPTIONS)\n q['variant_config']=dict(CONFIG);")
            source = source.replace("def restore(ctx,q,*,expected_endpoint=PG_UPDATES):\n", "def restore(ctx,q,*,expected_endpoint=PG_UPDATES):\n if q.get('particlegan_version')!=VERSION or q.get('penalty_options')!=PENALTY_OPTIONS:raise ValueError('native release or input domain checkpoint differs')\n")
        ast.parse(source); (NEW / name).write_text(source)
    protocol = read(OLD / 'protocol.json')
    for setting in protocol['stage1']['configs']:
        setting['id'] = 'v095_' + setting['id']
    protocol.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    owner_latest_steering='new particlegan elease make sure you are using it',
                    priority='Official0.9.5 only for new GPU training; first reproduce best20k rates before additional scalar tuning.',
                    native_version='0.9.5', native_API=protocol['native_API'].replace('ParticleGAN0.9.4', 'ParticleGAN0.9.5') + '; make_critic_penalty(...,input_domain="continuous")',
                    penalty_options={'input_domain':'continuous'},
                    native_latest_verification_path=str(ROOT / 'analysis/particlegan_new_release_check_20261105T0434.json'),
                    prior0_9_4_protocol_path=str(OLD / 'protocol.json'), prior0_9_4_protocol_SHA256=sha(OLD / 'protocol.json'),
                    historical0_9_4_rate_results_completed=8, historical0_9_4_results_counted_as0_9_5=False,
                    version_bridge={'candidate_id':'v095_lr_g050_d050_p050','fixed_endpoint':20000,'seed':0,
                        'parameters':protocol['stage1']['configs'][0]['parameters'],
                        'historical_result_path':str(ROOT/'analysis/CAL_GPU0_V1_released_rate_g050d050p050_resume1_f0_pg_s0-result.json'),
                        'use_historical_checkpoint_as_new_version_parent':False,
                        'rationale':'Best of8 prior-version rate settings. Test the released0.9.5 path with explicit continuous domain at the same20k budget.40k extension was negative; do not repeat it as the default.'},
                    version_boundary='New prefixed IDs; new-version metrics kept separate. Old results retain source/version/provenance and are contextual hypotheses only. This is prospective0.9.5 research, not retroactive0.9.4 qualification.')
    (NEW / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    qualifier = (OLD / 'qualify_cpu.py').read_text()
    qualifier = qualifier.replace('import numpy as np\n', 'from release_env import activate,VERSION,PENALTY_OPTIONS\nactivate()\nimport numpy as np\n')
    qualifier = qualifier.replace("actual.pop('native_recipe');actual.pop('released_hyperparameters')", "actual.pop('native_recipe');actual.pop('released_hyperparameters');actual.pop('particlegan_version');actual.pop('penalty_options')")
    extra = '''
    def test_latest_native_domain_arithmetic_and_recipe_constructor_migration(self):
        from particlegan import Recipe
        self.assertEqual(VERSION,'0.9.5')
        with self.assertRaises(TypeError): Recipe()
        rec=get_recipe('gan',trainer='bispf_optimized',total_steps=20000,batch_size=32,num_particles=512,z_dim=8)
        self.assertEqual(Recipe.from_dict(rec.to_dict()).to_dict(),rec.to_dict())
        for width in (4,170,1024):
            G=torch.nn.Linear(8,width);D=torch.nn.Linear(width,1)
            with torch.no_grad():D.weight.fill_(.2);D.bias.fill_(.3)
            P=rec.make_prior(generator=torch.Generator().manual_seed(444))
            og,od=rec.make_optimizers(G,D,P)
            pen=rec.make_critic_penalty(od,input_domain='continuous',collect_stats=True)
            real=torch.randn(32,width,generator=torch.Generator().manual_seed(445))
            fake=real*.5
            value=pen(D,real,fake)
            self.assertEqual(pen.regularizer.input_domain,'continuous')
            self.assertAlmostEqual(pen.last_stats['mean_grad_norm'],.4 if width==4 else .2,places=6)
            value.backward();self.assertTrue(torch.isfinite(D.weight.grad).all())

    def test_checkpoint_binds_actual_release_and_explicit_domain_before_restore(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache,fullwidth=True)
            update.configure_hyperparameters(PROTOCOL['stage1']['configs'][0]['parameters'])
            b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
            self.assertEqual(b[6].regularizer.input_domain,'continuous')
            adapter.advance(ctx,b,g,r,u,execution_limit=20000)
            q=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
            self.assertEqual(q['particlegan_version'],'0.9.5');self.assertEqual(q['penalty_options'],PENALTY_OPTIONS)
            before=fingerprint(q)
            for key,value in [('particlegan_version','0.9.4'),('penalty_options',{'input_domain':'auto'}),('penalty_options',None)]:
                bad=copy.deepcopy(q);bad[key]=value
                with self.assertRaisesRegex(ValueError,'native release or input domain'):adapter.restore(ctx,bad)
            historical=copy.deepcopy(q);historical.pop('particlegan_version');historical.pop('penalty_options')
            with self.assertRaisesRegex(ValueError,'native release or input domain'):adapter.restore(ctx,historical)
            self.assertEqual(before,fingerprint(adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')))

'''
    qualifier = qualifier.replace("\n\nif __name__=='__main__':", '\n' + extra + "\nif __name__=='__main__':")
    qualifier = qualifier.replace("ROOT/'routed_resume_runtime/state_fingerprint.py']", "ROOT/'routed_resume_runtime/state_fingerprint.py',HERE/'release_env.py',HERE/'native_release.json']")
    qualifier = qualifier.replace("native=json.loads((ROOT/'analysis/particlegan_release_reverification_20261104T1538.json').read_text())", "native=json.loads((HERE/'native_release.json').read_text())\n    paths.extend(Path(p) for p in native['official_wheel_source_SHA256'])\n    paths.append(Path(native['package_metadata_path']))")
    qualifier = qualifier.replace("'tests_passed':result.testsRun,", "'tests_passed':result.testsRun,'native_version':VERSION,'penalty_options':PENALTY_OPTIONS,'explicit_continuous_domain_arithmetic_and_version_bound_restore_verified':True,")
    qualifier = qualifier.replace('Original released native package and live E22 sources unchanged.', 'Actual isolated official0.9.5 imported. Public continuous domain declared; original ALZ update/streams preserved. Baseline parity is against original ALZ adapter running under0.9.5 auto150D, not a cross-version quality or full20k parity claim. Old0.9.4 package and prior sources unchanged.')
    ast.parse(qualifier); (NEW / 'qualify_cpu.py').write_text(qualifier)
    print(json.dumps({'status':'PREPARED_OFFICIAL095_VERSION_BOUND_MODEL_CPU_QUALIFICATION_PENDING','directory':str(NEW),'official_Python_sources':len(pins),'GPU_job_started':False}))


if __name__ == '__main__':
    main()
