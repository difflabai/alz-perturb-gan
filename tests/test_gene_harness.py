import copy
import dataclasses
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from alz_perturb_gan.gene_features import FeatureTable, file_sha256
from alz_perturb_gan.gene_protocol import StudyProtocol, StudyData
from alz_perturb_gan.gene_models import ModelConfig, build_models
from alz_perturb_gan.gene_checkpoint import state_hash, model_hashes, save_checkpoint, load_checkpoint, zero_update_guard
from alz_perturb_gan.gene_training import train_arm
from alz_perturb_gan.gene_evaluation import evaluate, paired_delta, all_seed_deltas, negative_indices, score_summary


@pytest.fixture
def study():
    names=tuple('ABCDEF');ids=tuple(f'ENSG_TEST_{n}' for n in names)
    protocol=StudyProtocol(names,ids,('A','B','C'),('D',),('E','F'),draws=3,data_kind='synthetic')
    rng=torch.Generator().manual_seed(10)
    cells={n:torch.randn(8,4,generator=rng)+i*.2 for i,n in enumerate(names)}
    cell_ids={n:tuple(f'{n}-{i}' for i in range(8)) for n in names}
    return StudyData.partition(protocol,torch.ones(4),cells,cell_ids,source_provenance={'data_sha256':'synthetic unit fixture'})


@pytest.fixture
def table(study):
    raw=np.random.default_rng(2).normal(size=(6,3)).astype('float32');raw[-1]=0
    return FeatureTable.from_array(study.protocol.library,study.protocol.gene_ids,raw,[False]*5+[True],study.protocol.fit,{'source':'synthetic'},verified=False)


def cfg(arm='B0',seed=42,**kw):
    return ModelConfig(4,6,hidden=8,z_dim=2,particles=8,seed=seed,steps=2,batch_size=4,arm=arm,**kw)


def test_feature_standardization_uses_fit_only_and_keeps_missing_rows(study):
    raw=np.arange(18,dtype='float32').reshape(6,3);raw[-1]=0
    a=FeatureTable.from_array(study.protocol.library,study.protocol.gene_ids,raw,[False]*5+[True],study.protocol.fit,{})
    changed=raw.copy();changed[3:5]=1e6
    b=FeatureTable.from_array(study.protocol.library,study.protocol.gene_ids,changed,[False]*5+[True],study.protocol.fit,{})
    assert torch.equal(a.mean,b.mean) and torch.equal(a.scale,b.scale) and torch.equal(a.values[:3],b.values[:3])
    assert torch.equal(a.values[-1],torch.tensor([0.,0.,0.,1.]))


def _sealed(tmp,study):
    matrix=np.random.default_rng(4).normal(size=(6,3)).astype('float32');matrix[-1]=0
    np.save(tmp/'matrix.npy',matrix)
    order={'names':list(study.protocol.library),'gene_ids':list(study.protocol.gene_ids),'missing':[False]*5+[True],
           'source_ids':[f'source-{i}' for i in range(5)]+[None]}
    (tmp/'order.json').write_text(json.dumps(order))
    man={'schema_version':1,'status':'frozen','coverage_kind':'usable_vectors','species':9606,
         'source':{'name':'unit fixture','revision':'fixture-v1','url':'https://example.test/fixture'},
         'matrix':{'path':'matrix.npy','sha256':file_sha256(tmp/'matrix.npy')},
         'order':{'path':'order.json','sha256':file_sha256(tmp/'order.json')},'dimension':3,
         'transform':{'kind':'identity','fit_scope':'external_frozen','fit_names':[],'uses_expression_outcomes':False},
         'license':{'asset_class':'annotation_data','license_id':'synthetic-test','license_url':'https://example.test/license',
                    'license_revision':'v1','license_sha256':'fixture-license-digest','citation':'synthetic unit fixture',
                    'local_research_reviewed':True,'code_license':'MIT'}}
    (tmp/'manifest.json').write_text(json.dumps(man));return tmp/'manifest.json',man


@pytest.mark.parametrize('fault',['draft','resolution','hash','stable_id','transform','outcomes','license'])
def test_feature_manifest_fails_closed(tmp_path,study,fault):
    path,man=_sealed(tmp_path,study)
    if fault=='draft':man['status']='draft'
    if fault=='resolution':man['coverage_kind']='identifier_resolution'
    if fault=='hash':man['matrix']['sha256']='0'*64
    if fault=='stable_id':
        q=json.loads((tmp_path/'order.json').read_text());q['gene_ids'][0]='WRONG';(tmp_path/'order.json').write_text(json.dumps(q));man['order']['sha256']=file_sha256(tmp_path/'order.json')
    if fault=='transform':man['transform'].update(fit_scope='fit_only',fit_names=list(study.protocol.library))
    if fault=='outcomes':man['transform']['uses_expression_outcomes']=True
    if fault=='license':man['license']['local_research_reviewed']=False
    path.write_text(json.dumps(man))
    with pytest.raises(ValueError):FeatureTable.load(path,file_sha256(path),names=study.protocol.library,gene_ids=study.protocol.gene_ids,fit_names=study.protocol.fit)


def test_verified_table_load_and_no_silent_library_exclusion(tmp_path,study):
    path,_=_sealed(tmp_path,study)
    loaded=FeatureTable.load(path,file_sha256(path),names=study.protocol.library,gene_ids=study.protocol.gene_ids,fit_names=study.protocol.fit)
    assert loaded.verified and loaded.values.shape==(6,4) and loaded.names==study.protocol.library


@pytest.mark.parametrize('arm',['G','H','GH','shuffle'])
def test_projection_gradients_inside_forward_raw_features_remain_frozen(study,table,arm):
    models=build_models(cfg(arm),table,data_kind='synthetic')
    ids=torch.tensor([0,1,2]);control=torch.ones(3,4);z=torch.randn(3,2,generator=torch.Generator().manual_seed(0))
    if models.generator.encoder is not None:
        models.generator(z,control,ids).square().mean().backward()
        enc=models.generator.encoder
        assert enc.projection.weight.grad is not None and enc.projection.weight.grad.norm()>0
        assert enc.raw.grad is None and not enc.raw.requires_grad
    if models.decision.encoder is not None:
        models.decision.logits(control,control+.2,control-.1,ids).square().mean().backward()
        enc=models.decision.encoder
        assert enc.projection.weight.grad is not None and enc.projection.weight.grad.norm()>0
        assert enc.raw.grad is None


def test_initial_backbones_and_active_shuffle_capacity_match(study,table):
    arms={a:build_models(cfg(a),table if a!='B0' else None,data_kind='synthetic') for a in ('B0','G','H','GH','shuffle')}
    assert len({state_hash(m.generator.backbone) for m in arms.values()})==1
    assert len({state_hash(m.decision.backbone) for m in arms.values()})==1
    assert len({state_hash(m.critic) for m in arms.values()})==1
    assert arms['GH'].counts()==arms['shuffle'].counts()
    shuffled=table.shuffled(142)
    assert torch.equal(shuffled.values[:,-1],table.values[:,-1]) and shuffled.names==table.names


def test_training_boundary_excludes_inner_exploratory_and_oracle_is_disjoint(study):
    before=study.training_tensors()
    changed=copy.deepcopy(study)
    for n in study.protocol.inner+study.protocol.exploratory:
        changed.cells[n].fill_(float('nan'));changed.query_profiles[n].fill_(1e6)
    after=changed.training_tensors()
    assert all(torch.equal(before[k],after[k]) for k in before)
    for n in study.protocol.library:assert not set(study.candidate_ids[n])&set(study.query_ids[n])
    with pytest.raises(ValueError):study.selection_view('exploratory')


def test_real_arm_and_budget_guards(study,table):
    with pytest.raises(ValueError):build_models(cfg('G'),table,data_kind='real')
    real=dataclasses.replace(study,protocol=dataclasses.replace(study.protocol,data_kind='real'))
    with pytest.raises(PermissionError):train_arm(build_models(cfg()),real)


def test_tiny_matched_arm_training_streams_and_generator_equivalence(study,table):
    arms={a:build_models(cfg(a),table if a!='B0' else None,data_kind='synthetic') for a in ('B0','H','G','GH','shuffle')}
    reports={a:train_arm(m,study,decision_cells=2,max_wall_seconds=30) for a,m in arms.items()}
    assert len({r['streams']['cell_batch'] for r in reports.values()})==1
    assert len({r['streams']['pairs'] for r in reports.values()})==1
    assert len({r['streams']['g_latents'] for r in reports.values()})==1
    assert state_hash(arms['B0'].generator)==state_hash(arms['H'].generator)
    assert state_hash(arms['G'].generator)==state_hash(arms['GH'].generator)
    assert all(r['steps']==2 and not r['selection_scoring_during_fit'] for r in reports.values())


def test_frozen_generator_probe_updates_head_only(study,table):
    source=build_models(cfg('G'),table,data_kind='synthetic');train_arm(source,study,decision_cells=2)
    probe=build_models(cfg('G_frozen',seed=43),table,data_kind='synthetic')
    old_head=state_hash(probe.decision)
    train_arm(probe,study,decision_cells=2,frozen_source=source)
    assert state_hash(probe.generator)==state_hash(source.generator) and state_hash(probe.prior)==state_hash(source.prior)
    assert old_head!=state_hash(probe.decision)


def test_zero_guard_rejects_optimizer_attempt_even_when_caught():
    p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.Adam([p])
    with pytest.raises(RuntimeError,match='attempted'):
        with zero_update_guard():
            try:optimizer.step()
            except RuntimeError:pass


def test_checkpoint_restore_is_zero_update_and_hash_bound(tmp_path,study,table,monkeypatch):
    models=build_models(cfg('GH'),table,data_kind='synthetic');path=tmp_path/'checkpoint.pt'
    digest=save_checkpoint(path,models,study.protocol,table)
    restored=load_checkpoint(path,digest)
    assert model_hashes(models)==model_hashes(restored)
    with pytest.raises(ValueError):load_checkpoint(path,'0'*64)
    import alz_perturb_gan.gan as legacy
    monkeypatch.setattr(legacy,'train_gan_decision',lambda *a,**k:pytest.fail('trainer called'))
    base=build_models(cfg());p=tmp_path/'legacy.pt'
    args={'hidden':8,'seed':42,'steps':2,'batch_size':4,'lr':.0002}
    torch.save({'generator':base.generator.backbone.state_dict(),'critic':base.critic.state_dict(),
                'decision':base.decision.backbone.state_dict(),'prior':base.prior.state_dict(),'args':args},p)
    loaded=load_checkpoint(p,file_sha256(p))
    assert state_hash(loaded.generator.backbone)==state_hash(base.generator.backbone)


def test_evaluation_records_exact_inputs_pairs_four_groups_and_no_updates(tmp_path,study):
    models=build_models(cfg());before=model_hashes(models);out=tmp_path/'audit'
    scored=evaluate(models,study,output=out,invocation={'checkpoint_sha256':'synthetic mock'})
    assert set(p.name for p in out.iterdir())=={'inputs','pairs','metrics','provenance'}
    assert before==model_hashes(models) and scored['summary']['optimizer_step_calls']==0
    tensors=torch.load(out/'inputs/scored_tensors.pt',weights_only=True)
    assert tensors['candidates'].shape==(6,4) and torch.equal(tensors['candidates'],scored['inputs']['candidates'])
    pairs=torch.load(out/'pairs/matrices.pt',weights_only=True)
    assert pairs['exploratory']['head_scores'].shape==(2,6)
    hashes=json.loads((out/'provenance/file_hashes.json').read_text())
    assert all(file_sha256(out/k)==v for k,v in hashes.items())
    assert json.loads((out/'provenance/manifest.json').read_text())['zero_optimizer_step_calls']==0
    with pytest.raises(FileExistsError):evaluate(models,study,output=out)


def test_oracle_context_uses_independent_cells_and_fixed_input_reuse(study):
    models=build_models(cfg());first=evaluate(models,study,profile_mode='independent_real_oracle')
    second=evaluate(models,study,fixed_inputs=first['inputs'])
    assert torch.equal(first['pairs']['inner']['head_scores'],second['pairs']['inner']['head_scores'])
    bad=copy.deepcopy(first['inputs']);bad['candidate_cell_ids']['A']=bad['query_cell_ids']['A']
    with pytest.raises(ValueError,match='overlap'):evaluate(models,study,fixed_inputs=bad)


def test_negative_identity_rule_is_order_independent_and_auc_ties_are_correct(study):
    a=negative_indices(list(study.protocol.library),['A','B','E'],46)
    b=negative_indices(list(study.protocol.library),['E','A','B'],46)
    assert torch.equal(a,b[[1,2,0]])
    report=score_summary(torch.full((2,6),.5),['A','E'],list(study.protocol.library),torch.tensor([1,0]),probabilities=True)
    assert report['macro_auroc']==.5 and report['micro_auroc']==.5 and report['balanced_brier']==.25


def test_paired_uncertainty_keeps_all_seeds_and_target_units(study):
    scored={s:evaluate(build_models(cfg(seed=s)),study)['summary'] for s in (42,43,44)}
    report=paired_delta(scored[42],scored[42],'exploratory',draws=20)
    assert report['mean_delta']==0 and report['ci95']==[0.,0.]
    all_seeds=all_seed_deltas(scored,scored,'exploratory',expected_seeds=[42,43,44],draws=20)
    assert len(all_seeds['per_seed'])==3 and all_seeds['hierarchical_ci95']==[0.,0.] and all_seeds['selected_seed'] is None
    with pytest.raises(ValueError):all_seed_deltas({42:scored[42]},scored,'exploratory',expected_seeds=[42,43,44],draws=20)


def test_tau_packet_adapter_reorders_by_stable_id_and_admits_only_valid_string(tmp_path,study):
    source_names=list(reversed(study.protocol.library));source_ids=list(reversed(study.protocol.gene_ids))
    a=np.arange(18,dtype=np.float32).reshape(6,3)+1;a[0]=0;np.save(tmp_path/'string.npy',a)
    packet={'gene_order':{'symbols':source_names,'ensembl':source_ids},'sources':{'string_spectral':{
        'url':'https://example.test/string-v12','file':'string.npy','sha256':file_sha256(tmp_path/'string.npy'),'shape':[6,3],'dim':3,
        'version':'STRING v12.0','license_model_or_data':'CC BY 4.0','all_zero_rows':['F'],'missing':['F'],
        'fit':{'gene_universe':'all 10 human proteins','score_threshold':'combined_score >= 700',
               'input_sha256':'graph-digest','id_map_sha256':'alias-digest'}}}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(packet));digest=file_sha256(path)
    supplements=_tau_addons(tmp_path,path,packet)
    t=FeatureTable.load_tau_packet(path,digest,source_key='string_spectral',names=study.protocol.library,gene_ids=study.protocol.gene_ids,fit_names=study.protocol.fit,**supplements)
    assert t.names==study.protocol.library and t.missing[-1] and t.verified
    assert t.provenance['transform_scope']=='external_graph_outcome_free'
    with pytest.raises(ValueError):FeatureTable.load_tau_packet(path,digest,source_key='go_svd',names=study.protocol.library,gene_ids=study.protocol.gene_ids,fit_names=study.protocol.fit)


def test_bundle_roundtrip_keeps_exact_cell_partitions(tmp_path,study):
    from alz_perturb_gan.gene_harness import save_study_bundle,load_study_bundle
    path=tmp_path/'bundle.pt';digest=save_study_bundle(path,study);loaded=load_study_bundle(path,digest)
    assert loaded.protocol.fingerprint==study.protocol.fingerprint
    assert loaded.candidate_ids==study.candidate_ids and loaded.query_ids==study.query_ids
    assert all(torch.equal(loaded.query_profiles[n],study.query_profiles[n]) for n in study.protocol.library)


def test_gpu_grant_requires_explicit_lease_and_protocol(study):
    from alz_perturb_gan.gene_harness import validate_grant
    good={'approved':True,'authorization_reference':'synthetic test','phases':['train','evaluate'],'arms':['B0'],'seeds':[42],
          'protocol_sha256':study.protocol.fingerprint,'max_steps_per_arm':2,'max_wall_seconds_per_arm':20}
    assert validate_grant(good,study,cfg(),'cpu','train',10)==good
    with pytest.raises(PermissionError):validate_grant(good,study,cfg(),'cuda:0','train',10)
    with pytest.raises(PermissionError):validate_grant(good,study,cfg(),'cpu','train',21)
    with pytest.raises(PermissionError):validate_grant({**good,'protocol_sha256':'different'},study,cfg(),'cpu','train',10)


def test_cli_zero_update_evaluation_with_exact_invocation(tmp_path,study):
    from alz_perturb_gan.gene_harness import save_study_bundle,main
    bundle=tmp_path/'bundle.pt';bd=save_study_bundle(bundle,study)
    checkpoint=tmp_path/'checkpoint.pt';cd=save_checkpoint(checkpoint,build_models(cfg()),study.protocol)
    grant={'approved':True,'authorization_reference':'synthetic unit test','phases':['evaluate'],'arms':['B0'],'seeds':[42],
           'protocol_sha256':study.protocol.fingerprint,'max_steps_per_arm':2,'max_wall_seconds_per_arm':20}
    gp=tmp_path/'grant.json';gp.write_text(json.dumps(grant));out=tmp_path/'evaluation'
    main(['evaluate','--bundle',str(bundle),'--bundle-sha256',bd,'--output',str(out),'--steps','2','--batch-size','4','--hidden','8',
          '--z-dim','2','--particles','8','--max-wall-seconds','20','--grant',str(gp),'--grant-sha256',file_sha256(gp),
          '--checkpoint',str(checkpoint),'--checkpoint-sha256',cd])
    manifest=json.loads((out/'provenance/manifest.json').read_text())
    assert manifest['checkpoint_provenance']['sha256']==cd
    assert manifest['data_provenance']['bundle_sha256']==bd
    assert manifest['zero_optimizer_step_calls']==0


def test_frozen_source_protocol_mismatch_fails(study):
    source=build_models(cfg());probe=build_models(cfg('G_frozen',frozen_generator_arm='B0'))
    with pytest.raises(ValueError,match='protocol'):train_arm(probe,study,frozen_source=source)


def test_real_evaluator_never_accepts_unbound_fresh_models(study):
    real=dataclasses.replace(study,protocol=dataclasses.replace(study.protocol,data_kind='real'))
    with pytest.raises(ValueError,match='hash-bound'):evaluate(build_models(cfg()),real)


def test_feature_order_and_fit_scope_cannot_silently_mismatch(study,table):
    wrong=dataclasses.replace(table,names=tuple(reversed(table.names)))
    with pytest.raises(ValueError,match='identity'):evaluate(build_models(cfg('G'),wrong,data_kind='synthetic'),study)
    wrong=dataclasses.replace(table,fit_names=('A','B'))
    with pytest.raises(ValueError,match='FIT'):train_arm(build_models(cfg('H'),wrong,data_kind='synthetic'),study)


def test_evaluation_budget_is_bounded_without_mutation(study):
    models=build_models(cfg());before=model_hashes(models)
    with pytest.raises(TimeoutError):evaluate(models,study,max_wall_seconds=0)
    assert before==model_hashes(models)


def test_training_records_gradient_scale_direction_saturation_and_update(study):
    report=train_arm(build_models(cfg()),study,decision_cells=2)
    row=report['history'][0]
    assert row['reconstruction_grad_norm']>0 and row['weighted_adversarial_grad_norm']>0
    assert -1.00001<=row['component_gradient_cosine']<=1.00001
    assert row['g_relative_parameter_update']>0 and row['prior_grad_norm']>=0
    assert 0<=row['j_real_saturation_fraction']<=1 and row['generated_within_candidate_variance']>=0


def test_gpu_grant_rejects_timezone_ambiguous_expiry(study,monkeypatch):
    from alz_perturb_gan.gene_harness import validate_grant
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-fixture')
    grant={'approved':True,'authorization_reference':'synthetic test','phases':['train'],'arms':['B0'],'seeds':[42],
           'protocol_sha256':study.protocol.fingerprint,'max_steps_per_arm':2,'max_wall_seconds_per_arm':20,
           'gpu_uuid':'GPU-fixture','capacity_receipt_sha256':'fixture','ownership_reference':'fixture',
           'lease_receipt':'fixture','capacity_valid_until':'2099-01-01T00:00:00'}
    with pytest.raises(PermissionError,match='timezone'):validate_grant(grant,study,cfg(),'cuda:0','train',10)


def test_missing_coverage_is_reported_and_oracle_is_not_paired_to_generated(study,table):
    models=build_models(cfg('G'),table,data_kind='synthetic')
    generated=evaluate(models,study)['summary'];oracle=evaluate(models,study,profile_mode='independent_real_oracle')['summary']
    strata=generated['splits']['exploratory']['coverage_strata']
    assert strata['observed_feature']['target_count']==1 and strata['missing_feature']['target_count']==1
    assert strata['missing_feature']['head']['candidate_count']==6
    with pytest.raises(ValueError,match='profile mode'):paired_delta(oracle,generated,'exploratory')


def test_serializing_untrained_model_cannot_create_fitted_probe_source(tmp_path,study,table):
    source=build_models(cfg('G'),table,data_kind='synthetic');p=tmp_path/'fresh.pt'
    digest=save_checkpoint(p,source,study.protocol,table);loaded=load_checkpoint(p,digest)
    assert loaded.training_protocol_sha256 is None
    assert not evaluate(loaded,study)['summary']['selection_eligible']
    with pytest.raises(ValueError,match='protocol'):train_arm(build_models(cfg('G_frozen'),table,data_kind='synthetic'),study,frozen_source=loaded)


def test_fixed_control_and_cell_partition_cannot_misdescribe_scores(study):
    models=build_models(cfg());first=evaluate(models,study)
    wrong=copy.deepcopy(first['inputs']);wrong['control'].add_(1)
    with pytest.raises(ValueError,match='control'):evaluate(models,study,fixed_inputs=wrong)
    wrong=copy.deepcopy(first['inputs']);wrong['candidate_cell_ids']['A']=tuple(reversed(wrong['candidate_cell_ids']['A']))
    with pytest.raises(ValueError,match='partition'):evaluate(models,study,fixed_inputs=wrong)


def _tau_addons(tmp,path,packet):
    entry=packet['sources']['string_spectral'];order=packet['gene_order']
    target={'manifest_sha256':file_sha256(path),'matrix_sha256':entry['sha256'],'source_key':'string_spectral'}
    mapping={'schema_version':1,'status':'frozen','kind':'tau_string_row_mapping_addon','applies_to':target,
             'alias_input_sha256':entry['fit']['id_map_sha256'],'links_input_sha256':entry['fit']['input_sha256'],
             'rule':'first listed Ensembl alias','rows':[{'symbol':n,'gene_id':g,'string_protein_id':'9606.TEST_'+n,
             'missing':n in entry['missing'],'feature_row_zero':n in entry['missing'],
             'edges_ge_700':0 if n in entry['missing'] else 1,'missing_reason':'no_edge' if n in entry['missing'] else None}
             for n,g in zip(order['symbols'],order['ensembl'])]}
    (tmp/'cc_by_4.0_legalcode.txt').write_text('synthetic license fixture')
    (tmp/'string_access_page.html').write_text('synthetic primary page fixture')
    license={'schema_version':1,'status':'frozen','kind':'tau_string_license_receipt_addon','applies_to':target,
             'source':{'name':'STRING','revision':'v12.0','url':entry['url']},'asset_class':'public annotation/network data (supplied table)',
             'license_id':'CC-BY-4.0','local_research_reviewed':True,'license_url':'https://example.test/license','license_revision':'4.0',
             'license_text_sha256':file_sha256(tmp/'cc_by_4.0_legalcode.txt'),'source_license_statement_url':'https://example.test/source',
             'source_license_statement_page_sha256':file_sha256(tmp/'string_access_page.html'),'source_license_statement_fetched_utc':'2026-10-14T00:00:00Z',
             'citation':'synthetic fixture','attribution_and_modification_notice':'synthetic fixture','code_license':'n/a'}
    mp=tmp/'mapping.json';lp=tmp/'license.json';mp.write_text(json.dumps(mapping));lp.write_text(json.dumps(license))
    return {'mapping_path':mp,'expected_mapping_sha256':file_sha256(mp),'license_path':lp,'expected_license_sha256':file_sha256(lp)}


def _real_bundle_fixture(tmp,study):
    import h5py
    real=copy.deepcopy(study);real.protocol=dataclasses.replace(real.protocol,data_kind='real')
    symbols=['A','B','C','D'];gene_ids=list(real.protocol.gene_ids[:4]);raw=tmp/'raw.h5ad'
    with h5py.File(raw,'w') as f:
        var=f.create_group('var');var.attrs['_index']='_index'
        var.create_dataset('_index',data=symbols,dtype=h5py.string_dtype());var.create_dataset('gene_id',data=gene_ids,dtype=h5py.string_dtype())
        obs=f.create_group('obs');obs.attrs['_index']='_index'
        ids=[c for n in real.protocol.library for c in real.cell_ids[n]]+['control-id']
        labels=[n for n in real.protocol.library for c in real.cell_ids[n]]+['non-targeting']
        obs.create_dataset('_index',data=ids,dtype=h5py.string_dtype());obs.create_dataset('target_gene',data=labels,dtype=h5py.string_dtype())
    # Perturbation E/F identities must also occur in the expression namespace.
    # Expand the synthetic expression fixture to the six-symbol source namespace.
    real.control=torch.ones(6);real.cells={n:torch.cat([x,torch.zeros(len(x),2)],1) for n,x in real.cells.items()}
    with h5py.File(raw,'a') as f:
        del f['var/_index'];del f['var/gene_id']
        f['var'].create_dataset('_index',data=list(real.protocol.library),dtype=h5py.string_dtype())
        f['var'].create_dataset('gene_id',data=list(real.protocol.gene_ids),dtype=h5py.string_dtype())
    split=tmp/'split.json';split.write_text(json.dumps({'train_names':list(real.protocol.fit+real.protocol.inner),'eval_names':list(real.protocol.exploratory)}))
    real.source_provenance={'protocol_sha256':real.protocol.fingerprint,'source':{'path':str(raw),'sha256':file_sha256(raw)},
            'split_source':{'path':str(split),'sha256':file_sha256(split)},'expression_gene_ids':list(real.protocol.gene_ids),
            'expression_gene_symbols':list(real.protocol.library),'control':{'cell_ids':['control-id']}}
    return StudyData.partition(real.protocol,real.control,real.cells,real.cell_ids,source_provenance=real.source_provenance)


def test_real_bundle_binds_protocol_namespace_and_source_cell_labels(tmp_path,study):
    from alz_perturb_gan.gene_harness import save_study_bundle,load_study_bundle
    real=_real_bundle_fixture(tmp_path,study);p=tmp_path/'bundle.pt';digest=save_study_bundle(p,real)
    loaded=load_study_bundle(p,digest,expected_protocol_sha256=real.protocol.fingerprint)
    assert loaded.source_provenance['identity_contract']['raw_source_metadata_verified']
    with pytest.raises(ValueError,match='granted'):load_study_bundle(p,digest,expected_protocol_sha256='wrong')
    blob=torch.load(p,weights_only=True);blob['source_provenance']['expression_gene_ids'].reverse();torch.save(blob,p)
    with pytest.raises(ValueError,match='gene order'):load_study_bundle(p,file_sha256(p))


@pytest.mark.parametrize('fault',['duplicate_partition','empty_partition','wrong_cell_identity','wrong_stable_id','wrong_original_order'])
def test_real_bundle_contract_rejects_identity_partition_faults(tmp_path,study,fault):
    from alz_perturb_gan.gene_harness import save_study_bundle,load_study_bundle
    real=_real_bundle_fixture(tmp_path,study);p=tmp_path/'bundle.pt';save_study_bundle(p,real);blob=torch.load(p,weights_only=True)
    if fault=='duplicate_partition':blob['partition']['candidate_ids']['A']+=blob['partition']['candidate_ids']['A'][:1]
    if fault=='empty_partition':blob['partition']['query_ids']['A']=()
    if fault=='wrong_cell_identity':blob['cell_ids']['A']=tuple(blob['cell_ids']['B'])
    if fault=='wrong_stable_id':
        blob['protocol']['gene_ids']=('WRONG',)+blob['protocol']['gene_ids'][1:]
        from alz_perturb_gan.gene_features import json_hash
        blob['source_provenance']['protocol_sha256']=json_hash(blob['protocol'])
    if fault=='wrong_original_order':
        sp=Path(blob['source_provenance']['split_source']['path']);d=json.loads(sp.read_text());d['train_names'].reverse();sp.write_text(json.dumps(d))
        blob['source_provenance']['split_source']['sha256']=file_sha256(sp)
    torch.save(blob,p)
    with pytest.raises(ValueError):load_study_bundle(p,file_sha256(p))


def test_grant_bound_cuda_cap_is_applied_without_gpu_execution(monkeypatch):
    from alz_perturb_gan.gene_harness import apply_cuda_allocator_cap
    calls=[]
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda n:type('Props',(),{'total_memory':16*1024**3})())
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda fraction,device:calls.append((fraction,device)))
    apply_cuda_allocator_cap('cuda:0',4,{'cuda_allocator_cap_gib':4})
    assert calls==[(.25,0)]
    with pytest.raises(PermissionError):apply_cuda_allocator_cap('cuda:0',4,{'cuda_allocator_cap_gib':2})
    with pytest.raises(PermissionError):apply_cuda_allocator_cap('cpu',4,{'cuda_allocator_cap_gib':4})


@pytest.mark.parametrize('fault',['packet_binding','row_order','stable_id','missing_mask','license_review','license_bytes','supplement_hash'])
def test_string_supplements_fail_closed_on_essential_bindings(tmp_path,study,fault):
    names=list(study.protocol.library);ids=list(study.protocol.gene_ids)
    matrix=np.ones((6,3),dtype=np.float32);matrix[-1]=0;np.save(tmp_path/'string.npy',matrix)
    packet={'gene_order':{'symbols':names,'ensembl':ids},'sources':{'string_spectral':{
        'url':'https://example.test/string-v12','file':'string.npy','sha256':file_sha256(tmp_path/'string.npy'),
        'shape':[6,3],'dim':3,'version':'STRING v12.0','license_model_or_data':'CC BY 4.0',
        'all_zero_rows':['F'],'missing':['F'],'fit':{'gene_universe':'all 10 human proteins','score_threshold':'combined_score >= 700',
        'input_sha256':'graph','id_map_sha256':'alias'}}}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(packet));extras=_tau_addons(tmp_path,path,packet)
    asset=extras['license_path'] if fault in ('license_review','license_bytes') else extras['mapping_path']
    body=json.loads(asset.read_text())
    if fault=='packet_binding':body['applies_to']['matrix_sha256']='different'
    if fault=='row_order':body['rows'].reverse()
    if fault=='stable_id':body['rows'][0]['gene_id']='WRONG'
    if fault=='missing_mask':body['rows'][-1]['missing']=False
    if fault=='license_review':body['local_research_reviewed']=False
    if fault=='license_bytes':(tmp_path/'cc_by_4.0_legalcode.txt').write_text('changed')
    if fault=='supplement_hash':extras['expected_mapping_sha256']='0'*64
    asset.write_text(json.dumps(body))
    if fault!='supplement_hash':extras['expected_license_sha256' if asset==extras['license_path'] else 'expected_mapping_sha256']=file_sha256(asset)
    with pytest.raises(ValueError):FeatureTable.load_tau_packet(path,file_sha256(path),source_key='string_spectral',names=names,gene_ids=ids,fit_names=study.protocol.fit,**extras)


def test_provenance_git_children_receive_owned_lease_fds(tmp_path,study,monkeypatch):
    import os
    import alz_perturb_gan.gene_evaluation as ev
    fd=os.open(tmp_path/'fake-owner-fd',os.O_CREAT|os.O_RDONLY,0o600)
    seen=[];original=ev.subprocess.run
    def capture(*args,**kwargs):
        seen.append(kwargs.get('pass_fds'));return original(*args,**kwargs)
    try:
        monkeypatch.setenv('ALZ_LEASE_FDS',str(fd));monkeypatch.setattr(ev.subprocess,'run',capture)
        evaluate(build_models(cfg()),study,output=tmp_path/'evaluation')
        assert seen==[(fd,),(fd,)]
    finally:os.close(fd)
