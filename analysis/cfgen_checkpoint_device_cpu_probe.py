"""Read-only native restore/count replay readiness; not a comparator quality fit."""
import hashlib,json,os,sys,datetime
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
OLD=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/execution-control-20261103')
sys.path.insert(0,str(OLD));sys.path.insert(0,'/ml2/alz/scvi_calibration_runtime')
from target_model import restore,sample,seed,torch,NETWORK,SOURCE
from state_fingerprint import fingerprint
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checkpoint=OLD/'target-pilot-output/sealed.ckpt'
assert sha(checkpoint)=='e57885edd7609054ae7575459edceb82055c781cb4fa72fa60390913d4b983e2'
q=torch.load(checkpoint,map_location='cpu',weights_only=False)
model=restore(checkpoint,'cpu')
assert fingerprint(model.state_dict())==fingerprint(q['state_dict'])
assert len(q['state_dict'])==59
assert all(v.device.type=='cpu' for v in model.state_dict().values())
assert all(p.device.type=='cpu' for f in model.feature_embeddings.values() for p in f.parameters())
assert all(str(f.device)=='cpu' for f in model.feature_embeddings.values())
assert not model.training
labels=torch.arange(4).repeat_interleave(2);exposure=torch.full((8,1),float(torch.tensor(6000.).log()))
seed(77412);counts=sample(model,labels,exposure)
assert tuple(counts.shape)==(8,2000) and bool(torch.isfinite(counts).all() and (counts>=0).all() and (counts==counts.floor()).all())
assert not torch.cuda.is_initialized() and not NETWORK
sources=[Path(__file__),OLD/'target_model.py',Path('/ml2/alz/scvi_calibration_runtime/state_fingerprint.py'),*SOURCE.rglob('*.py')]
receipt={'status':'CFGEN_EXISTING_FAILED_PILOT_NATIVE_CPU_DEVICE_EXPLICIT_RESTORE_AND_COUNTS_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_path':str(checkpoint),'checkpoint_SHA256':sha(checkpoint),'original_registered_state_tensors':59,'registered_state_fingerprint':fingerprint(q['state_dict']),'registered_state_load_exact':True,'external_embedding_state_fingerprint':fingerprint({name:f.state_dict() for name,f in model.feature_embeddings.items()}),'all_registered_and_external_modules_on_cpu':True,'explicit_feature_device_attributes_cpu':True,'sample_count_shape':list(counts.shape),'native_sample_counts_fingerprint':fingerprint(counts),'new_model_updates':0,'new_real_data_files_read':False,'GPU_context_initialized':torch.cuda.is_initialized(),'GPU_placement_or_replay_proven':False,'native_checkpoint_resaved':False,'checkpoint_unchanged_after_sampling':sha(checkpoint)=='e57885edd7609054ae7575459edceb82055c781cb4fa72fa60390913d4b983e2','quality_metric_computed':False,'eligible_matched_comparator_model':False,'source_SHA256':{str(p):sha(p) for p in sources},'scope':'Existing4AE/8FM-step resource-pilot checkpoint trained on historical57fittingdonors; not current33donor matched model. Unmodified target_model.restore and upstream CFGen map registered/external tensors and explicit feature device together in fresh CPU process. GPU inference/resource qualification and a properly fitted matched comparator remain required. Two ODE steps here are the retained resource probe, not a final CFGen quality protocol.'}
destination=Path(sys.argv[1]);assert not destination.exists();destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','native_sample_counts_fingerprint','GPU_placement_or_replay_proven','quality_metric_computed')}))
