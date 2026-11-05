"""Independent CPU audit of native CFGen GPU state/count replay and cleanup."""
import argparse,json,os,sys,datetime
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
def finish(attempt):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 f=ROOT/'artifacts/allocations'/attempt;dep=b.read(f/'deployment.json');auth=b.read(dep['authorization_path']);scope.validate_authorization(auth)
 assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(scope.CELL)
 sys.path.extend(map(str,b.native_site_roots(scope.CELL,auth['files_SHA256'])))
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from state_fingerprint import fingerprint
 ledger=Path(auth['ledger_path']);j=b.read(ledger)['jobs'][0];settle=b.read(ledger.with_name(attempt+'-settlement.json'))
 assert j['status']=='completed' and j['exit_code']==0 and j['reservation_released'] is True
 assert settle['uncertain'] is None and settle['settlement']['status']=='settled'
 assert not b.read(Path(auth['registry_path'])/'registry.json')['jobs']
 out=Path(auth['output_dir']);q=b.read(out/'completed.json');writer=b.read(out/'writer.json');reader=b.read(out/'reader.json');cpu=b.read(scope.MODEL_PROOF)
 assert q['operation']=='qualify_cfgen_gpu_reload' and q['qualification_only'] is True and q['new_model_updates']==0 and q['quality_score_computed'] is False
 assert q['fresh_GPU_processes']==2 and q['registered_external_state_and_counts_replayed'] is True
 assert q['source_checkpoint_SHA256']==b.sha(scope.CHECKPOINT)==auth['checkpoint_SHA256']
 assert q['writer_SHA256']==b.sha(out/'writer.json') and q['reader_SHA256']==b.sha(out/'reader.json') and q['payload_SHA256']==b.sha(out/'writer.pt')
 payload=torch.load(out/'writer.pt',map_location='cpu',weights_only=False)
 assert len(payload['registered_state'])==59
 assert fingerprint(payload['registered_state'])==writer['registered_fingerprint']==reader['registered_fingerprint']==cpu['registered_state_fingerprint']
 assert fingerprint(payload['external_state'])==writer['external_fingerprint']==reader['external_fingerprint']==cpu['external_embedding_state_fingerprint']
 counts=payload['expected_counts'];assert counts.shape==(256,2000)
 assert bool(torch.isfinite(counts).all() and (counts>=0).all() and (counts==counts.floor()).all())
 assert fingerprint(counts)==writer['count_fingerprint']==reader['count_fingerprint']
 for row in (writer,reader):
  assert row['placement']['registered_tensor_devices']==row['placement']['external_parameter_devices']==['cuda:0']
  assert all(p=='cuda:0' for p in row['placement']['explicit_feature_devices'])
  assert row['new_model_updates']==0 and row['quality_score_computed'] is False
 assert not torch.cuda.is_initialized()
 result={'status':'NATIVE_CFGEN_GPU_RELOAD_FRESH_PROCESS_STATE_AND_COUNTS_REPLAY_AUDITED_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,'owned_release_and_clean_settlement_verified':True,'independent_CPU_saved_state_and_counts_audit_passed':True,'native_two_fresh_GPU_processes_count_replay_passed':True,'registered_state_fingerprint':writer['registered_fingerprint'],'external_state_fingerprint':writer['external_fingerprint'],'native_GPU_count_fingerprint':writer['count_fingerprint'],'checkpoint_SHA256':b.sha(scope.CHECKPOINT),'native_GPU_sample_shape':[256,2000],'native_sample_seconds_writer':writer['native_sampling_seconds'],'native_sample_seconds_reader':reader['native_sampling_seconds'],'native_peak_reserved_bytes_writer':writer['native_peak_reserved_bytes'],'native_peak_reserved_bytes_reader':reader['native_peak_reserved_bytes'],'charged_GPU_seconds':settle['settlement']['charged_GPU_seconds'],'model_updates':0,'quality_score_computed':False,'matched_CFGen_quality_available':False,'source_SHA256':{str(p):b.sha(p) for p in [Path(__file__),scope.PROTOCOL,scope.MODEL_PROOF,scope.CHECKPOINT,f/'authorization.json',out/'completed.json',out/'writer.json',out/'reader.json',out/'writer.pt']},'scope':'Actual bounded nativeGPUresource/reload qualification from preserved4AE/8FM-step historical57donor checkpoint. Not a current33-donor matchedmodel or quality/SOTA result; real matchedfit and fixed calibration still required.'}
 p=ROOT/'analysis'/(attempt+'-qualification.json');assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','native_sample_seconds_writer','native_sample_seconds_reader','charged_GPU_seconds')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
