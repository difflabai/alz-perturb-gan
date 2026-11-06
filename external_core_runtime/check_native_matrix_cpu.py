"""Verify complete native authorization gates for all30 cells, without admission.

Temporary authorization files are source/profile checks only. No Controller,
reservation, resource lease, CUDA import or worker launch is performed here.
"""
import datetime,json,os,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,prepare,preset_scope as scope
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    checks=[]
    with tempfile.TemporaryDirectory(prefix='alz-all30-native-source-wire-') as directory:
        for fold in range(3):
            for method in ('pg','scvi'):
                for seed in range(5):
                    path=Path(directory)/f'f{fold}_{method}_s{seed}.json'
                    auth,_=prepare.build_authorization(fold=fold,method=method,seed=seed,auth_path=path)
                    path.write_text(json.dumps(auth,indent=2)+'\n')
                    verified,_=b.verify(path)
                    assert verified['cell']==scope.cell(fold,method,seed)
                    checks.append(dict(cell=verified['cell'],status='COMPLETE_NATIVE_AUTH_SOURCE_PROFILE_GATE_PASS'))
    assert 'torch' not in sys.modules and len(checks)==30
    result=dict(status='ALL30_COMPLETE_NATIVE_SOURCE_PROFILE_AUTHORIZATIONS_CPU_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                checks=checks,canonical_admission_mutated=False,reservation_created=False,resource_lease_created=False,
                scientific_model_imported=False,GPU_context_created=False,
                source_SHA256={str(p):b.sha(p) for p in HERE.glob('*.py')})
    out=HERE.parent/'analysis/external_core_all30_native_wire_v2_CPU.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','source_SHA256')}))
if __name__=='__main__':main()
