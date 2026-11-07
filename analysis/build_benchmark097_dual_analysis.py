from pathlib import Path
import ast
R=Path('/ml2/alz')
p=R/'analysis/aggregate_benchmark097.py';t=p.read_text().replace('benchmark097_runtime','benchmark097_dual_runtime').replace('artifacts/benchmark097-fullmatrix','artifacts/benchmark097-dual-fullmatrix')
t=t.replace('import preset_scope as scope','import preset_scope as scope\nfrom dual_coordination import GPUS,RESOURCE_PROTOCOL,assigned_gpu,validate_resource_protocol\nimport resource_profile as profile')
t=t.replace(" assert not read(ROOT/'artifacts/gpu1-benchmark097-fullmatrix-20261106/live_registry/registry.json')['jobs']", " for gpu in GPUS:assert not read(profile.registry_path(gpu)/'registry.json')['jobs']\n validate_resource_protocol();assert matrix['resource_protocol_SHA256']==identity['resource_protocol_SHA256']==sha(RESOURCE_PROTOCOL)")
t=t.replace("    assert auth['cell']==q['cell']", "    assert auth['gpu_uuid']==assigned_gpu(auth['cell']) and auth['dynamic_shared_gpu0']==(auth['gpu_uuid']==GPUS[0])\n    assert auth['cell']==q['cell']")
t=t.replace("    ledger=read(auth['ledger_path']);assert len(ledger['jobs'])==1;job=ledger['jobs'][0]", "    ledger=read(auth['ledger_path']);assert ledger['gpu_uuid']==auth['gpu_uuid'];assert len(ledger['jobs'])==1;job=ledger['jobs'][0]")
(R/'analysis/aggregate_benchmark097_dual.py').write_text(t)
p=R/'analysis/score_calibration_excluded53.py';t=p.read_text().replace('benchmark097_runtime','benchmark097_dual_runtime').replace('import aggregate_benchmark097 as core','import aggregate_benchmark097_dual as core').replace("ROOT/'analysis/aggregate_benchmark097.py'","ROOT/'analysis/aggregate_benchmark097_dual.py'")
(R/'analysis/score_calibration_excluded53_dual.py').write_text(t)
for original,new,oldimp,newimp in [('test_aggregate_benchmark097.py','test_aggregate_benchmark097_dual.py','aggregate_benchmark097','aggregate_benchmark097_dual'),('test_calibration_excluded53.py','test_calibration_excluded53_dual.py','score_calibration_excluded53','score_calibration_excluded53_dual')]:
 t=(R/'analysis'/original).read_text().replace(oldimp,newimp);(R/'analysis'/new).write_text(t)
for p in [R/'analysis/aggregate_benchmark097_dual.py',R/'analysis/score_calibration_excluded53_dual.py']:
 ast.parse(p.read_text())
def fn(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
for name in ('summarize_points','contrast_interval','main'):
 assert ast.dump(fn(R/'analysis/aggregate_benchmark097.py',name))==ast.dump(fn(R/'analysis/aggregate_benchmark097_dual.py',name)),name
for name in ('validate_plan','repool'):
 assert ast.dump(fn(R/'analysis/score_calibration_excluded53.py',name))==ast.dump(fn(R/'analysis/score_calibration_excluded53_dual.py',name)),name
print('Scientific summary/statistical functions identical; only resource closure and provenance adapter paths differ')
