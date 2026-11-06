"""Prospective one-dimensional boundary check; selection reads seed0 only."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');BOUNDARY=ROOT/'analysis/hpo500_boundary'
PROTOCOL=BOUNDARY/'prospective_protocol.json';ANCHOR=BOUNDARY/'anchor.json'
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def anchor():return read(ANCHOR)
def control():return read(ROOT/'analysis/hpo500/results/CAL_GPU0_V1_h5_c000_search_f0_pg_s0.json')['metrics']
def feasible(row):
    return row['finite'] and row['metrics']['cell_energy_distance']<=control()['cell_energy_distance'] and .7<=row['metrics']['spread_ratio']<=1.3
def anchor_row():
    a=anchor();return dict(candidate_id=a['anchor_candidate'],hyperparameters=a['anchor_hyperparameters'],metrics=a['anchor_metrics'],finite=True)
def best(records):
    return min([anchor_row()]+[r for r in records if feasible(r)],key=lambda r:(r['metrics']['energy_score'],r['metrics']['cell_energy_distance'],r['candidate_id']))
def refinements(records):
    assert len(records)>=4
    first=records[:4];winner=best(first);a=anchor()
    if winner['candidate_id']==a['anchor_candidate']:return []
    k=a['anchor_hyperparameters']['reg_kappa'];value=winner['hyperparameters']['reg_kappa']
    grid=sorted([k]+[r['hyperparameters']['reg_kappa'] for r in first])
    if value==grid[0]:return [k*.03125,k*.015625]
    index=grid.index(value);assert 0<index<len(grid)-1
    return [math.sqrt(grid[index-1]*value),math.sqrt(value*grid[index+1])]
def proposal(index,records):
    assert 0<=index<6 and len(records)==index
    assert [r['candidate_id'] for r in records]==[f'k{i:03d}' for i in range(index)]
    assert all(r['phase']=='search' and r['seed']==0 and r['endpoint']==500 for r in records)
    a=anchor();assert a['trigger'] and a['confirmation_result_files_present_at_capture']==0
    if index<4:
        raw=a['initial_proposals'][index];hyper=raw['hyperparameters'];method='prospective_boundary_halving';used=[]
    else:
        values=refinements(records);assert index-4<len(values),'Boundary already bracketed without an improvement; no extra fit'
        hyper={**a['anchor_hyperparameters'],'reg_kappa':values[index-4]};method='prospective_geometric_refinement_or_lower_extension';used=[r['attempt'] for r in records[:4]]
    return dict(id=f'k{index:03d}',native_version='0.9.6',endpoint=500,hyperparameters=hyper,anchor_candidate=a['anchor_candidate'],only_parameter_changed='reg_kappa',proposal_method=method,source_results=used,selection_seed=0,heldout_results_used=False,fresh_confirmation_results_used=False,protocol_SHA256=sha(PROTOCOL),anchor_SHA256=sha(ANCHOR))
