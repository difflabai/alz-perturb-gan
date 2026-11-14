"""Prevent source/control/predecessor metadata from false passing."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pair_common as B
import generate_pair as D
import run_pair as R


def component_fixture():
    train=[f't{i}' for i in range(18)];held=[f'h{i}' for i in range(19)]
    ncells={d:31+i for i,d in enumerate(train+held)}
    info={'baseline':'paired_resample_blind','parts':2,'empirical_role':'component','empirical_stream':0,
          'M':16,'K':256,'kpc':64,'targets':37,'frac_cells_depth_unmatched':.1,
          'source_selection':'training donors only; target excluded when in training',
          'count_rule':'sum_j Binomial(X_j,min(1,L/L_j)/parts)',
          'target_bag_source_donors':{d:B.component_sources(train,d,ncells[d],0,0) for d in train+held}}
    return info,{'train_donors':train,'heldout_donors':held,'donor_cell_counts':ncells}


def test_valid_paired_source_and_count_contract():D.validate_component(*component_fixture(),0,0)


@pytest.mark.parametrize('mutation',['heldout','self','parts','role','repeat','duplicate','bag_count','missing_target',
                                    'permuted_cycle','spoofed_replicate_cycle','spoofed_fold_cycle'])
def test_invalid_paired_component_is_rejected(mutation):
    info,cache=component_fixture();assign=info['target_bag_source_donors']
    if mutation=='heldout':assign['t0'][0]='h0'
    elif mutation=='self':assign['t0'][0]='t0'
    elif mutation=='parts':info['parts']=1
    elif mutation=='role':info['empirical_role']='reference'
    elif mutation=='repeat':info['empirical_stream']=1
    elif mutation=='duplicate':assign['t0'][1]=assign['t0'][0]
    elif mutation=='bag_count':assign['t0'].pop()
    elif mutation=='missing_target':assign.pop('h0')
    elif mutation=='permuted_cycle':assign['t0'][0],assign['t0'][1]=assign['t0'][1],assign['t0'][0]
    elif mutation=='spoofed_replicate_cycle':assign['t0']=B.component_sources(cache['train_donors'],'t0',31,0,1)
    else:assign['t0']=B.component_sources(cache['train_donors'],'t0',31,1,0)
    with pytest.raises(ValueError):D.validate_component(info,cache,0,0)


def previous_fixture():
    status={'status':'completed','screen_passes':False}
    proof={'status':'passed','complete':True,'scope':'full_conditional_study','verified_artifacts':24,
           'required_artifacts':24,'missing':[],'zero_factor_fidelity_verified':True,
           'screen_passes':False,'confirmation_disposition':'not_triggered',
           'prelaunch_manifest_sha256':B.PRIOR_MANIFEST_SHA,
           'artifacts':[{'recipe':name,'fold':f,'replicate':r} for name in B.D.RECIPES for f in range(3) for r in (0,1)]}
    comparison={'phase':'screen','folds':[0,1,2],'candidate':'factor32','verdict':{'status':'complete','passes':False}}
    return status,proof,comparison,{'runtime_sha256':{name:'hash' for name in R.PRIOR_FILES}}


def test_complete_failed_predecessor_is_required():R.validate_previous(*previous_fixture())


@pytest.mark.parametrize('mutation',['running','partial','missing_zero','wrong_pins','empty','duplicate','passing','missing_link'])
def test_predecessor_lineage_cannot_be_reblessed(mutation):
    status,proof,comparison,pinned=previous_fixture()
    if mutation=='running':status['status']='running'
    elif mutation=='partial':proof['complete']=False
    elif mutation=='missing_zero':proof['zero_factor_fidelity_verified']=False
    elif mutation=='wrong_pins':proof['prelaunch_manifest_sha256']='new-pins'
    elif mutation=='empty':proof['artifacts']=[]
    elif mutation=='duplicate':proof['artifacts'][-1]=proof['artifacts'][0]
    elif mutation=='passing':comparison['verdict']['passes']=True
    else:pinned['runtime_sha256'].pop('artifact_audit.json')
    with pytest.raises(ValueError):R.validate_previous(status,proof,comparison,pinned)


def test_old185_sources_cannot_be_reblessed(monkeypatch):
    monkeypatch.setattr(B.D,'source_manifest',lambda:{})
    with pytest.raises(ValueError,match='prior185'):B.source_manifest()


def identity_fixture(tmp_path,monkeypatch):
    root=tmp_path/'results';modes=tmp_path/'modes';root.mkdir()
    monkeypatch.setattr(B,'ROOT',root);monkeypatch.setattr(B,'MODES',modes)
    receipt=modes/'empirical32/fold0/seed0.receipt.json';receipt.parent.mkdir(parents=True)
    fingerprints={name:'a'*64 for name in (*B.H.ARRAYS,*B.H.IDENTITY)}
    receipt.write_text(json.dumps({'array_fingerprints':fingerprints}))
    proof={'passed':True,'parts':1,'fold':0,'replicate':0,'sources':{},
           'reference_receipt_sha256':B.sha(receipt),'control_array_fingerprints':fingerprints,
           'pruned_control_sha256':'b'*64}
    path=root/'single_cell_identity.json';path.write_text(json.dumps(proof))
    return path,proof


def test_full_identity_metadata_is_accepted(tmp_path,monkeypatch):
    identity_fixture(tmp_path,monkeypatch);D.verify_identity({})


@pytest.mark.parametrize('mutation',['source','parts','fingerprint','missing_fingerprint','digest','missing_digest','receipt','unpruned'])
def test_full_identity_requires_exact_canonical_arrays_hashes_and_pruning(tmp_path,monkeypatch,mutation):
    path,proof=identity_fixture(tmp_path,monkeypatch)
    if mutation=='source':proof['sources']={'changed':'code'}
    elif mutation=='parts':proof['parts']=2
    elif mutation=='fingerprint':proof['control_array_fingerprints']['pcs']='g'*64
    elif mutation=='missing_fingerprint':proof['control_array_fingerprints'].pop('pcs')
    elif mutation=='digest':proof['pruned_control_sha256']='short'
    elif mutation=='missing_digest':proof.pop('pruned_control_sha256')
    elif mutation=='receipt':proof['reference_receipt_sha256']='c'*64
    else:
        archive=B.MODES/'identity_control/fold0/seed0.npz';archive.parent.mkdir(parents=True);archive.write_bytes(b'not-pruned')
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):D.verify_identity({})
