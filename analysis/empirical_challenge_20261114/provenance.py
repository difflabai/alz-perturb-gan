"""Hash-only frozen donor/cache/fit anchors, without opening count arrays."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'improve_20261109'
RUNS=HERE.parents[1]/'runs/improve_20261109'
ANCHORS=HERE/'cache_and_fit_anchors.json'
CACHE_FILES={'train/counts.npz','train/metadata.npz','heldout/counts.npz',
             'heldout/metadata.npz','space.npz','type_classifier.pt'}
FIT_FILES=('config.json','fit_receipt.json','final.pt','snap_15000.pt','snap_16000.pt',
           'snap_17000.pt','snap_18000.pt','snap_19000.pt')
PROMOTION_SHA='3fb5bcb23efd58c47b25946a00cf63cd051b40f9b86d1b9ad4f7ec62498dcbdc'
FOLDS_SHA='6e5c587cc9f4e08dfa2c17e420da333c971edc8a3d961dfbdc1ad783433f765a'
AUDIT_SHA='7cd0fef29753be11ce7903741ad5e92a28ae1ae6291a91f3531d83cbe98f9970'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def build_anchors():
    promotion_path=OLD/'default_config_distribution_evidence_20261113.json'
    if sha(promotion_path)!=PROMOTION_SHA:raise ValueError('immutable promotion changed')
    promotion=read(promotion_path)
    audit_path=OLD/'architecture_screen_complete_audit_20261113.json'
    if sha(audit_path)!=AUDIT_SHA:raise ValueError('immutable complete audit changed')
    audit=read(audit_path)
    for name,h in audit['sources'].items():
        if sha(OLD/'r5'/name)!=h:raise ValueError('frozen source anchor changed')
    foldpath=OLD/'r5/folds_r2.json'
    if sha(foldpath)!=FOLDS_SHA:raise ValueError('frozen donor partitions changed')
    definitions=read(foldpath);partitions={p['inner_fold']:p for p in definitions['inner_folds']}
    development=set(definitions['development_donors'])
    if len(development)!=37:raise ValueError('wrong development scope')
    caches,fits={},{}
    for fold in range(9):
        cache=RUNS/f'caches/hvg2000/fold{fold}'
        manifest=read(cache/'manifest.json');part=partitions[fold]
        train,held=manifest['train_donors'],manifest['heldout_donors']
        if (train!=part['train_donors'] or held!=part['heldout_donors']
                or len(train)!=len(set(train)) or len(held)!=len(set(held))
                or set(train)&set(held) or set(train)|set(held)!=development
                or manifest['inner_fold']!=fold or manifest['panel']!='hvg2000'
                or manifest['n_genes']!=2000 or manifest['n_pcs']!=50
                or set(manifest['files_SHA256'])!=CACHE_FILES):
            raise ValueError('wrong cache partition/payload set')
        payloads={name:sha(cache/name) for name in sorted(CACHE_FILES)}
        if payloads!=manifest['files_SHA256']:raise ValueError('cache payload changed')
        caches[str(fold)]={'manifest_sha256':sha(cache/'manifest.json'),'payload_sha256':payloads,
                           'train_donors':train,'heldout_donors':held}
        for seed in (0,1):
            key=f'fold{fold}/seed{seed}'
            fit=RUNS/f'fits_r3/pbulk/hvg2000/{key}'
            config=read(fit/'config.json');receipt=read(fit/'fit_receipt.json')
            old=(audit['matched_original_references'][key] if fold<3
                 else promotion['artifact_audit']['fits'][key])
            if (sha(fit/'config.json')!=old['config_sha256']
                    or config['cache_manifest_SHA256']!=caches[str(fold)]['manifest_sha256']
                    or config['train_donors']!=train or config['heldout_donors']!=held
                    or config['updates']!=20000 or config['flags']!=['pbulk']
                    or config['fold']!=fold or config['seed']!=seed
                    or receipt['completed_updates']!=20000 or receipt['status']!='completed'
                    or sha(fit/'final.pt')!=receipt['final_SHA256']):
                raise ValueError('frozen fit no longer matches old anchors')
            hashes={name:sha(fit/name) for name in FIT_FILES}
            for step in (15000,16000,17000,18000,19000):
                if hashes[f'snap_{step}.pt']!=receipt['snapshots_SHA256'][str(step)]:
                    raise ValueError('snapshot differs from original fit receipt')
            if fold>=3 and any(hashes[name]!=h for name,h in old['checkpoints'].items()):
                raise ValueError('checkpoint differs from immutable promotion')
            fits[key]=hashes
    return {'scope':'Same37 development donors only; immutable prior config/checkpoint anchors',
            'promotion_sha256':PROMOTION_SHA,'complete_audit_sha256':AUDIT_SHA,
            'fold_definitions_sha256':FOLDS_SHA,
            'checkpoint_anchor_scope':'folds3-8 checked against immutable promotion checkpoints; folds0-2 checked against current original fit receipts, then frozen in committed prelaunch anchors. All18 configs checked against immutable old evidence.',
            'caches':caches,'fits':fits}


def verify_cache(fold, payloads=True):
    expected=read(ANCHORS)['caches'][str(fold)]
    cache=RUNS/f'caches/hvg2000/fold{fold}'
    manifest=read(cache/'manifest.json')
    if (sha(cache/'manifest.json')!=expected['manifest_sha256']
            or manifest['train_donors']!=expected['train_donors']
            or manifest['heldout_donors']!=expected['heldout_donors']
            or set(manifest['files_SHA256'])!=CACHE_FILES):
        raise ValueError('frozen cache manifest/donor identities changed')
    if payloads and any(sha(cache/name)!=h for name,h in expected['payload_sha256'].items()):
        raise ValueError('frozen cache payload changed under unchanged manifest')
    return expected


def verify_fit(fold, replicate):
    expected=read(ANCHORS)['fits'][f'fold{fold}/seed{replicate}']
    fit=RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}'
    if set(expected)!=set(FIT_FILES) or any(sha(fit/name)!=h for name,h in expected.items()):
        raise ValueError('frozen fit artifact changed')
    return expected


if __name__=='__main__':
    if ANCHORS.exists():raise FileExistsError('prelaunch anchors already exist')
    ANCHORS.write_text(json.dumps(build_anchors(),indent=2)+'\n')
    print('Verified/froze9 donor caches,54payload hashes and18original fits.')
