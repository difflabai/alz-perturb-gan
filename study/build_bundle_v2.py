"""Plan v4 section 1: derive study_bundle_v2.pt (FIT-80 + calibration-16) from the existing bundle. CPU only.

The existing bundle, its receipt and protocol are read only and never modified; every output is a new file
created exclusively (O_CREAT|O_EXCL|O_NOFOLLOW via create_new). Cells, cell IDs, control, candidate/query
partition and gene order are copied verbatim; only the protocol (fit -> fit minus calibration, new
calibration group) and the provenance (protocol hash, derived_from) change.
"""
import argparse, io, json, sys, time
from pathlib import Path
import torch

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--study-dir', default='/ml2/altgan/studies/gene-feature-20261014')
    ap.add_argument('--old-sha256', default='c50072314cdc12272da083def3322413ab01880a9363fd2d6fbad778ad35ad7b')
    ap.add_argument('--out-dir', default=None, help='defaults to --study-dir')
    a = ap.parse_args(argv)
    from alz_perturb_gan.paths import scoped, create_new
    from alz_perturb_gan.gene_features import file_sha256
    from alz_perturb_gan.gene_protocol import StudyProtocol
    from alz_perturb_gan.gene_calibration import calibration_split, split_hash, CALIBRATION_SEED
    from alz_perturb_gan.gene_harness import load_study_bundle
    study_dir = scoped(a.study_dir); out_dir = scoped(a.out_dir or a.study_dir)
    old = scoped(study_dir / 'study_bundle.pt'); old_receipt = scoped(study_dir / 'bundle_receipt.json')
    if file_sha256(old) != a.old_sha256: raise SystemExit('old bundle sha mismatch')
    blob = torch.load(old, map_location='cpu', weights_only=True)
    if file_sha256(old) != a.old_sha256: raise SystemExit('old bundle changed during read')
    p_old = StudyProtocol(**{k: tuple(v) if isinstance(v, list) else v for k, v in blob['protocol'].items()})
    assert len(p_old.fit) == 96 and len(p_old.inner) == 24 and len(p_old.exploratory) == 30 and p_old.calibration == ()
    cal = tuple(calibration_split(p_old.fit))
    fit = tuple(n for n in p_old.fit if n not in set(cal))
    assert len(cal) == 16 and len(fit) == 80
    pd = dict(blob['protocol']); pd['fit'] = list(fit); pd['calibration'] = list(cal)
    p_new = StudyProtocol(**{k: tuple(v) if isinstance(v, list) else v for k, v in pd.items()})
    prov = dict(blob['source_provenance']); prov['protocol_sha256'] = p_new.fingerprint
    prov['derived_from'] = {'bundle_path': str(old), 'bundle_sha256': a.old_sha256, 'protocol_sha256': p_old.fingerprint,
                            'bundle_receipt_sha256': file_sha256(old_receipt), 'builder': 'build_bundle_v2.py (plan v4 section 1)',
                            'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                            'calibration_rule': f'calibration_split(old.fit, seed={CALIBRATION_SEED}, count=16)',
                            'calibration_split_sha256': split_hash(cal), 'changes': 'protocol.fit 96->80, protocol.calibration 16; nothing else'}
    new = {'schema_version': blob['schema_version'], 'protocol': pd, 'control': blob['control'], 'cells': blob['cells'],
           'cell_ids': blob['cell_ids'], 'partition': blob['partition'], 'source_provenance': prov}
    buf = io.BytesIO(); torch.save(new, buf); data = buf.getvalue()
    bpath = create_new(out_dir, 'study_bundle_v2.pt', data, mode=0o444); digest = file_sha256(bpath)
    again = load_study_bundle(bpath, digest, expected_protocol_sha256=p_new.fingerprint)
    ref = blob
    assert again.protocol.library == p_old.library and again.protocol.gene_ids == p_old.gene_ids
    assert all(tuple(ref['cell_ids'][n]) == again.cell_ids[n] for n in p_old.library)
    assert all(tuple(ref['partition'][k][n]) == getattr(again, k)[n] for k in ('candidate_ids', 'query_ids') for n in p_old.library)
    assert all(torch.equal(ref['cells'][n], again.cells[n]) for n in p_old.library) and torch.equal(ref['control'], again.control)
    assert set(again.protocol.calibration).isdisjoint(again.protocol.fit) and set(fit) | set(cal) == set(p_old.fit)
    pjson = (json.dumps(pd, indent=1, sort_keys=True) + '\n').encode()
    create_new(out_dir, 'protocol_v2.json', pjson)
    receipt = {'bundle_path': str(bpath), 'bundle_sha256': digest, 'protocol_sha256': p_new.fingerprint,
               'protocol_json_sha256': __import__('hashlib').sha256(pjson).hexdigest(),
               'fit': list(fit), 'calibration': list(cal), 'inner': list(p_new.inner), 'exploratory': list(p_new.exploratory),
               'calibration_split_sha256': split_hash(cal), 'derived_from': prov['derived_from'],
               'verified_equal': ['library', 'gene_ids', 'cell_ids', 'candidate/query partition', 'cells', 'control']}
    create_new(out_dir, 'bundle_receipt_v2.json', (json.dumps(receipt, indent=1) + '\n').encode())
    print('BUNDLE_V2', digest, p_new.fingerprint, split_hash(cal), flush=True)

if __name__ == '__main__':
    sys.exit(main())
