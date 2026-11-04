"""Qualify the fixed fitting-only identity partition; no count arrays opened."""
import datetime
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import partition as p
from contracts import fold

ROOT = Path('/ml2/alz')


class PartitionChecks(unittest.TestCase):
    def fixture(self):
        donors = np.array(['a']*13+['b']*9)
        # Cell barcodes may coincide across donors; pairs must remain distinct.
        names = np.array([f'cell{i}' for i in range(13)]+[f'cell{i}' for i in range(9)])
        return donors,names,np.array(['b','a'])

    def test_disjoint_exhaustive_donor_local_rank_counts(self):
        d,c,e=self.fixture();q=p.partition(d,c,e)
        self.assertEqual(q.donors,('b','a'))
        self.assertEqual([len(x) for x in q.protected_by_donor],[1,2])
        for donor,fit,guard in zip(q.donors,q.fitting_by_donor,q.protected_by_donor):
            self.assertEqual(set(d[fit]),{donor});self.assertEqual(set(d[guard]),{donor})
            self.assertFalse(set(fit)&set(guard))
        np.testing.assert_array_equal(np.sort(np.r_[q.fitting,q.protected]),np.arange(len(d)))

    def test_membership_invariant_to_input_row_permutation(self):
        d,c,e=self.fixture();q=p.partition(d,c,e)
        order=np.random.default_rng(20261104).permutation(len(d));z=p.partition(d[order],c[order],e)
        self.assertEqual(q.membership_SHA256,z.membership_SHA256)
        self.assertEqual(set(zip(d[q.fitting],c[q.fitting])),set(zip(d[order][z.fitting],c[order][z.fitting])))

    def test_fixed_hash_rank_is_exact_and_arrays_readonly(self):
        d,c,e=self.fixture();q=p.partition(d,c,e)
        for donor,guard in zip(q.donors,q.protected_by_donor):
            expected=sorted(np.flatnonzero(d==donor).tolist(),key=lambda i:(hashlib.sha256(p.key(donor,str(c[i]))).digest(),str(c[i])))
            self.assertEqual(guard.tolist(),expected[:max(1,len(expected)//5)])
        with self.assertRaises(ValueError):q.fitting[0]=0

    def test_external_missing_duplicate_or_small_donor_refused(self):
        d,c,e=self.fixture()
        for observed,names,expected in [(d,c,np.array(['a','b','external'])),
                                        (d,c,np.array(['a'])),(d,c,np.array(['a','b','a'])),
                                        (np.array(['a','b','b']),np.array(['x','y','z']),np.array(['a','b']))]:
            with self.assertRaises(ValueError):p.partition(observed,names,expected)
        c[1]=c[0]
        with self.assertRaisesRegex(ValueError,'Duplicate donor/cell'):p.partition(d,c,e)

    def test_empty_numeric_or_misaligned_identifiers_refused(self):
        d,c,e=self.fixture()
        for observed,names,expected in [(d[:-1],c,e),(np.arange(len(d)),c,e),
                                        (d,c,np.array([])),(d.reshape(2,-1),c,e)]:
            with self.assertRaises(ValueError):p.partition(observed,names,expected)
        c[0]=''
        with self.assertRaisesRegex(ValueError,'Empty'):p.partition(d,c,e)


def main():
    assert Path(p.__file__).resolve()==ROOT/'routed_model/partition.py'
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PartitionChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun==5
    auth_path=ROOT/'artifacts/allocations/CAL_GPU0_V1_halloween_native_v1_f0_pg_s0/authorization.json'
    auth=json.loads(auth_path.read_text());cache=Path(auth['cache_dir']);manifest_path=cache/'manifest.json'
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert sha(manifest_path)==auth['files_SHA256'][str(manifest_path)]
    manifest=json.loads(manifest_path.read_text());f=fold(0)
    assert manifest['train_donors']==f['optimization_donors'] and manifest['fold']==0
    meta=cache/'train/metadata.npz';assert sha(meta)==manifest['files_SHA256']['train/metadata.npz']
    with np.load(meta,allow_pickle=False) as a:
        assert a['donors'].tolist()==f['optimization_donors']
        q=p.partition(a['donor'],a['cell'],a['donors'])
        assert len(q.donors)==33 and len(q.fitting)+len(q.protected)==manifest['train_cells']
        assert not set(q.donors)&set(f['internal_calibration_donors']+f['evaluation_donors'])
        order=np.random.default_rng(20261104221).permutation(len(a['cell']))
        again=p.partition(a['donor'][order],a['cell'][order],a['donors'])
        assert again.membership_SHA256==q.membership_SHA256
    assert 'torch' not in sys.modules
    output=ROOT/'analysis/routed_fitting_partition_CPU_qualification.json';assert not output.exists()
    proof={'status':'ACTUAL_FITTING_ONLY_FIXED_CELL_PARTITION_AND_CPU_REFUSALS_PASSED',
           'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':5,
           'actual_fold0_metadata_partition':q.summary(),
           'real_fitting_metadata_accessed':True,'fitting_count_arrays_opened':False,
           'calibration_or_evaluation_count_arrays_opened':False,
           'partition_uses_only_donor_cell_IDs':True,'metadata_row_permutation_preserves_membership':True,
           'Torch_or_GPU_imported':False,'optimizer_or_model_created':False,'quality_measured':False,
           'source_SHA256':{str(path):sha(path) for path in [Path(__file__).resolve(),Path(p.__file__).resolve()]},
           'input_SHA256':{str(path):sha(path) for path in [manifest_path,meta]},
           'scope':'Actual33fittingdonor metadata-only partition and five CPU boundary tests. Fixed20percent per-donor hash-rank protection with floor and nonempty pools. Sampler/gradient isolation and fit-only statistics still require loader qualification. Common feature transform predates partition; no independent preprocessing/biological confirmation claim.'}
    output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('actual_fold0_metadata_partition','source_SHA256','input_SHA256')},indent=2))
    print('Actual cells: fitting',len(q.fitting),'protected',len(q.protected))


if __name__=='__main__':main()
