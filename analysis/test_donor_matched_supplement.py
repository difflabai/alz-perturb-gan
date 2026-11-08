"""Analytic estimator and source-law checks on synthetic data only."""
import copy
import pathlib
import sys
import tempfile
import unittest
ROOT=pathlib.Path('/ml2/alz');PROJECT=pathlib.Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(ROOT/'analysis'),str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
import score_benchmark097_donor_matched_supplement as m


class DonorMatched(unittest.TestCase):
    def test_weighted_distance_has_known_nonuniform_value(self):
        self.assertAlmostEqual(m.weighted_ED([[0.],[2.]],[[1.],[3.]],[.25,.75],[.5,.5]),.75,places=12)

    def test_equal_donor_weights_correct_unequal_cell_counts(self):
        x=np.array([[0.],[0.],[0.],[2.]]);y=np.array([[0.],[2.],[2.],[2.]])
        self.assertAlmostEqual(m.weighted_ED(x,y,np.full(4,.25),np.full(4,.25)),1.,places=12)
        self.assertAlmostEqual(m.weighted_ED(x,y,[1/6,1/6,1/6,.5],[.5,1/6,1/6,1/6]),0.,places=12)

    def test_uniform_weights_match_original_V_statistic(self):
        from audit_released097_completed_studies import independent_cell_ED
        rng=np.random.default_rng(7);a=rng.normal(size=(7,3));b=rng.normal(size=(9,3))
        self.assertAlmostEqual(m.weighted_ED(a,b,np.full(7,1/7),np.full(9,1/9)),independent_cell_ED(a,b),places=12)

    def test_invalid_weights_or_nonfinite_values_refuse(self):
        for a in [[-.1,1.1],[.2,.2],[np.nan,.5],[1.]]:
            with self.assertRaises(AssertionError):m.weighted_ED([[0.],[1.]],[[0.],[1.]],a,[.5,.5])
        with self.assertRaises(AssertionError):m.weighted_ED([[np.nan]],[[1.]],[1.],[1.])

    def fixture(self):
        donors=[f'D{i}' for i in range(7)];labels=np.asarray(sum(([d]*(2 if i==0 else 10) for i,d in enumerate(donors)),[]));stages={d:i%4 for i,d in enumerate(donors)}
        return donors,labels,stages

    def test_reference_keeps_every_donor_with_exact_mass_and_records_replacement(self):
        donors,labels,stages=self.fixture();r=m.real_reference(labels,donors,stages,0,0,size=30)
        self.assertEqual(set(r['labels']),set(donors));self.assertTrue(r['replacement'][0]);self.assertEqual(len(r['indices']),30)
        self.assertEqual(int(r['allocation'].max()-r['allocation'].min()),1)
        for d in donors:self.assertAlmostEqual(r['weights'][r['labels']==d].sum(),1/7,places=12)

    def test_reference_is_shared_and_does_not_consume_global_rng(self):
        donors,labels,stages=self.fixture();np.random.seed(47);before=np.random.get_state()
        a=m.real_reference(labels,donors,stages,0,1,size=30);b=m.real_reference(labels,donors,stages,0,1,size=30)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        after=np.random.get_state();np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])

    def test_missing_or_duplicate_donor_and_too_small_reference_refuse(self):
        donors,labels,stages=self.fixture()
        for ds,size in [(donors+['missing'],30),(donors+[donors[0]],30),(donors,6)]:
            with self.assertRaises(AssertionError):m.real_reference(labels,ds,stages,0,0,size=size)

    def test_excluded_donor_cannot_appear_in_reference(self):
        donors,labels,stages=self.fixture();r=m.real_reference(labels,donors[1:],stages,1,1,size=30)
        self.assertNotIn(donors[0],r['labels']);self.assertEqual(set(r['labels']),set(donors[1:]))

    def test_generated_plan_matches_stage_mass_with_common_rows_and_no_duplicates(self):
        donors,labels,stages=self.fixture();r=m.real_reference(labels,donors,stages,0,2,size=30)
        a=m.generated_row_plan(r,0,2,3,1,pool_size=40);b=m.generated_row_plan(r,0,2,3,1,pool_size=40)
        np.testing.assert_array_equal(a,b)
        for s in range(4):self.assertEqual(len(np.unique(a[r['stages']==s])),int((r['stages']==s).sum()))
        def loader(s,idx):return np.full((len(idx),50),s,dtype=np.float32)
        x=m.select_generated(r,a,loader);np.testing.assert_array_equal(x[:,0],r['stages'])

    def test_insufficient_existing_model_pool_refuses_resampling(self):
        donors,labels,stages=self.fixture();r=m.real_reference(labels,donors,stages,0,0,size=30)
        with self.assertRaises(AssertionError):m.generated_row_plan(r,0,0,0,0,pool_size=1)

    def test_incorrect_generated_width_cannot_be_broadcast_into_PCs(self):
        donors,labels,stages=self.fixture();r=m.real_reference(labels,donors,stages,0,0,size=30)
        rows=m.generated_row_plan(r,0,0,0,0,pool_size=40)
        with self.assertRaises(AssertionError):m.select_generated(r,rows,lambda s,idx:np.ones((len(idx),1),dtype=np.float32))

    def test_nonfinite_complete_point_set_cannot_be_reported(self):
        points=[dict(method=method,fold=f,seed=s,replicate=r,donor_matched_cell_energy_distance=.25) for method in m.METHODS for f in range(3) for s in range(5) for r in (0,1)]
        points[0]['donor_matched_cell_energy_distance']=float('nan')
        with self.assertRaises(AssertionError):m.summarize(points)

    def test_full_matrix_of_points_is_mandatory_for_summary(self):
        points=[dict(method=method,fold=f,seed=s,replicate=r,donor_matched_cell_energy_distance=.25) for method in m.METHODS for f in range(3) for s in range(5) for r in (0,1)]
        self.assertEqual(len(m.summarize(points)['pg']['fold_seed_rows']),15)
        with self.assertRaises(AssertionError):m.summarize(points[:-1])
        points[-1]=copy.deepcopy(points[0])
        with self.assertRaises(AssertionError):m.summarize(points)

    def test_partial_quality_files_stay_opaque_and_run_refuses_before_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);paths=m.required_paths(root)
            for p in paths[:-1]:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'OPAQUE_QUALITY')
            ready=m.readiness(root);self.assertEqual(ready['missing_paths'],[str(paths[-1])]);self.assertFalse(ready['expression_or_quality_values_read'])
            with self.assertRaisesRegex(AssertionError,'All45 models and both original reports required'):m.run(root)
            self.assertFalse((root/'artifacts/benchmark097-donor-matched-cell-distance-20261108').exists())


if __name__=='__main__':unittest.main()
