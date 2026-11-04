"""Actual native CPU checkpoint and fingerprint checks, no GPU context."""
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_routed_model import fixture
import routed_pg_adapter as a
from checkpoint_audit import audit_checkpoint,selected_states
from state_fingerprint import fingerprint


class CheckpointAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cache=Path(cls.temp.name)
        (cache/'manifest.json').write_text('{}');cls.ctx=fixture(cache,genes=2000)
        cls.q=torch.load('/ml2/alz/artifacts/diagnostics/routed_final_adapter_CPU_fullwidth_probe20_v1/before19.pt',map_location='cpu',weights_only=False)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_actual_fullwidth_native_checkpoint_and_serving_weights(self):
        result=audit_checkpoint(self.q,self.ctx,steps=19,expected_device='cpu')
        self.assertEqual(result['model_floating_values'],371617)
        runtime=a.restore(self.ctx,self.q)
        served=runtime.policy.served_model()
        self.assertEqual(result['served_models_fingerprint'],fingerprint({k:v.state_dict() for k,v in served.models.items()}))
        self.assertEqual(result['served_table_fingerprint'],fingerprint(served.table))
        self.assertEqual(result['policy_fingerprint'],fingerprint(runtime.policy.state_dict()))

    def test_fingerprint_preserves_float_bits_tensor_shapes_types_and_numpy(self):
        nan1=struct.unpack('<d',bytes.fromhex('010000000000f87f'))[0]
        nan2=struct.unpack('<d',bytes.fromhex('020000000000f87f'))[0]
        self.assertEqual(fingerprint({'a':nan1,'b':torch.tensor([1.])}),fingerprint({'b':torch.tensor([1.]),'a':nan1}))
        for left,right in [(nan1,nan2),(0.,-0.),(torch.tensor([1.]),torch.tensor(1.)),([1],(1,)),(np.array([1],dtype=np.int32),np.array([1],dtype=np.int64))]:
            self.assertNotEqual(fingerprint(left),fingerprint(right))
        with self.assertRaises(TypeError):fingerprint({1,2})

    def test_wrong_device_clock_config_and_partition_refused(self):
        for key,value in [('completed_updates',20),('routed_config',{}),('partition',{}),('training_streams',{})]:
            q=dict(self.q);q[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):audit_checkpoint(q,self.ctx,steps=19,expected_device='cpu')
        with self.assertRaises(AssertionError):audit_checkpoint(self.q,self.ctx,steps=19,expected_device='cuda:0')

    def test_averaged_selection_uses_coupled_native_parameters_and_mass(self):
        # Private synthetic counterfactual validates only the audited native
        # selection law; no real checkpoint or selector is changed.
        runtime=a.build(self.ctx,0);state=runtime.policy.state_dict()
        state['served_source']='averaged';found=False
        for row,roles in zip(state['lr_settle'],state['roles']):
            for value,role in zip(row,roles):
                if value is not None and role=='table':value['last_decisive']=-1;found=True
        self.assertTrue(found)
        state['averages']['prior']['z']+=.125
        state['averaged_table']=state['averages']['prior']['z'].clone()
        state['averages']['router']['log_mass']-=.25
        models,table=selected_states(state,runtime)
        self.assertEqual(fingerprint(table),fingerprint(state['averaged_table']))
        self.assertEqual(fingerprint(models['router']),fingerprint(state['averages']['router']))
        self.assertNotEqual(fingerprint(models['prior']),fingerprint(state['models']['prior']))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main(verbosity=2)
