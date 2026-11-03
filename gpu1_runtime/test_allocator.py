"""No Torch import: exact native allocator API arguments and early refusal."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import worker_gpu0 as worker
class Allocator(unittest.TestCase):
 def test_finite_allocator_limit_and_device_are_explicit(self):
  cuda=SimpleNamespace(get_device_properties=Mock(return_value=SimpleNamespace(total_memory=48*1024**3)),set_per_process_memory_fraction=Mock())
  worker.configure_allocator(SimpleNamespace(cuda=cuda),{'torch_allocator_cap_bytes':2*1024**3})
  cuda.get_device_properties.assert_called_once_with(0);cuda.set_per_process_memory_fraction.assert_called_once_with(1/24,device=0)
 def test_missing_or_larger_cap_refuses_before_cuda_query(self):
  for value in [None,3*1024**3,0,float('nan')]:
   cuda=SimpleNamespace(get_device_properties=Mock(),set_per_process_memory_fraction=Mock())
   with self.assertRaises(worker.b.Refusal):worker.configure_allocator(SimpleNamespace(cuda=cuda),{'torch_allocator_cap_bytes':value})
   cuda.get_device_properties.assert_not_called();cuda.set_per_process_memory_fraction.assert_not_called()
if __name__=='__main__':unittest.main()
