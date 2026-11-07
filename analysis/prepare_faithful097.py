"""Validate the frozen four-fit restart and native trainer proof without admission."""
import json,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'faithful097_runtime'))
import preset_scope as scope
if __name__=='__main__':
 p=scope.validate_protocol()
 print(json.dumps(dict(status=p['status'],required_fits=p['maximum_new_fits'],native_parity_proof=str(scope.MODEL_PROOF),all_training_from_scratch=p['all_new_training_from_scratch'],GPU_allocation_created=False)))
