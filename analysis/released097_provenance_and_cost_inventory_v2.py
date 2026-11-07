"""Extend the unchanged settled-cost inventory to the optional follow-up phase."""
import hashlib, importlib.util, json, sys
from pathlib import Path
ROOT=Path('/ml2/alz');source=ROOT/'analysis/released097_provenance_and_cost_inventory.py'
def main():
    spec=importlib.util.spec_from_file_location('original_cost_inventory',source)
    inventory=importlib.util.module_from_spec(spec);spec.loader.exec_module(inventory)
    inventory.PHASES=('faithful097','localfaith097','followfaith097','confirmfaith097','benchmark097')
    inventory.main()
    output=Path(sys.argv[sys.argv.index('--output')+1]);q=json.loads(output.read_text())
    q['source_SHA256'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    q['phase_extension']='Includes optional followfaith097 actual results and charges; prior controls stay in their original phases and are never charged twice. The original inventory source is unchanged.'
    output.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
