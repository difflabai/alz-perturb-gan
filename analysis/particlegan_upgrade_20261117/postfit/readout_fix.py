"""Explicitly register the existing fixed readout; preserve all326 source pins."""
import argparse
import importlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import common as C
import host


def sources():
    return {**C.check_sources(),str(C.PIN):host.sha(C.PIN),
        **{str(p):host.sha(p) for p in sorted(HERE.glob('*.py'))},
        str(HERE/'AMENDMENT.txt'):host.sha(HERE/'AMENDMENT.txt')}


def check():
    assert sources()==json.loads((HERE/'source_manifest.json').read_text())


def supplement(lane):
    assert C.valid_score(lane)
    path=host.RUNS/f'results/{lane}.readout_registration.json'
    score=host.RUNS/f'results/{lane}.json'
    generation=host.RUNS/f'generated/{lane}.receipt.json'
    expected={'status':'completed_with_fixed_readout_registration','lane':lane,
        'training_source_manifest_sha256':host.sha(C.PIN),
        'additive_execution_manifest_sha256':host.sha(HERE/'source_manifest.json'),
        'score_sha256':host.sha(score),'generation_receipt_sha256':host.sha(generation),
        'registered_mode':'cellshape@0.5:biascal:stagehalf',
        'registered_source_sha256':{str(host.OLD/n):host.sha(host.OLD/n)
                                   for n in ('mode_cellshape.py','mode_zcal.py')},
        'scope':'registration only; original native training, fixed readout and scoring unchanged'}
    if path.exists():assert json.loads(path.read_text())==expected
    else:C.write(path,expected)
    return expected


def run(lane):
    check()
    import worker
    original=host.activate
    def registered_host(selected_lane):
        result=original(selected_lane)
        module=importlib.import_module('mode_cellshape')
        assert Path(module.__file__).resolve()==host.OLD/'mode_cellshape.py'
        result[0].arms.parse_mode(C.MODE)
        return result
    host.activate=registered_host
    try:worker.generate(lane)
    finally:host.activate=original
    check();supplement(lane)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane',choices=('compat','rhbi'),required=True)
    run(p.parse_args().lane)
