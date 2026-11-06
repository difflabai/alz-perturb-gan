"""One fixed benchmark cell: train/reuse, native restore, evaluate and resample."""
import argparse,json,os,socket,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b
import preset_scope as scope

def validate_pg(q,auth):
    v=auth['cell']
    b.require(q['schema']==1 and q['method']=='pg' and q['native_recipe']=='bispf_optimized','Native PG checkpoint required')
    b.require(q['fold']==v['fold'] and q['seed']==v['seed'],'Checkpoint fold/seed differs')
    b.require(q['status']=='completed' and q['completed_updates']==q['fixed_endpoint']==q['policy']['completed_steps']==20000,'Fixed20k final checkpoint required')
    b.require(q['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json'),'Checkpoint optimization cache differs')
    b.require(q['particlegan_version']=='0.9.6' and q['penalty_options']=={'input_domain':'continuous'},'Native release/domain differs')
    b.require(q['released_hyperparameters']==scope.HYPER and q['variant_config']==scope.CFG,'Frozen configuration differs')

