"""Explicit CLI for an owned cohort; no acquisition, scheduler or automatic expansion."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import time
import torch
from .gene_features import FeatureTable, file_sha256
from .gene_protocol import StudyProtocol, StudyData, validate_grant
from .gene_models import ModelConfig, build_models
from .gene_training import train_arm
from .gene_checkpoint import save_checkpoint, load_checkpoint
from .gene_evaluation import evaluate


def save_study_bundle(path, study):
    """Caller prepares data; all raw cells/IDs and outcome source provenance are retained."""
    torch.save({'schema_version':1,'protocol':asdict(study.protocol),'control':study.control,
                'cells':study.cells,'cell_ids':study.cell_ids,'partition':{'candidate_ids':study.candidate_ids,'query_ids':study.query_ids},
                'source_provenance':study.source_provenance},path)
    return file_sha256(path)


def load_study_bundle(path, expected_sha256, *, expected_protocol_sha256=None):
    if file_sha256(path)!=expected_sha256: raise ValueError('data bundle hash mismatch')
    blob=torch.load(path,map_location='cpu',weights_only=True)
    if blob.get('schema_version')!=1: raise ValueError('unsupported study bundle')
    protocol=StudyProtocol(**{k:tuple(v) if k in ('library','gene_ids','fit','inner','exploratory','final','calibration') else v for k,v in blob['protocol'].items()})
    if expected_protocol_sha256 is not None and protocol.fingerprint != expected_protocol_sha256:
        raise ValueError('bundle protocol differs from granted protocol')
    identity = {}
    if protocol.data_kind != 'synthetic':
        from .gene_data_contract import validate_real_bundle_identity
        identity = validate_real_bundle_identity(protocol,blob['control'],blob['cell_ids'],blob['source_provenance'])
    # Recreate the original partition by exact IDs, never silently draw a new split.
    partition_seed = blob['source_provenance'].get('sampling', {}).get('partition_seed', 43)
    study=StudyData.partition(protocol,blob['control'],blob['cells'],blob['cell_ids'],seed=partition_seed,source_provenance=blob['source_provenance'])
    part=blob['partition']
    if set(part['candidate_ids']) != set(protocol.library) or set(part['query_ids']) != set(protocol.library):
        raise ValueError('stored partitions must cover the exact common library')
    for name in protocol.library:
        for kind in ('candidate_ids','query_ids'):
            ids = part[kind][name]
            if not ids or len(set(ids)) != len(ids): raise ValueError('stored partition contains duplicate or empty cell IDs')
        if set(part['candidate_ids'][name]) & set(part['query_ids'][name]): raise ValueError('stored oracle partitions overlap')
        if set(part['candidate_ids'][name])|set(part['query_ids'][name]) != set(study.cell_ids[name]): raise ValueError('stored partitions do not cover source cells')
        lookup={cell:i for i,cell in enumerate(study.cell_ids[name])}
        ci=[lookup[cell] for cell in part['candidate_ids'][name]];qi=[lookup[cell] for cell in part['query_ids'][name]]
        study.candidate_cells[name]=study.cells[name][ci].clone();study.candidate_ids[name]=tuple(part['candidate_ids'][name])
        study.query_profiles[name]=study.cells[name][qi].mean(0);study.query_ids[name]=tuple(part['query_ids'][name])
    study.source_provenance=dict(study.source_provenance,identity_contract=identity,bundle_path=str(Path(path).resolve()),bundle_sha256=expected_sha256)
    if file_sha256(path)!=expected_sha256:raise ValueError('study bundle changed during reading')
    return study


def apply_cuda_allocator_cap(device, cap_gib, grant):
    """Optional grant-bound cap; owner still accounts for non-Torch overhead."""
    declared = grant.get('cuda_allocator_cap_gib')
    if cap_gib is None and declared is None: return
    if device != 'cuda:0' or cap_gib is None or cap_gib <= 0 or cap_gib != declared:
        raise PermissionError('CUDA allocator cap must match the owned grant and cuda:0')
    total = torch.cuda.get_device_properties(0).total_memory
    requested = cap_gib * 1024**3
    if requested > total: raise ValueError('allocator cap exceeds visible device capacity')
    torch.cuda.set_per_process_memory_fraction(float(requested / total), 0)


from .paths import check_new_target, scoped_new_dir, create_new
from . import parent_watch


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['train','evaluate'])
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--bundle-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',default='B0')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--z-dim',type=int,default=16);p.add_argument('--particles',type=int,default=256)
    p.add_argument('--lr',type=float,default=.0002);p.add_argument('--max-wall-seconds',type=int,required=True)
    p.add_argument('--cuda-allocator-cap-gib',type=float)
    p.add_argument('--device',default='cpu');p.add_argument('--grant',type=Path,required=True);p.add_argument('--grant-sha256',required=True)
    p.add_argument('--feature-manifest',type=Path);p.add_argument('--feature-manifest-sha256');p.add_argument('--tau-source')
    p.add_argument('--string-row-mapping',type=Path);p.add_argument('--string-row-mapping-sha256')
    p.add_argument('--string-license-receipt',type=Path);p.add_argument('--string-license-receipt-sha256')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--checkpoint-sha256')
    p.add_argument('--frozen-source',type=Path);p.add_argument('--frozen-source-sha256');p.add_argument('--frozen-generator-arm',default='G')
    p.add_argument('--profile-mode',default='generated',choices=['generated','independent_real_oracle'])
    a=p.parse_args(argv)
    parent_watch.start()   # v4.4: no-op unless launched by the supervisor (ALZ_SUPERVISOR_IDENT)
    deadline = time.monotonic() + a.max_wall_seconds
    # v4.2 blocker 3: containment is checked before ANY work or write (training included).
    if a.output.exists():raise FileExistsError('new output required; do not overwrite receipts')
    a.output = check_new_target(a.output)
    if file_sha256(a.grant)!=a.grant_sha256:raise PermissionError('cohort grant hash changed')
    grant=json.loads(a.grant.read_text())
    if grant.get('bundle_sha256') is not None and grant['bundle_sha256'] != a.bundle_sha256:
        raise PermissionError('bundle differs from approved cohort')
    study=load_study_bundle(a.bundle,a.bundle_sha256,expected_protocol_sha256=grant.get('protocol_sha256'))
    config=ModelConfig(len(study.control),len(study.protocol.library),hidden=a.hidden,z_dim=a.z_dim,particles=a.particles,
                       seed=a.seed,steps=a.steps,batch_size=a.batch_size,lr=a.lr,arm=a.arm,frozen_generator_arm=a.frozen_generator_arm)
    validate_grant(grant,study,config,a.device,a.phase,a.max_wall_seconds)
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0: raise TimeoutError('total cohort wall budget exhausted')
        return seconds
    table=None
    if a.feature_manifest:
        if grant.get('feature_manifest_sha256')!=a.feature_manifest_sha256:raise PermissionError('feature packet differs from approved cohort')
        kwargs={'names':study.protocol.library,'gene_ids':study.protocol.gene_ids,'fit_names':study.protocol.fit}
        if a.tau_source:
            for key in ('string_row_mapping_sha256','string_license_receipt_sha256'):
                value=getattr(a,key)
                if not value or grant.get(key) != value: raise PermissionError('STRING supplement differs from approved cohort')
            table=FeatureTable.load_tau_packet(a.feature_manifest,a.feature_manifest_sha256,source_key=a.tau_source,
                mapping_path=a.string_row_mapping,expected_mapping_sha256=a.string_row_mapping_sha256,
                license_path=a.string_license_receipt,expected_license_sha256=a.string_license_receipt_sha256,**kwargs)
        else:table=FeatureTable.load(a.feature_manifest,a.feature_manifest_sha256,**kwargs)
    apply_cuda_allocator_cap(a.device,a.cuda_allocator_cap_gib,grant)
    invocation={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    if a.phase=='train':
        validate_grant(grant,study,config,a.device,'evaluate',a.max_wall_seconds)
        models=build_models(config,table,data_kind=study.protocol.data_kind)
        models.validate_study(study)
        frozen=load_checkpoint(a.frozen_source,a.frozen_source_sha256) if a.frozen_source else None
        validate_grant(grant,study,config,a.device,'train',a.max_wall_seconds)
        models.to(a.device)
        if frozen is not None: frozen.to(a.device)
        receipt=train_arm(models,study,max_wall_seconds=remaining(),grant=grant,frozen_source=frozen)
        # Run root keeps model/training records separate; evaluation itself has exactly four groups.
        out=scoped_new_dir(a.output)  # re-verified at creation time
        digest=save_checkpoint(out/'checkpoint.pt',models,study.protocol,table,training_receipt=receipt)
        create_new(out,'training.json',(json.dumps(receipt,indent=2)+'\n').encode())
        models=load_checkpoint(a.output/'checkpoint.pt',digest)
        validate_grant(grant,study,models.config,a.device,'evaluate',a.max_wall_seconds)
        models.to(a.device)
        evaluate(models,study,output=a.output/'evaluation',profile_mode=a.profile_mode,invocation=invocation,max_wall_seconds=remaining())
    else:
        if not a.checkpoint or not a.checkpoint_sha256:raise ValueError('hash-bound checkpoint required')
        models=load_checkpoint(a.checkpoint,a.checkpoint_sha256)
        if models.config.arm!=a.arm or models.config.seed!=a.seed:raise ValueError('checkpoint arm/seed differs from invocation')
        validate_grant(grant,study,models.config,a.device,'evaluate',a.max_wall_seconds)
        models.validate_study(study)
        models.to(a.device)
        evaluate(models,study,output=a.output,profile_mode=a.profile_mode,invocation=invocation,max_wall_seconds=remaining())


if __name__=='__main__':main()
