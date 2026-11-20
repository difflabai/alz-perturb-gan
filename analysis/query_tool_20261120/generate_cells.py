"""Generation Monte Carlo replicates of the selected native EMPF model with cell-level subtype sums.

One replicate = the default contract (37 development-donor conditions, M=32 generated
donors x K=256 cells, readout cellshape@0.5:biascal:stagehalf, six late states) with the
generation streams SeedSequence([20261114, 6, fold, development donor index, REPLICATE]).
Replicate 0 uses exactly the streams of the scored default archive. The archive arrays
(pb_counts, mean_log, sq_log, nonzero, pcs, lib) are written in the frozen scorer's format, and it records, per target donor, the sum of per-cell
log1p(panel CP10k) and the cell count in each of six subtypes assigned by the fold's frozen
TRAIN-only type classifier (column 0 = all cells, column k+1 = type k), the analysis
convention of the frozen 0.9.7 stage-replicate summaries.

No weights change, no training update is made, and nothing is written in the fit, the
cache, the derived host or any pinned source. Outputs go to --out (outside git).
"""
from __future__ import annotations

import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_host  # noqa: E402

ROOT = native_host.ROOT
MODE = 'cellshape@0.5:biascal:stagehalf'
SELECTION = ROOT / 'analysis/default_empf_20261117/selected_model.json'
FIT = ROOT / 'runs/empf_comparison_20261117/fits/native/pbulk/hvg2000/fold0/seed0'
CACHE = ROOT / 'runs/improve_20261109/caches/hvg2000/fold0'
TYPES = ['Micro-PVM_1', 'Micro-PVM_2', 'Micro-PVM_2_1-SEAAD', 'Micro-PVM_2_3-SEAAD',
         'Micro-PVM_3-SEAAD', 'Micro-PVM_4-SEAAD']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_selected(fit):
    selected = json.loads(SELECTION.read_text())
    for name, digest in selected['weights_sha256'].items():
        if sha(fit / name) != digest:
            raise ValueError(f'selected weight differs: {name}')
    for name, digest in selected['metadata_sha256'].items():
        if sha(fit / name) != digest:
            raise ValueError(f'selected metadata differs: {name}')
    return selected['weights_sha256']


def classifier(torch, path, device):
    net = torch.nn.Sequential(torch.nn.Linear(50, 64), torch.nn.ReLU(), torch.nn.Linear(64, len(TYPES)))
    state = torch.load(path, map_location='cpu', weights_only=True)
    net.load_state_dict({k.removeprefix('net.'): v for k, v in state.items()}, strict=True)
    return net.to(device).eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replicate', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--gpu-mem-fraction', type=float, default=.06)
    p.add_argument('--max-targets', type=int, default=None, help='validation runs only')
    p.add_argument('--check-archive', type=Path, default=None,
                   help='require bitwise-equal mean_log/nonzero/lib to this archive (replicate 0, same device)')
    a = p.parse_args(argv)
    if a.replicate < 0 or not 0 < a.gpu_mem_fraction <= .08:
        p.error('invalid replicate or GPU fraction')
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(f'{out} exists')
    if a.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    weights = check_selected(FIT)
    cache_manifest = sha(CACHE / 'manifest.json')
    private = out.parent / f'.host_{out.stem}'
    T, G, provenance = native_host.activate(private)
    torch, np = T.torch, T.np
    torch.set_num_threads(3)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def streams(fold, index):
        children = np.random.SeedSequence([20261114, 6, fold, index, a.replicate]).spawn(5)
        return {key: int(child.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 16)
                for key, child in zip(G.GEN_STREAMS, children)}
    G.generation_streams = streams

    config = json.loads((FIT / 'config.json').read_text())
    if config['cache_manifest_SHA256'] != cache_manifest:
        raise ValueError('cache differs from the fit')
    ctx = T.Context(CACHE, a.device, tuple(config['flags']), allow_nonprotocol_split=not config['protocol_split'])
    targets = G.targets(ctx)
    states = G.available_states(FIT, config)
    plans, spec, value = G.mode_plans(ctx, MODE, targets, states, updates=config.get('updates'))
    used = [s for s in states if any(s in names for _, names in plans)]
    started = time.monotonic()
    models = G.load_models(ctx, FIT, config, used)
    with torch.no_grad():
        for name, (Gm, Pm) in models.items():
            spec.prepare(ctx, value, name, Gm, Pm)
    space = np.load(CACHE / 'space.npz', allow_pickle=False)
    pc_sd = torch.as_tensor(space['pc_sd'], device=ctx.device)
    clf = classifier(torch, CACHE / 'type_classifier.pt', ctx.device)
    if a.max_targets is not None:
        targets, plans = targets[:a.max_targets], plans[:a.max_targets]

    M, K, CH = G.M, G.K, G.CHUNK
    n_t, n_g = len(targets), ctx.n_genes
    arr = {'mean_log': np.zeros((n_t, M, n_g), np.float32), 'sq_log': np.zeros((n_t, M, n_g), np.float16),
           'pb_counts': np.zeros((n_t, M, n_g), np.float32), 'pcs': np.zeros((n_t, M, G.PC_CELLS, 50), np.float16),
           'nonzero': np.zeros((n_t, M, n_g), np.float32), 'lib': np.zeros((n_t, M), np.float32),
           'subtype_sums': np.zeros((n_t, 7, n_g), np.float64), 'subtype_n': np.zeros((n_t, 7), np.int64),
           'subtype_conf': np.zeros((n_t,), np.float64)}
    dev_order = T.development_order()
    for Gm, Pm in models.values():
        Gm.eval(); Pm.eval()
    with torch.no_grad():
        for ti, (donor, _, _, pool) in enumerate(targets):
            s = G.generation_streams(ctx.fold, dev_order[donor])
            torch.manual_seed(s['global'])
            lat = torch.Generator(device=ctx.device).manual_seed(s['latent'])
            cell = torch.Generator(device=ctx.device).manual_seed(s['cell'])
            count = torch.Generator(device=ctx.device).manual_seed(s['count'])
            libs = np.random.default_rng(s['library']).choice(pool, size=(M, K), replace=True).astype(np.float32)
            cond, names = plans[ti]
            ct = torch.as_tensor(np.ascontiguousarray(cond), device=ctx.device)
            runs, start_run = [], 0
            for m in range(1, M + 1):
                if m == M or names[m] != names[start_run]:
                    runs.append((start_run, m, names[start_run])); start_run = m
            conf = []
            for lo, hi, name in runs:
                Gm, Pm = models[name]
                for start in range(lo, hi, CH):
                    n = min(CH, hi - start)
                    z, _ = Pm.sample(n, generator=lat)
                    lib = torch.as_tensor(libs[start:start + n], device=ctx.device)
                    rate = Gm.rates(z, ct[start:start + n], K, lib, cell)
                    counts = torch.poisson(rate, generator=count)
                    if not bool(torch.isfinite(counts).all()):
                        raise FloatingPointError('generated count contract failed')
                    tot = counts.sum(-1)
                    logcp = torch.log1p(counts * (1e4 / tot.clamp_min(1))[..., None])
                    sl = slice(start, start + n)
                    arr['mean_log'][ti, sl] = logcp.mean(1).cpu().numpy()
                    arr['sq_log'][ti, sl] = logcp.square().mean(1).cpu().numpy().astype(np.float16)
                    arr['pb_counts'][ti, sl] = counts.sum(1).cpu().numpy()
                    arr['pcs'][ti, sl] = (G.standardize_counts(counts[:, :G.PC_CELLS], ctx.mean, ctx.sd) @ ctx.W
                                          ).cpu().numpy().astype(np.float16)
                    arr['nonzero'][ti, sl] = (counts > 0).float().mean(1).cpu().numpy()
                    arr['lib'][ti, sl] = tot.mean(1).cpu().numpy()
                    flat = logcp.reshape(-1, n_g)
                    pcs = ((flat - ctx.mean) / ctx.sd) @ ctx.W / pc_sd
                    prob = torch.softmax(clf(pcs.float()), -1)
                    label = prob.argmax(-1)
                    conf.append(prob.max(-1).values.double().cpu().numpy())
                    onehot = torch.nn.functional.one_hot(label, len(TYPES)).double()
                    arr['subtype_sums'][ti, 1:] += (onehot.T @ flat.double()).cpu().numpy()
                    arr['subtype_n'][ti, 1:] += onehot.sum(0).long().cpu().numpy()
                    arr['subtype_sums'][ti, 0] += flat.double().sum(0).cpu().numpy()
                    arr['subtype_n'][ti, 0] += flat.shape[0]
            arr['subtype_conf'][ti] = float(np.concatenate(conf).mean())
    raws = np.stack([t[2] for t in targets])
    arr.update(targets=np.array([t[0] for t in targets]), is_heldout=np.array([t[1] for t in targets], dtype=bool),
               stage=G.stage_of(raws).astype(np.int64), genes=np.asarray(ctx.genes).astype(str),
               state_index=np.array([[used.index(s) for s in names] for _, names in plans], dtype=np.int8),
               state_names=np.array(used), replicate=np.array(a.replicate), mode=np.array(MODE))
    check = None
    if a.check_archive is not None:
        with np.load(a.check_archive, allow_pickle=False) as ref:
            check = {k: bool(np.array_equal(arr[k], ref[k][:n_t]))
                     for k in ('mean_log', 'nonzero', 'lib', 'sq_log', 'pb_counts', 'pcs')}
            check['archive_sha256'] = sha(a.check_archive)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name('.' + out.stem + '.partial.npz')
    np.savez_compressed(tmp, **arr)
    tmp.replace(out)
    if check_selected(FIT) != weights or sha(CACHE / 'manifest.json') != cache_manifest:
        raise ValueError('fit or cache changed during generation')
    receipt = dict(replicate=a.replicate, mode=MODE, device=a.device, targets=n_t, M=M, K=K,
                   generation_streams=[20261114, 6, ctx.fold, 'development donor index', a.replicate],
                   states_used=used, seconds=time.monotonic() - started, output_sha256=sha(out),
                   fit=str(FIT), weights_sha256=weights, cache_manifest_sha256=cache_manifest,
                   classifier_sha256=sha(CACHE / 'type_classifier.pt'), host_snapshot=provenance,
                   source_sha256=sha(__file__), loader_sha256=sha(native_host.__file__),
                   archive_check=check, training_updates=0, weights_changed=False,
                   subtype_assignment='frozen fold0 TRAIN-only classifier on (z@W)/pc_sd; predictions, not observed labels')
    out.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    print(json.dumps({k: receipt[k] for k in ('replicate', 'device', 'targets', 'seconds', 'archive_check')}), flush=True)
    if check is not None and not all(v for k, v in check.items() if k != 'archive_sha256'):
        raise SystemExit('archive check failed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
