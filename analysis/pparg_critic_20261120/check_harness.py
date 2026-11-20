"""Verify the study harness reproduces the retained native EMPF fit: control seed0 at U updates
must equal the retained fold0/seed0 fit's snap_U.pt generator and prior weights."""
import json
import sys
from pathlib import Path
import torch

RETAINED = Path('/ml2/alz/runs/empf_comparison_20261117/fits/native/pbulk/hvg2000/fold0/seed0')
RUN = Path('/ml2/alz/runs/research_20261120/pparg-critic')


def main(updates=5000):
    mine = torch.load(RUN/'fits/pbulk/hvg2000/fold0/seed0/final.pt', map_location='cpu', weights_only=False)
    snap = torch.load(RETAINED/f'snap_{updates}.pt', map_location='cpu', weights_only=False)
    models = mine['policy']['models'] if 'models' in mine['policy'] else None
    G = mine['policy']['models']['generator'] if models and 'generator' in models else None
    out = {}
    for key, ref in (('generator', snap['generator']), ('prior', snap['prior'])):
        live = mine['policy']['models'][key]
        out[key] = dict(tensors=len(ref), equal=all(torch.equal(live[k].cpu(), ref[k]) for k in ref) and set(live) == set(ref),
                        max_abs_diff=max(float((live[k].cpu().float() - ref[k].float()).abs().max()) for k in ref))
    out['completed_updates'] = mine['completed_updates']
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    v = main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
    Path(__file__).with_name('harness_check.json').write_text(json.dumps(v, indent=2) + '\n')
