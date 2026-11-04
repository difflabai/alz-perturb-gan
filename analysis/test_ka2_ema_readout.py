"""Actual-parent CPU EMA state loading, count replay and refusal qualification."""
import copy,json,os,unittest
from pathlib import Path
import torch
import ka2_ema as a
ROOT=Path('/ml2/alz');H=json.loads((ROOT/'analysis/ka2_ema_readout_hypothesis.json').read_text())
Q=torch.load(H['parent_checkpoint_path'],map_location='cpu',weights_only=False)
CACHE=Q['cache_manifest_SHA256']
torch.set_num_threads(1)

class EMA(unittest.TestCase):
    def load(self,q=None,**kw):return a.load(torch.ones(2000),torch.ones(2000),'cpu',Q if q is None else q,cache_sha=kw.get('cache_sha',CACHE),fold=kw.get('fold',0),seed=kw.get('seed',0))
    def test_full_actual_parent_EMA_state_exact_and_storage_independent(self):
        G,P,info=self.load();self.assertEqual(info['raw_model_floating_values'],371105);self.assertEqual(info['EMA_model_floating_values'],316832)
        for name,module in [('generator',G),('prior',P)]:
            self.assertEqual(a.fingerprint(module.state_dict()),a.fingerprint(Q['policy']['averages'][name]))
            self.assertTrue(any(not torch.equal(v,Q['policy']['models'][name][k]) for k,v in module.state_dict().items()))
            for k,v in module.state_dict().items():self.assertNotEqual(v.data_ptr(),Q['policy']['averages'][name][k].data_ptr())
        self.assertTrue(all(not p.requires_grad for p in G.parameters()));self.assertTrue(all(not p.requires_grad for p in P.parameters()))
        self.assertEqual(info['native_default_served_source'],'fast');self.assertEqual(info['training_updates_executed'],0)
    def test_full2000_gene_native_integer_count_sampler_replays(self):
        G,P,_=self.load();before=(a.fingerprint(G.state_dict()),a.fingerprint(P.state_dict()))
        def draw():
            torch.manual_seed(20261104031);z,idx=P.sample(1,generator=torch.Generator().manual_seed(20261104032))
            c=torch.zeros(1,7);c[:,3]=2/3;lib=torch.full((1,400),8000.)
            rates=G.rates(z,c,400,lib,torch.Generator().manual_seed(20261104033));counts=torch.poisson(rates,generator=torch.Generator().manual_seed(20261104034));return idx,rates,counts
        with torch.no_grad():x=draw();y=draw()
        for v,w in zip(x,y):self.assertTrue(torch.equal(v,w))
        counts=x[2];self.assertEqual(tuple(counts.shape),(1,400,2000));self.assertTrue(torch.isfinite(counts).all());self.assertTrue((counts>=0).all());self.assertTrue(torch.equal(counts,counts.floor()))
        self.assertEqual(before,(a.fingerprint(G.state_dict()),a.fingerprint(P.state_dict())))
    def test_incomplete_wrong_cache_seed_recipe_or_original_contract_refused(self):
        for key,value in [('status','recovery'),('completed_updates',19999),('native_recipe','k3p'),('serving_contract',{'weights':'EMA','output_noise':False}),('variant_config',{**a.CFG,'z_dim':2})]:
            q=copy.deepcopy(Q);q[key]=value
            with self.assertRaises(ValueError):self.load(q)
        for kw in ({'cache_sha':'0'*64},{'fold':1},{'seed':1}):
            with self.assertRaises(ValueError):self.load(**kw)
        q=copy.deepcopy(Q);q['recipe']['ema_decay']=.99
        with self.assertRaises(ValueError):self.load(q)
    def test_missing_corrupt_or_wrong_dtype_averages_refused_and_parent_unchanged(self):
        before={family:{name:a.fingerprint(state) for name,state in Q['policy'][family].items()} for family in ('models','averages')}
        q=copy.deepcopy(Q);q['policy']['averages'].pop('prior')
        with self.assertRaises(ValueError):self.load(q)
        key=next(iter(Q['policy']['averages']['generator']))
        for bad in ('nan','dtype','shape'):
            q=copy.deepcopy(Q);v=q['policy']['averages']['generator'][key]
            if bad=='nan':v.flatten()[0]=float('nan')
            elif bad=='dtype':q['policy']['averages']['generator'][key]=v.double()
            else:q['policy']['averages']['generator'][key]=v.flatten()[:1]
            with self.assertRaises(ValueError):self.load(q)
        self.load()
        after={family:{name:a.fingerprint(state) for name,state in Q['policy'][family].items()} for family in ('models','averages')};self.assertEqual(before,after)
        self.assertFalse(torch.cuda.is_initialized());self.assertEqual(Q['policy']['device'],'cuda:0')

if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    unittest.main()
