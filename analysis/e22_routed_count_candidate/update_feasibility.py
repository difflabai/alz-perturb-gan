"""One synthetic native E22 optimizer pair through the complete count callback.

The same fit contexts/targets bind both halves of the pair so native row
evidence describes the actual differentiated function. Native GAN output
noise affects fake critic-observation coordinates; raw counts stay integer.
"""
import datetime
import json
import os
from pathlib import Path

import torch
from particlegan import RoutedBatch, serial_autograd
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from pg_model import standardize_counts
from forward_feasibility import ROOT, build, context, sha


def synthetic_batch(forward):
    counts = torch.poisson(torch.full((32,8,8),5.), generator=torch.Generator().manual_seed(20261104203))
    guard = torch.poisson(torch.full((32,8,8),5.), generator=torch.Generator().manual_seed(20261104204))
    fit_context, guard_context = context(20261104205), context(20261104206)
    fit_context[:,15:-12], guard_context[:,15:-12] = counts.sum(-1), guard.sum(-1)
    pack = lambda x: torch.cat([x, standardize_counts(x,forward.mean,forward.sd)],-1)
    assert not torch.equal(counts,guard) and not torch.equal(fit_context,guard_context)
    return RoutedBatch(fit_context,pack(counts),guard_context,pack(guard))


def advance(rec, policy, forward, batch):
    G, D, P, og, od = policy.G, policy.D, policy.prior, policy.opt_g, policy.opt_d
    loss, reg = rec.make_loss(), rec.make_prior_regularizer(prior=P,weight=1.)
    c = batch.context[:,8:15]
    real = forward.observation(batch.targets)
    with serial_autograd():
        step_noise = policy.begin_step(real,routed=batch,execution_limit=20000)
        assert step_noise.input_sigma == 0
        G.eval(); D.train()
        with torch.no_grad():
            fake = forward.observation(policy.routed_generate(batch.context,sigma=0))
            fake = fake + step_noise.output_sigma*torch.randn(fake.shape,generator=policy.noise_generator)
        policy.observe_critic_pair(real,fake)
        adv = loss.d_loss(D(real,c),D(fake,c))
        penalty = policy.penalty(D,real,fake,c)
        ld = adv+penalty
        od.zero_grad();ld.backward();od.step();policy.after_critic_step()
        D.eval();G.train();D.requires_grad_(False)
        try:
            fake = forward.observation(policy.routed_generate(batch.context,sigma=0))
            fake = fake + step_noise.output_sigma*torch.randn(fake.shape,generator=policy.noise_generator)
            lg = loss.g_loss(D(fake,c),D(real,c))
            raw = reg(P.z)
            total = lg+rec.prior_reg*raw
            og.zero_grad();constraint_geometry_backward(total,og,(lg,))
            gradients = {name:float(p.grad.norm()) if p.grad is not None else None
                         for name,p in [('G_output',G.out[-1].weight),('G_dispersion',G.log_theta),
                                        ('table',P.z),('learned_output_sigma',policy.log_output_sigma)]}
            policy.after_generator_backward(loss_gan=lg.detach(),loss_critic=adv.detach())
            og.step();policy.after_generator_step()
        finally:
            D.requires_grad_(True)
        policy.finish_step()
    return {'step':policy.completed_steps,'D_total':float(ld.detach()),'G_total':float(total.detach()),
            'gradients':gradients,'native_output_sigma':float(policy.output_sigma())}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    rec,policy,forward = build()
    batch = synthetic_batch(forward)
    metrics = advance(rec,policy,forward,batch)
    assert metrics['step'] == 1 and policy._phase == 'ready'
    for value in metrics['gradients'].values():
        assert value is not None and value > 0 and torch.isfinite(torch.tensor(value))
    for module in (policy.G,policy.D,policy.prior):
        for p in module.parameters():assert bool(torch.isfinite(p).all())
    assert policy.recipe.row_evidence_gate and policy.recipe.particle_birth_death
    assert policy.routed_control is not None and not torch.cuda.is_initialized()
    out=Path(__file__).with_suffix('.json');assert not out.exists()
    result={'status':'ACTUAL_ONE_NATIVE_ROUTED_E22_COUNT_OPTIMIZER_PAIR_CPU_FEASIBLE_NOT_MODEL_QUALIFIED',
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metrics':metrics,
            'native_optimizer_pairs_executed':1,'native_row_controls_disabled':False,
            'synthetic_fit_guard_contexts_separate':True,'raw_counts_not_output_noised':True,
            'native_model_qualified':False,'real_fitting_partition_implemented':False,
            'probe_boundary_replay_verified':False,'GPU_context_created':False,'quality_measured':False,
            'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),ROOT/'analysis/e22_routed_count_candidate/forward_feasibility.py']},
            'scope':'One actual8gene/8cell synthetic native optimizer pair with E22Policy/RoutedBatch, full count callback, critic EMA/native penalty and fake-observation output noise. Same fit contexts bind both substeps; protected targets are separate synthetic observations. Not real-data fitting, full probe lifecycle, checkpoint continuation, full2000gene qualification, native allocation or quality.'}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
