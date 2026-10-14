"""Small explicit feature trainer; real cohorts require a separately approved budget."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import time
import torch
import torch.nn.functional as F
from particlegan import scale_learning_rates
from .gene_checkpoint import model_hashes
from .gene_calibration import calibration_split, split_hash, relativistic_realness


@contextmanager
def frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters(): p.requires_grad_(False)
    try: yield
    finally:
        for p, flag in zip(module.parameters(), flags): p.requires_grad_(flag)


def _norm(module):
    values = [p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.sqrt(sum(values))) if values else 0.


def _finite(loss, module):
    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None):
        raise FloatingPointError('nonfinite objective/gradient; cohort stopped')


def _gradient_geometry(adv, reconstruction, module):
    """Measure weighted component scale/conflict on shared generator parameters."""
    parameters = [p for p in module.parameters() if p.requires_grad]
    ga = torch.autograd.grad(.1 * adv, parameters, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(reconstruction, parameters, retain_graph=True, allow_unused=True)
    aa = sum((a.square().sum() for a in ga if a is not None), adv.new_zeros(()))
    rr = sum((b.square().sum() for b in gr if b is not None), adv.new_zeros(()))
    dot = sum(((a * b).sum() for a,b in zip(ga,gr) if a is not None and b is not None), adv.new_zeros(()))
    return {'weighted_adversarial_grad_norm':float(aa.sqrt()), 'reconstruction_grad_norm':float(rr.sqrt()),
            'adversarial_to_reconstruction_grad_ratio':float((aa / rr.clamp_min(1e-30)).sqrt()),
            'component_gradient_cosine':float(dot / (aa * rr).sqrt()) if aa > 0 and rr > 0 else None}


def _snapshot(module):
    return [p.detach().clone() for p in module.parameters() if p.requires_grad]


def _relative_update(module, before):
    parameters = [p for p in module.parameters() if p.requires_grad]
    numerator = sum(((p.detach()-b).square().sum() for p,b in zip(parameters,before)))
    denominator = sum((b.square().sum() for b in before)).clamp_min(1e-30)
    return float((numerator/denominator).sqrt())


def _sample_real(batch, candidate, tensors, generator, k):
    u = torch.rand(batch, k, generator=generator)
    local = (u * tensors['counts'][candidate, None]).long()
    return tensors['starts'][candidate, None] + local


def _train_arm(models, study, *, decision_cells=8, max_wall_seconds=60, grant=None, frozen_source=None):
    """FIT identities only; no validation scoring, early stopping or source selection.

    Synthetic unit checks need no grant. A real cohort requires explicit parent
    authorization/capacity; this function does not discover/reserve GPUs.
    """
    started = time.monotonic()
    cfg = models.config
    models.validate_study(study)
    if max_wall_seconds <= 0: raise ValueError('positive wall budget required')
    if study.protocol.data_kind != 'synthetic':
        from .gene_protocol import validate_grant
        validate_grant(grant or {},study,cfg,str(models.device),'train',max_wall_seconds)
    if decision_cells < 1: raise ValueError('decision_cells must be positive')
    held_out = calibration_split(study.protocol.fit)
    t = study.training_tensors(exclude=held_out); device = models.device
    g, d, j, prior, recipe = models.generator, models.critic, models.decision, models.prior, models.recipe
    probe = cfg.arm == 'G_frozen'
    if probe:
        if frozen_source is None or frozen_source.config.arm != cfg.frozen_generator_arm:
            raise ValueError('frozen-generator probe requires an explicitly fitted G source')
        if cfg.frozen_generator_arm == 'G' and frozen_source.feature_provenance != models.feature_provenance:
            raise ValueError('frozen probe feature source differs')
        if frozen_source.training_protocol_sha256 != study.protocol.fingerprint:
            raise ValueError('frozen source was not fitted on this protocol; legacy/inner leakage cannot select arms')
        g.load_state_dict(frozen_source.generator.state_dict(), strict=True)
        prior.load_state_dict(frozen_source.prior.state_dict(), strict=True)
        g.requires_grad_(False); prior.requires_grad_(False); d.requires_grad_(False)
        before_frozen = model_hashes(models)
        opt_g = opt_d = None
    else:
        import copy
        opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
        penalty, loss = recipe.make_critic_penalty(opt_d), recipe.make_loss()
        base = [[group['lr'] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    opt_j = torch.optim.Adam([p for p in j.parameters() if p.requires_grad], lr=cfg.lr, betas=(0., .999))
    # Independent streams keep constructor/projector and head draws out of G batches/latents.
    batch_rng = torch.Generator().manual_seed(cfg.seed + 201)
    pair_rng = torch.Generator().manual_seed(cfg.seed + 202)
    latent_g = torch.Generator(device=device).manual_seed(cfg.seed + 203)
    latent_j = torch.Generator(device=device).manual_seed(cfg.seed + 204)
    digests = {k: hashlib.sha256() for k in ('cell_batch', 'pairs', 'g_latents', 'j_latents')}
    history = []; steps = 0
    for step in range(cfg.steps):
        if time.monotonic() - started > max_wall_seconds: raise TimeoutError('fixed training budget exhausted; no extension')
        row = {'step': step + 1}
        diagnostics = step == 0 or step == cfg.steps-1 or (step+1) % 1000 == 0
        if not probe:
            scale_learning_rates(step, recipe, (opt_g, opt_d), base, prior)
            choice = torch.randint(len(t['outcomes']), (cfg.batch_size,), generator=batch_rng)
            digests['cell_batch'].update(choice.numpy().tobytes())
            ids = t['ids'][choice].to(device); outcome = t['outcomes'][choice].to(device)
            control = study.control.to(device).expand(cfg.batch_size, -1)
            condition = torch.cat([control, g.onehot[ids]], -1)  # Critic condition stays one-hot in all arms.
            z, ix = prior.sample(cfg.batch_size, generator=latent_g)
            digests['g_latents'].update(ix.cpu().numpy().tobytes())
            fake = g(z, control, ids)
            opt_d.zero_grad(set_to_none=True)
            real_d, fake_d = d(outcome, condition), d(fake.detach(), condition)
            dloss = loss.d_loss(real_d, fake_d) + penalty(d, outcome, fake.detach(), condition)
            dloss.backward(); _finite(dloss, d); row['d_grad_norm'] = _norm(d); opt_d.step()
            opt_d.zero_grad(set_to_none=True); opt_g.zero_grad(set_to_none=True)
            with frozen(d):
                adv = loss.g_loss(d(fake, condition), d(outcome, condition))
                rec = F.mse_loss(fake, outcome)
                gloss = .1 * adv + rec
                if diagnostics: row.update(_gradient_geometry(adv, rec, g))
                gloss.backward()
            _finite(gloss, g); _finite(gloss, prior); row.update(g_loss=float(gloss.detach()), d_loss=float(dloss.detach()),
                                        reconstruction=float(rec.detach()), adversarial=float(adv.detach()), g_grad_norm=_norm(g))
            if diagnostics:
                before_g = _snapshot(g)
                row.update(prior_grad_norm=_norm(prior), d_real_logit_mean=float(real_d.detach().mean()),
                           d_fake_logit_mean=float(fake_d.detach().mean()),
                           d_fake_saturation_fraction=float((fake_d.detach().abs()>4.595).float().mean()),
                           g_projection_grad_norm=_norm(g.encoder) if g.encoder is not None else 0.)
            opt_g.step()
            if diagnostics: row['g_relative_parameter_update'] = _relative_update(g, before_g)
        n = len(t['fit_ids'])
        targets = torch.randint(n, (cfg.batch_size,), generator=pair_rng)
        labels = (torch.rand(cfg.batch_size, generator=pair_rng) < .5).float()
        offset = torch.randint(1, n, (cfg.batch_size,), generator=pair_rng)
        candidates = torch.where(labels > 0, targets, (targets + offset) % n)
        rows = _sample_real(cfg.batch_size, candidates, t, pair_rng, decision_cells)
        digests['pairs'].update(torch.stack([targets, candidates], 1).numpy().tobytes())
        ids = t['fit_ids'][candidates].to(device)
        control = study.control.to(device).expand(cfg.batch_size, -1)
        target = t['targets'][targets].to(device)
        real = t['real_candidates'][rows].mean(1).to(device)
        labels = labels.to(device)
        real_logits = j.logits(control, real, target, ids)
        with torch.no_grad():
            many_ids = ids.repeat_interleave(decision_cells)
            many_ctrl = study.control.to(device).expand(len(many_ids), -1)
            z, ix = prior.sample(len(many_ids), generator=latent_j)
            digests['j_latents'].update(ix.cpu().numpy().tobytes())
            generated_cells = g(z, many_ctrl, many_ids)
            generated = generated_cells.reshape(cfg.batch_size, decision_cells, -1).mean(1)
            raw_gate = 1.
            if probe: realness = torch.ones(cfg.batch_size, device=device)
            else:
                cond = torch.cat([many_ctrl, g.onehot[many_ids]], -1)
                # Relativistic gate: RpGAN logits have a free shared offset, so only D(gen)-D(real) is meaningful.
                real_cells = t['real_candidates'][rows].reshape(-1, t['real_candidates'].shape[-1]).to(device)
                fake_d_j = d(generated_cells, cond).reshape(cfg.batch_size, decision_cells)
                real_d_j = d(real_cells, cond).reshape(cfg.batch_size, decision_cells)
                realness = relativistic_realness(fake_d_j, real_d_j).mean(1)
                raw_gate = float(fake_d_j.sigmoid().mean())  # diagnostic only: old offset-dependent gate
        fake_logits = j.logits(control, generated, target, ids)
        # Stable logits BCE fixes numerical saturation; weights/objective are declared for every matched arm.
        jloss = F.binary_cross_entropy_with_logits(real_logits, labels)
        jloss = jloss + (F.binary_cross_entropy_with_logits(fake_logits, labels, reduction='none') * realness).mean()
        opt_j.zero_grad(set_to_none=True); jloss.backward(); _finite(jloss, j)
        row.update(j_loss=float(jloss.detach()), j_grad_norm=_norm(j), real_positive_rate=float(labels.mean()),
                   real_accuracy=float(((real_logits.detach() >= 0) == (labels >= .5)).float().mean()))
        if diagnostics:
            row.update(j_projection_grad_norm=_norm(j.encoder) if j.encoder is not None else 0.,
                       j_real_saturation_fraction=float((real_logits.detach().abs()>4.595).float().mean()),
                       j_fake_saturation_fraction=float((fake_logits.detach().abs()>4.595).float().mean()),
                       realness_weight_mean=float(realness.mean()), realness_weight_min=float(realness.min()), raw_sigmoid_gate_mean=raw_gate,
                       generated_within_candidate_variance=float(generated_cells.reshape(cfg.batch_size,decision_cells,-1).var(1,unbiased=False).mean()))
        opt_j.step(); history.append(row); steps += 1
    if probe:
        after = model_hashes(models)
        if any(before_frozen[k] != after[k] for k in ('generator', 'prior', 'critic')):
            raise RuntimeError('frozen probe changed generator/prior/critic')
    models.training_protocol_sha256 = study.protocol.fingerprint
    return {'arm': cfg.arm, 'steps': steps, 'history': history, 'streams': {k:v.hexdigest() for k,v in digests.items()},
            'wall_seconds': time.monotonic() - started, 'parameter_counts': models.counts(),
            'objective': 'single-cell MSE + .1 GAN; independent candidate/query pools; stable logits BCE',
            'diagnostics_interval':1000, 'update_ratio':'1D:1G:1J; frozen probe 1J only',
            'selection_scoring_during_fit': False, 'frozen_generator_probe': probe,
            'gate': 'constant_1_frozen_probe' if probe else 'relativistic_sigmoid(D(gen)-D(real_matched))',
            'calibration_identities': list(held_out), 'calibration_split_sha256': split_hash(held_out),
            'excluded_from': ['generator', 'critic', 'decision_head'] if held_out else []}


def train_arm(models, study, **kwargs):
    # Scope incidental public-recipe randomness too; feature constructors cannot alter it.
    devices = [models.device.index] if models.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(models.config.seed + 205)
        return _train_arm(models, study, **kwargs)
