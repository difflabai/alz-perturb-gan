"""Plain PyTorch training loop for the conditional GAN and the decision head.

Optimizers and the critic penalty come from ParticleGAN 0.8's recipe API
(`make_optimizers`, `make_critic_penalty`, `make_loss`, `scale_learning_rates`).
The decision head is not a GAN network. It answers "is candidate perturbation
P the source of target pseudobulk T?" Each step draws a training target T (a
training perturbation's pseudobulk) and a candidate P: with probability
`decision_positive_rate` P is T's own perturbation (label 1), otherwise a
different training perturbation (label 0). Everything is pseudobulk-level:

* real term: P is represented by the mean of `decision_cells` sampled real
  training cells of P (a noisy pseudobulk), never a single cell
* generated term: P is represented by the mean of the same number of generator
  draws for P, with the same label, weighted by the detached critic probability
  that those draws look real

The second term is the adversarial signal. It does not update the critic.
At evaluation the candidate profile is the generator's averaged prediction,
so held-out perturbations are scored without any identity shortcut.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from particlegan import scale_learning_rates
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from .particle_integration import make_recipe

from .data import PerturbationSet
from .models import ConditionalCritic, ConditionalGenerator, DecisionModel


@dataclass
class GANDecision:
    generator: ConditionalGenerator
    critic: ConditionalCritic
    decision: DecisionModel
    prior: torch.nn.Module
    recipe: object
    history: dict[str, list[float]] = field(default_factory=dict)
    steps_completed: int = 0

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    @torch.no_grad()
    def predict_outcome(self, control: torch.Tensor, embedding: torch.Tensor, n_draws: int = 8) -> torch.Tensor:
        """Average generator draws. `control` is [G] or [N, G]; embedding matches."""
        single = control.ndim == 1
        if single:
            control = control.unsqueeze(0)
            embedding = embedding.unsqueeze(0)
        control = control.to(self.device)
        embedding = embedding.to(self.device)
        self.generator.eval()
        draws = []
        for _ in range(n_draws):
            z, _ = self.prior.sample(control.shape[0])
            draws.append(self.generator(z, control, embedding))
        mean = torch.stack(draws).mean(dim=0)
        self.generator.train()
        mean = mean.cpu()
        return mean[0] if single else mean

    @torch.no_grad()
    def score_candidates(self, control: torch.Tensor, profiles: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Noul for one target against a batch of candidate pseudobulk profiles [L, G]."""
        self.decision.eval()
        if profiles.ndim == 1:
            profiles = profiles.unsqueeze(0)
        target = target.reshape(-1, profiles.shape[-1]).expand(profiles.shape[0], -1).to(self.device)
        control = control.reshape(-1, profiles.shape[-1]).expand(profiles.shape[0], -1).to(self.device)
        scores = self.decision(control, profiles.to(self.device), target)
        self.decision.train()
        return scores.cpu()

    def predict_profiles(self, data: PerturbationSet, names: list[str], n_draws: int = 8) -> torch.Tensor:
        lookup = data.by_name()
        embedding = torch.stack([lookup[name].embedding for name in names])
        control = data.control_mean.unsqueeze(0).expand(len(names), -1)
        return self.predict_outcome(control, embedding, n_draws=n_draws)


class CandidateSampler:
    """Draw (target, candidate, label) triples over training perturbations.

    Positive: candidate == target's perturbation. Negative: a uniformly chosen
    different training perturbation. Needs at least two training perturbations
    with cells, otherwise only one class could ever appear.
    """

    def __init__(self, data: PerturbationSet, positive_rate: float = 0.5):
        if not 0.0 < positive_rate < 1.0:
            raise ValueError("decision positive rate must be in (0, 1)")
        names, profiles = data.train_profiles()
        lookup = {name: index for index, name in enumerate(names)}
        raw = torch.tensor([lookup.get(name, -1) for name in data.cell_perturbation], dtype=torch.long)
        keep = [i for i in range(len(names)) if bool((raw == i).any())]
        if len(keep) < 2:
            raise ValueError("decision head needs at least two training perturbations with cells")
        remap = torch.full((len(names) + 1,), -1, dtype=torch.long)
        remap[torch.tensor(keep)] = torch.arange(len(keep))
        cell_index = torch.where(raw >= 0, remap[raw.clamp(min=0)], raw)
        order = torch.argsort(cell_index, stable=True)
        order = order[cell_index[order] >= 0]
        self.names = [names[i] for i in keep]
        self.profiles = profiles[keep]
        self.sorted_cells = order
        self.counts = torch.bincount(cell_index[order], minlength=len(keep))
        self.starts = torch.cumsum(self.counts, 0) - self.counts
        self.embeddings = torch.stack([data.by_name()[name].embedding for name in self.names])
        self.positive_rate = positive_rate

    def __len__(self) -> int:
        return len(self.names)

    def sample(self, batch: int, generator: torch.Generator):
        n = len(self.names)
        target = torch.randint(n, (batch,), generator=generator)
        label = (torch.rand(batch, generator=generator) < self.positive_rate).float()
        offset = torch.randint(1, n, (batch,), generator=generator)
        candidate = torch.where(label > 0, target, (target + offset) % n)
        return target, candidate, label

    def cells_for(self, candidate: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
        """Indices [B, k] into the training cells of each candidate, with replacement."""
        draw = torch.rand(candidate.shape[0], k, generator=generator)
        local = (draw * self.counts[candidate, None]).long()
        local = torch.minimum(local, self.counts[candidate, None] - 1)
        return self.sorted_cells[self.starts[candidate, None] + local]


def train_gan_decision(
    data: PerturbationSet,
    *,
    steps: int = 40,
    batch_size: int = 16,
    hidden: int = 64,
    seed: int = 0,
    rec_weight: float = 1.0,
    generator_adv_weight: float = 0.1,
    decision_adv_weight: float = 1.0,
    decision_positive_rate: float = 0.5,
    decision_cells: int = 8,
    n_particles: int = 64,
    z_dim: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
    trainer: str = "bcap",
    progress: Callable[[int, dict[str, float]], None] | None = None,
) -> GANDecision:
    """Fit the conditional GAN and the decision head on training cells only."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if data.cell_control.shape[0] < 1:
        raise ValueError("dataset has no training cells")
    torch.manual_seed(seed)
    if trainer == "bispf":
        raise ValueError("BISPF conditional transport contract is not reviewed; use bcap")
    recipe = make_recipe(
        trainer,
        total_steps=steps,
        batch_size=min(batch_size, data.cell_control.shape[0]),
        z_dim=z_dim,
        num_particles=n_particles,
        lr=lr,
        d_lr_mult=1.0,
        prior_lr_mult=1.0,
    )
    n_genes = data.n_genes
    emb_dim = data.embedding_dim
    condition_dim = n_genes + emb_dim
    generator = ConditionalGenerator(n_genes, emb_dim, recipe.z_dim, hidden=hidden).to(device)
    critic = ConditionalCritic(n_genes, condition_dim, hidden=hidden).to(device)
    decision = DecisionModel(n_genes, hidden=hidden).to(device)
    prior = recipe.make_prior().to(device)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior, ema_critic=copy.deepcopy(critic))
    opt_j = torch.optim.Adam(decision.parameters(), lr=lr, betas=(0.0, 0.999))
    penalty = recipe.make_critic_penalty(opt_d)
    gan_loss = recipe.make_loss()
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    sampler = CandidateSampler(data, positive_rate=decision_positive_rate)
    if decision_cells < 1:
        raise ValueError("decision_cells must be positive")
    batch_rng = torch.Generator().manual_seed(seed + 1)
    history = {"d_loss": [], "g_loss": [], "j_loss": [], "real_accuracy": [], "real_positive_rate": []}
    n_cells = data.cell_control.shape[0]
    batch = recipe.batch_size

    for step in range(recipe.total_steps):
        scale_learning_rates(step, recipe, (opt_g, opt_d), base_lrs, prior)
        choice = torch.randint(n_cells, (batch,), generator=batch_rng)
        control = data.cell_control[choice].to(device)
        outcome = data.cell_outcome[choice].to(device)
        embedding = data.cell_embedding[choice].to(device)
        condition = torch.cat([control, embedding], dim=-1)

        z, indices = prior.sample(batch)
        fake = generator(z, control, embedding)

        d_real = critic(outcome, condition)
        d_fake_detached = critic(fake.detach(), condition)
        d_loss = gan_loss.d_loss(d_real, d_fake_detached) + penalty(critic, outcome, fake.detach(), condition)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        d_fake = critic(fake, condition)
        d_real_for_g = critic(outcome, condition)
        adversarial = gan_loss.g_loss(d_fake, d_real_for_g)
        reconstruction = F.mse_loss(fake, outcome)
        g_loss = generator_adv_weight * adversarial + rec_weight * reconstruction
        opt_g.zero_grad()
        if hasattr(opt_g, "set_sampled_rows"): opt_g.set_sampled_rows(prior.z, indices)
        constraint_geometry_backward(g_loss, opt_g, (generator_adv_weight * adversarial,))
        opt_g.step()

        target_index, candidate, real_label = sampler.sample(batch, batch_rng)
        cells = sampler.cells_for(candidate, decision_cells, batch_rng)  # [B, k]
        target = sampler.profiles[target_index].to(device)
        real_label = real_label.to(device)
        flat = cells.reshape(-1)
        cell_ctrl = data.cell_control[flat].to(device)
        candidate_profile = data.cell_outcome[flat].to(device).reshape(batch, decision_cells, -1).mean(1)
        decision_control = cell_ctrl.reshape(batch, decision_cells, -1).mean(1)
        noul = decision(decision_control, candidate_profile, target).clamp(1e-6, 1.0 - 1e-6)
        supervised = F.binary_cross_entropy(noul, real_label)
        with torch.no_grad():
            cand_emb = sampler.embeddings[candidate].to(device).repeat_interleave(decision_cells, 0)
            z_j, _ = prior.sample(cand_emb.shape[0])
            fake_cells = generator(z_j, cell_ctrl, cand_emb)
            realness = torch.sigmoid(critic(fake_cells, torch.cat([cell_ctrl, cand_emb], -1)))
            realness = realness.reshape(batch, decision_cells).mean(1)
            fake_profile = fake_cells.reshape(batch, decision_cells, -1).mean(1)
        noul_fake = decision(decision_control, fake_profile, target).clamp(1e-6, 1.0 - 1e-6)
        adversarial_examples = (
            F.binary_cross_entropy(noul_fake, real_label, reduction="none") * realness
        ).mean()
        j_loss = supervised + decision_adv_weight * adversarial_examples
        opt_j.zero_grad()
        j_loss.backward()
        opt_j.step()

        history["d_loss"].append(float(d_loss.detach()))
        history["g_loss"].append(float(g_loss.detach()))
        history["j_loss"].append(float(j_loss.detach()))
        history["real_accuracy"].append(float(((noul.detach() >= 0.5) == (real_label >= 0.5)).float().mean()))
        history["real_positive_rate"].append(float(real_label.mean()))
        if progress is not None:
            progress(step + 1, {key: values[-1] for key, values in history.items()})

    return GANDecision(
        generator=generator,
        critic=critic,
        decision=decision,
        prior=prior,
        recipe=recipe,
        history=history,
        steps_completed=recipe.total_steps,
    )
