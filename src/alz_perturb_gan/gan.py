"""Plain PyTorch training loop for the conditional GAN and the decision head.

Optimizers and the critic penalty come from ParticleGAN 0.8's recipe API
(`make_optimizers`, `make_critic_penalty`, `make_loss`, `scale_learning_rates`).
The decision head is not a GAN network. It is trained on two labels:

* real cells: 1 when the observed outcome is closer to the target than the control is
* generated outcomes: the same geometric label, weighted by the detached critic
  probability that the generated outcome looks real

The second term is the adversarial signal. It does not update the critic.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from particlegan import get_recipe, scale_learning_rates

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
    def score_moves_toward(self, control: torch.Tensor, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Noul for one target against a batch of embeddings. Higher means more yes."""
        self.decision.eval()
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        control = control.expand(embedding.shape[0], -1).to(self.device)
        target = target.expand(embedding.shape[0], -1).to(self.device)
        embedding = embedding.to(self.device)
        scores = self.decision(control, embedding, target)
        self.decision.train()
        return scores.cpu()

    def predict_profiles(self, data: PerturbationSet, names: list[str], n_draws: int = 8) -> torch.Tensor:
        lookup = data.by_name()
        embedding = torch.stack([lookup[name].embedding for name in names])
        control = data.control_mean.unsqueeze(0).expand(len(names), -1)
        return self.predict_outcome(control, embedding, n_draws=n_draws)


def _geometry_label(outcome: torch.Tensor, control: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    moved = (outcome - target).norm(dim=-1) < (control - target).norm(dim=-1)
    return moved.float()


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
    n_particles: int = 64,
    z_dim: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
    progress: Callable[[int, dict[str, float]], None] | None = None,
) -> GANDecision:
    """Fit the conditional GAN and the decision head on training cells only."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if data.cell_control.shape[0] < 1:
        raise ValueError("dataset has no training cells")
    torch.manual_seed(seed)
    recipe = get_recipe(
        "gan",
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
    decision = DecisionModel(n_genes, emb_dim, hidden=hidden).to(device)
    prior = recipe.make_prior().to(device)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior, ema_critic=copy.deepcopy(critic))
    opt_j = torch.optim.Adam(decision.parameters(), lr=lr, betas=(0.0, 0.999))
    penalty = recipe.make_critic_penalty(opt_d)
    gan_loss = recipe.make_loss()
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    train_names, train_profiles = data.train_profiles()
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
        target_index = torch.randint(len(train_names), (batch,), generator=batch_rng)
        target = train_profiles[target_index].to(device)
        condition = torch.cat([control, embedding], dim=-1)
        real_label = _geometry_label(outcome, control, target)

        z, _ = prior.sample(batch)
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
        g_loss.backward()
        opt_g.step()

        noul = decision(control, embedding, target).clamp(1e-6, 1.0 - 1e-6)
        supervised = F.binary_cross_entropy(noul, real_label)
        with torch.no_grad():
            fake_label = _geometry_label(fake.detach(), control, target)
            realness = torch.sigmoid(critic(fake.detach(), condition))
        adversarial_examples = (
            F.binary_cross_entropy(noul, fake_label, reduction="none") * realness
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
