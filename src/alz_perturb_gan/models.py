"""Small conditional generator, critic, and decision head.

The generator maps (particle latent, control state, perturbation embedding)
to a perturbed expression profile. The critic scores real versus generated
outcomes under the same condition. The decision head answers one typed
question: does this perturbation move the control state toward a target?
"""

from __future__ import annotations

import torch
from torch import nn

from .judgments import MOVES_TOWARD, NoulJudgment


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden, hidden),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden, out_dim),
    )


class ConditionalGenerator(nn.Module):
    """Predict a residual on top of the control state."""

    def __init__(self, n_genes: int, emb_dim: int, z_dim: int, hidden: int = 64):
        super().__init__()
        self.net = _mlp(z_dim + n_genes + emb_dim, hidden, n_genes)

    def forward(self, z: torch.Tensor, control: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([z, control, embedding], dim=-1))
        return control + delta


class ConditionalCritic(nn.Module):
    """Score an outcome. Extra arguments are the ParticleGAN conditioning slot."""

    def __init__(self, n_genes: int, condition_dim: int, hidden: int = 64):
        super().__init__()
        self.net = _mlp(n_genes + condition_dim, hidden, 1)

    def forward(self, outcome: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([outcome, condition], dim=-1)).squeeze(-1)


class DecisionModel(nn.Module):
    """P(candidate perturbation P is the one that produced target T), a single noul.

    Inputs are pseudobulk-level: the control mean, an aggregated profile for the
    candidate P (a mean of real cells in training, the generator's averaged
    prediction at evaluation), and the target pseudobulk T. Features are the
    candidate and target deltas from control plus their product, so the head
    scores whether P's predicted shift matches T's shift. There is no identity
    embedding input, so held-out perturbations are scored through the generator.
    """

    def __init__(self, n_genes: int, hidden: int = 64):
        super().__init__()
        self.question = MOVES_TOWARD
        self.net = _mlp(3 * n_genes, hidden, 1)

    def forward(self, control: torch.Tensor, profile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_delta = profile - control
        t_delta = target - control
        logits = self.net(torch.cat([p_delta, t_delta, p_delta * t_delta], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)

    def judge(self, control: torch.Tensor, profile: torch.Tensor, target: torch.Tensor) -> NoulJudgment:
        if control.ndim != 1:
            raise ValueError("judge expects one control, one candidate profile, and one target")
        with torch.no_grad():
            probability = float(self.forward(control.unsqueeze(0), profile.unsqueeze(0), target.unsqueeze(0))[0])
        return NoulJudgment(question=self.question, noul=probability)
