"""Run with ParticleGAN on PYTHONPATH. Uses CPU, synthetic rows and no training.

Compare the public EMPF penalty's reported residual for a correct batch and a
batch with the two groups' spreads exchanged. As a positive control, also halve
both groups' spreads. Width changes only by zero-padding.
The table is a characterization, so an improved observer can run the same probe.
"""

import particlegan
import torch
from torch import nn

from particlegan import get_recipe, init


def fixture(width):
    # Both groups have mean zero. The only informative coordinate has SD 0.5
    # or 2.0. Exchanging the two blocks leaves every pooled row present.
    labels = torch.arange(2).repeat_interleave(16)
    condition = torch.zeros(32, 7)
    condition[:, 0] = labels
    real = torch.zeros(32, width)
    sign = torch.tensor([-1.0, 1.0]).repeat(16)
    real[:, 0] = sign * torch.where(labels == 0, 0.5, 2.0)
    swapped = real.roll(16, dims=0)
    assert torch.equal(real[:, 0].sort().values, swapped[:, 0].sort().values)
    assert torch.equal(real.reshape(2, 16, width).mean(1), swapped.reshape(2, 16, width).mean(1))
    assert torch.equal(swapped[:, 0].reshape(2, 16).std(1, unbiased=False), torch.tensor([2.0, 0.5]))
    return real, swapped, condition


class ConditionalCritic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width + 7, 1)
        init.deterministic_orthogonal_(self, seed=0)

    def forward(self, rows, condition):
        return self.linear(torch.cat((rows, condition), dim=1)).squeeze(1)


def observed_residual(real, fake, condition):
    recipe = get_recipe("empf", reg_every=1, batch_size=real.shape[0])
    critic = ConditionalCritic(real.shape[1])
    optimizer = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(optimizer, input_domain="continuous", collect_stats=True)
    value = penalty(critic, real, fake, condition)
    value.backward()
    assert penalty.last_stats["applied"] and optimizer.record.calls == 1
    assert critic.linear.weight.grad is not None
    assert torch.isfinite(critic.linear.weight.grad).all() and critic.linear.weight.grad.norm() > 0
    return penalty.last_stats["ema_residual"]


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(f"Loaded {particlegan.__file__} | fresh penalty per case | continuous input")
    print("width  correct residual  swapped residual  both spreads halved")
    for width in (64, 65, 2150):
        real, swapped, condition = fixture(width)
        correct_value = observed_residual(real, real, condition)
        swapped_value = observed_residual(real, swapped, condition)
        halved_value = observed_residual(real, real * 0.5, condition)
        print(f"{width:5}  {correct_value:16.6f}  {swapped_value:16.6f}  {halved_value:19.6f}")
