"""Isolation properties for input-only batched inverse fitting."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from inverse_donor import inverse_fit, expected_profile, particle_dictionary


class ToyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(21)
        self.weight = torch.nn.Parameter(torch.randn(8, 20, generator=gen), requires_grad=False)

    def parameters_nb(self, z, condition, cells, library, generator):
        p = torch.softmax(z @ self.weight, -1)
        return p[:, None].expand(-1, cells, -1)*library[..., None], torch.ones(20)


def test_each_donor_fit_is_isolated_from_other_donors_observations():
    torch.set_num_threads(3)
    gen = torch.Generator().manual_seed(22)
    model = ToyGenerator()
    particles = torch.randn(16, 8, generator=gen)
    c = torch.zeros(2, 7)
    dictionary = particle_dictionary(model, particles, c)
    observed = expected_profile(model, particles[:2], c).detach()
    first, _ = inverse_fit(model, particles, dictionary, c, observed, steps=5)
    changed = observed.clone()
    changed[1] += torch.linspace(-1, 1, 20)
    second, _ = inverse_fit(model, particles, dictionary, c, changed, steps=5)
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])
    assert all(p.grad is None for p in model.parameters())
