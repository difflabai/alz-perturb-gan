"""Check mass conservation and neutral behavior of the fixed count-shape read-out."""
import sys
sys.dont_write_bytecode = True
import torch
import mode_cellshape as C


def main():
    torch.set_num_threads(3)
    g = torch.Generator().manual_seed(31)
    mu = torch.exp(torch.randn(4, 256, 23, generator=g, dtype=torch.float64))
    assert C.shrink_mu(mu, 0) is mu
    for alpha in (.5, 1):
        new = C.shrink_mu(mu, alpha)
        torch.testing.assert_close(new.sum(1), mu.sum(1), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(new.sum(2), mu.sum(2), rtol=1e-12, atol=1e-12)
        assert bool((new > 0).all())
        # At equal library depth this is a mean-preserving contraction; concavity
        # of 1-exp(-mu) must increase the attainable detection fraction.
        same_depth = mu / mu.sum(-1, keepdim=True) * 100
        shrunk = C.shrink_mu(same_depth, alpha)
        assert bool(((-torch.expm1(-shrunk)).mean(1) >=
                     (-torch.expm1(-same_depth)).mean(1) - 1e-12).all())
    class G:
        def parameters_nb(self, *args):
            return mu, torch.ones(23)
    model = G()
    C.patch_generator(model, 0)
    got, theta = model.parameters_nb(None, None, None, None)
    assert got is mu and torch.equal(theta, torch.ones(23))
    spec, value, canon = C.arms.parse_mode('cellshape@0.5:biascal:stagehalf')
    assert value[0] == .5 and spec.name == 'cellshape'
    for bad in ('nan:biascal:stagehalf', '-1:biascal:stagehalf', '1.1:biascal:stagehalf', '0.5'):
        try:
            C.parse_value(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    print('PASS: expected gene totals, cell libraries, positivity, neutral wrapper, and mode grammar')


if __name__ == '__main__':
    main()
