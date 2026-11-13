"""Synthetic verification of finite-sample coverage and training-only fitting."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import interval_architecture_audit as A


def main():
    result = A.null_experiment()
    for n, row in result.items():
        assert abs(row['interpolated_empirical'] - row['interpolated_uniform_exact']) < .007
        assert abs(row['rank_empirical'] - row['rank_exact']) < .007
    assert result['32']['interpolated_uniform_exact'] < .76
    assert result['32']['rank_exact'] >= .8
    rng = np.random.default_rng(41)
    low, high = rng.normal(size=(10, 100)) - 1, rng.normal(size=(10, 100)) + 1
    high = np.maximum(high, low + .1)
    observed = rng.normal(size=(24, 10, 100))
    train = np.arange(10) < 6
    first, info = A.fit_interval_diagnostics(low, high, observed, train)
    observed[:, ~train] = 1000
    second, second_info = A.fit_interval_diagnostics(low, high, observed, train)
    assert info == second_info
    for method in first:
        for a, b in zip(first[method], second[method]):
            assert np.array_equal(a, b)
    assert 1 <= A.effective_dimension(rng.normal(size=(200, 5))) <= 5
    print('PASS: finite-M null coverage, rank correction, held-out exclusion and effective-dimension bounds')


if __name__ == '__main__':
    main()
