import math

import pytest
from sklearn.metrics import cohen_kappa_score

from evalharness.stats import bootstrap_ci, cohen_kappa, paired_bootstrap_delta, spearman


def test_bootstrap_ci_of_constant():
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0, 1.0)


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1] * 10
    a = bootstrap_ci(values, seed=3)
    assert a == bootstrap_ci(values, seed=3)
    mean, low, high = a
    assert mean == pytest.approx(0.7) and low < mean < high and 0.55 < low and high < 0.85
    continuous = [(i * 7919) % 101 / 100 for i in range(60)]  # 0/1 data can land on the same percentile grid
    assert bootstrap_ci(continuous, seed=3) != bootstrap_ci(continuous, seed=4)


def test_bootstrap_ci_wider_with_fewer_items_and_with_higher_level():
    big = [i % 2 for i in range(400)]
    small = big[:20]
    w = lambda ci: ci[2] - ci[1]  # noqa: E731
    assert w(bootstrap_ci(small)) > w(bootstrap_ci(big))
    assert w(bootstrap_ci(big, level=0.99)) > w(bootstrap_ci(big, level=0.9))


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])


def test_bootstrap_ci_handles_a_single_value():
    assert bootstrap_ci([0.5]) == (0.5, 0.5, 0.5)


def test_paired_delta_sign():
    mean, low, high = paired_bootstrap_delta([0.0] * 10, [1.0] * 10)
    assert mean == 1.0 and low > 0
    mean, low, high = paired_bootstrap_delta([1.0] * 10, [0.0] * 10)
    assert mean == -1.0 and high < 0


def test_paired_delta_needs_equal_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_delta([1, 2], [1])


A = [0, 1, 2, 2, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2]
B = [0, 2, 2, 1, 1, 0, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2]


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_cohen_kappa_matches_sklearn(weights):
    assert cohen_kappa(A, B, weights=weights) == pytest.approx(cohen_kappa_score(A, B, weights=weights))


def test_cohen_kappa_perfect_and_degenerate():
    assert cohen_kappa(A, A) == 1.0
    assert math.isnan(cohen_kappa([1, 1, 1], [1, 1, 1]))


def test_cohen_kappa_labels_missing_from_one_rater():
    a, b = [0, 0, 1, 1, 2], [0, 1, 1, 1, 1]  # rater b never says 2
    assert cohen_kappa(a, b, weights="linear") == pytest.approx(cohen_kappa_score(a, b, weights="linear"))


def test_cohen_kappa_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="same length"):
        cohen_kappa([1, 2], [1])


def test_spearman_matches_known_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)  # ties get average ranks
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
