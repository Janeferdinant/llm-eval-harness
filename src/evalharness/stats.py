"""Bootstrap confidence intervals, paired deltas, Cohen's kappa and Spearman correlation (numpy only)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

_CHUNK_CELLS = 2_000_000  # resample matrix cells per chunk, to bound memory


def bootstrap_ci(values: Sequence[float], *, n_resamples: int = 10_000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high): a percentile bootstrap interval for the mean of `values`."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("cannot bootstrap an empty sequence")
    mean = float(x.mean())
    if x.size == 1 or bool(np.all(x == x[0])):
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    chunk = max(1, _CHUNK_CELLS // x.size)
    for start in range(0, n_resamples, chunk):
        k = min(chunk, n_resamples - start)
        means[start:start + k] = x[rng.integers(0, x.size, size=(k, x.size))].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return mean, float(low), float(high)


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], **kw) -> tuple[float, float, float]:
    """Mean of (b minus a) over paired items, with its bootstrap interval. Positive means b is higher."""
    if len(a) != len(b):
        raise ValueError(f"paired samples must have the same length, got {len(a)} and {len(b)}")
    return bootstrap_ci(np.asarray(b, dtype=float) - np.asarray(a, dtype=float), **kw)


def cohen_kappa(a: Sequence[int], b: Sequence[int], weights: Literal["linear", "quadratic"] | None = None) -> float:
    """Cohen's kappa between two raters. Labels are ordered by value, which weighted kappa needs.
    Returns nan when chance agreement leaves nothing to measure (for example, a single label on both sides)."""
    if len(a) != len(b):
        raise ValueError(f"ratings must have the same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("cannot compute kappa on empty ratings")
    labels = sorted(set(a) | set(b))
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    observed = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        observed[index[x], index[y]] += 1
    observed /= len(a)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    denominator = float((w * expected).sum())
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - (w * observed).sum() / denominator)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, with tied values given their average rank. nan if either side is constant."""
    if len(x) != len(y):
        raise ValueError(f"samples must have the same length, got {len(x)} and {len(y)}")
    rx, ry = _average_ranks(np.asarray(x, dtype=float)), _average_ranks(np.asarray(y, dtype=float))
    if rx.std() == 0.0 or ry.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
