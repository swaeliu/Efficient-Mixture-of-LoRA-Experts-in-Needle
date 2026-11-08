"""Experiment-protocol helpers (Phase 4A): stable seeds and robustness statistics.

Everything here is deterministic and independent of Python's process-randomised
``hash()``. The statistics keep the *run* (seed) as the unit of analysis.
"""
from __future__ import annotations

import math
import zlib
from typing import Sequence

import numpy as np

__all__ = ["derive_seed", "is_failure", "binomial_ci", "paired_bootstrap_ci", "summary_stats"]


def derive_seed(base_seed: int, tag: str) -> int:
    """Stable 32-bit seed for a named randomness source derived from a base seed.

    ``derive_seed(s, "data") != derive_seed(s, "init")`` and both are reproducible
    across processes (crc32 of the tag, not ``hash``).
    """
    return int((int(base_seed) * 1_000_003 + zlib.crc32(tag.encode("utf-8"))) % (2**32))


def is_failure(val_mse: float, threshold: float) -> bool:
    """Convergence-failure criterion: validation MSE above the pre-registered threshold (or non-finite)."""
    return (not math.isfinite(val_mse)) or val_mse > threshold


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def binomial_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided confidence interval for a binomial proportion."""
    if not 0 <= successes <= n or n <= 0:
        raise ValueError("need 0 <= successes <= n and n > 0")

    lower = 0.0 if successes == 0 else _bisect_increasing(lambda p: 1 - _binom_cdf(successes - 1, n, p), alpha / 2)
    upper = 1.0 if successes == n else _bisect_increasing(lambda p: 1 - _binom_cdf(successes, n, p), 1 - alpha / 2)
    return lower, upper


def _bisect_increasing(f, target, lo=0.0, hi=1.0, iters=200):
    """Solve f(p) = target for an increasing f on [0, 1]."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05, statistic=np.mean) -> dict:
    """Bootstrap CI for statistic(a - b) over paired observations (resampling pairs)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of equal length")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = statistic(d[idx], axis=1)
    return {"estimate": float(statistic(d)), "ci_low": float(np.quantile(boots, alpha / 2)),
            "ci_high": float(np.quantile(boots, 1 - alpha / 2)), "n": int(len(d)), "n_boot": n_boot}


def summary_stats(values: Sequence[float]) -> dict:
    v = np.asarray(values, float)
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(med),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "min": float(v.min()), "max": float(v.max())}
