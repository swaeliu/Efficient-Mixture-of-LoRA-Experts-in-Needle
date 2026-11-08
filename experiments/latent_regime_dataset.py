#!/usr/bin/env python
"""Phase 4B — latent-regime synthetic dataset: the regime is NOT an input feature.

Generative model (all parameters recorded in the returned problem dict):

    z ~ Uniform{0..R-1}
    x | z ~ N(mu_z, sigma^2 I_d),   mu_z = delta * u_z,  u_z orthonormal random directions
    y = x (W0 + dW_z),              dW_z = c_z A_z B_z  rank `regime_rank`, ||dW_z||_F = rel * ||W0||_F

Separability is controlled by `delta` alone (sigma = 1): the Mahalanobis
distance between any two regime means is delta * sqrt(2). Regime identity z is
kept for diagnostics only; models receive x and y and nothing else. Train /
validation / test are independent samples from the same distribution.

Diagnostics provided here (evaluated separately from any MoLoRA model):
  * Bayes posterior p(z | x) under the TRUE generative parameters, its argmax
    accuracy, and an ambiguity flag (max posterior < `ambiguity_threshold`);
  * a nearest-centroid classifier fit on the training labels (data-driven
    identifiability), evaluated on the test split;
  * oracle MSEs: true-z oracle (planted dW_z, = 0 without noise), hard-Bayes
    oracle (dW_{argmax p(z|x)}) and soft-Bayes oracle (sum_z p(z|x) dW_z) --
    lower bounds for any router that only sees x, assuming perfect experts.
"""
from __future__ import annotations

import numpy as np

import needle.nn as nn

SEPARABILITY = {  # delta = ||mu_z|| in units of sigma; pairwise mean distance = delta * sqrt(2)
    "easy": 4.0,  # pairwise Mahalanobis distance 5.66
    "medium": 2.0,  # 2.83
    "hard": 1.0,  # 1.41
}


def make_latent_problem(cfg: dict, seed: int, delta: float | None = None) -> dict:
    d = cfg["data"]
    dtype = np.dtype(cfg["dtype"])
    in_dim, out_dim, R = d["input_dim"], d["output_dim"], d["n_regimes"]
    delta = d["delta"] if delta is None else float(delta)
    sigma = float(d.get("sigma", 1.0))

    np.random.seed(seed)  # Needle initialisers use the global RNG
    base = nn.Linear(in_dim, out_dim, bias=d["bias"], dtype=cfg["dtype"])
    W0 = base.weight.numpy().astype(np.float64)

    rng = np.random.default_rng(seed)
    # orthonormal regime directions -> means
    Q, _ = np.linalg.qr(rng.standard_normal((in_dim, R)))
    mus = delta * Q.T  # (R, in_dim)
    deltas = []
    for _ in range(R):
        A = rng.standard_normal((in_dim, d["regime_rank"]))
        B = rng.standard_normal((d["regime_rank"], out_dim))
        dW = A @ B
        dW *= d["delta_rel_norm"] * np.linalg.norm(W0) / np.linalg.norm(dW)
        deltas.append(dW)

    def sample(n):
        assert n % R == 0
        z = rng.permutation(np.repeat(np.arange(R), n // R))
        x = mus[z] + sigma * rng.standard_normal((n, in_dim))
        y = np.einsum("ni,nio->no", x, np.stack([W0 + deltas[zi] for zi in z]))
        if d["bias"]:
            y = y + base.bias.numpy().astype(np.float64)
        if d["noise_std"] > 0:
            y = y + d["noise_std"] * rng.standard_normal(y.shape)
        return {"x": x.astype(dtype), "y": y.astype(dtype), "z": z}

    problem = {"base": base, "W0": W0.astype(dtype), "deltas": [dw.astype(dtype) for dw in deltas], "mus": mus, "sigma": sigma,
               "delta": delta, "pairwise_mean_distance_over_sigma": float(delta * np.sqrt(2) / sigma),
               "train": sample(d["n_train"]), "val": sample(d["n_val"]), "test": sample(d["n_test"]), "rng": rng}
    return problem


# --------------------------------------------------------------------------- #
# Identifiability diagnostics (independent of any adapter model)
# --------------------------------------------------------------------------- #
def bayes_posterior(x: np.ndarray, mus: np.ndarray, sigma: float) -> np.ndarray:
    """p(z | x) for equiprobable isotropic Gaussians with the TRUE means."""
    x = x.astype(np.float64)
    logits = -0.5 * ((x[:, None, :] - mus[None, :, :]) ** 2).sum(-1) / sigma**2
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def identifiability(problem: dict, ambiguity_threshold: float = 0.9) -> dict:
    """Bayes-optimal and nearest-centroid regime identification on the test split, plus ambiguity stats."""
    tr, te = problem["train"], problem["test"]
    post = bayes_posterior(te["x"], problem["mus"], problem["sigma"])
    bayes_pred = post.argmax(1)
    maxp = post.max(1)
    centroids = np.stack([tr["x"][tr["z"] == r].mean(0) for r in range(len(problem["mus"]))]).astype(np.float64)
    nc_pred = ((te["x"].astype(np.float64)[:, None, :] - centroids[None]) ** 2).sum(-1).argmin(1)
    amb = maxp < ambiguity_threshold
    return {
        "bayes_accuracy_test": float((bayes_pred == te["z"]).mean()),
        "nearest_centroid_accuracy_test": float((nc_pred == te["z"]).mean()),
        "mean_max_posterior_test": float(maxp.mean()),
        "ambiguity_threshold": ambiguity_threshold,
        "ambiguous_fraction_test": float(amb.mean()),
        "chance_accuracy": 1.0 / len(problem["mus"]),
        "pairwise_mean_distance_over_sigma": problem["pairwise_mean_distance_over_sigma"],
    }


def oracle_mses(problem: dict, split: str = "test") -> dict:
    """Lower bounds using the planted corrections (NOT fair baselines; labelled oracles)."""
    s = problem[split]
    x = s["x"].astype(np.float64)
    W0 = problem["W0"].astype(np.float64)
    D = np.stack([dw.astype(np.float64) for dw in problem["deltas"]])  # (R, in, out)
    post = bayes_posterior(s["x"], problem["mus"], problem["sigma"])
    y_true_z = np.einsum("ni,nio->no", x, W0[None] + D[s["z"]])
    y_hard = np.einsum("ni,nio->no", x, W0[None] + D[post.argmax(1)])
    dW_soft = np.einsum("nr,rio->nio", post, D)
    y_soft = np.einsum("ni,nio->no", x, W0[None] + dW_soft)
    y = s["y"].astype(np.float64)
    return {"true_regime_oracle_mse": float(((y_true_z - y) ** 2).mean()),
            "hard_bayes_oracle_mse": float(((y_hard - y) ** 2).mean()),
            "soft_bayes_oracle_mse": float(((y_soft - y) ** 2).mean()),
            "frozen_backbone_mse": float(((x @ W0 - y) ** 2).mean())}


def ambiguity_mask(problem: dict, split: str, threshold: float = 0.9) -> np.ndarray:
    post = bayes_posterior(problem[split]["x"], problem["mus"], problem["sigma"])
    return post.max(1) < threshold
