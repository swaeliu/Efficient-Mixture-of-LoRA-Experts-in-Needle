"""Phase 4B — latent-regime dataset: no regime feature, independence, separability, diagnostics."""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import latent_regime_dataset as lrd  # noqa: E402

CFG = {"dtype": "float32", "data": {"input_dim": 32, "output_dim": 16, "n_regimes": 4, "regime_rank": 3, "delta_rel_norm": 0.5,
                                      "noise_std": 0.0, "n_train": 400, "n_val": 80, "n_test": 80, "bias": False, "delta": 2.0}}


def test_shapes_balance_and_no_regime_feature():
    p = lrd.make_latent_problem(CFG, 7)
    for split in ("train", "val", "test"):
        s = p[split]
        assert s["x"].shape[1] == 32 and s["y"].shape[1] == 16 and s["x"].dtype == np.float32
        np.testing.assert_array_equal(np.bincount(s["z"]), len(s["z"]) // 4)
    # x carries NO deterministic regime code: every coordinate is continuous noise around the mean
    x, z = p["train"]["x"], p["train"]["z"]
    assert np.all(np.std(x, axis=0) > 0.5)
    for r in range(4):  # within-regime scatter is unit variance in every coordinate
        np.testing.assert_allclose(x[z == r].std(axis=0).mean(), 1.0, atol=0.15)


def test_targets_match_planted_corrections_and_means():
    p = lrd.make_latent_problem(CFG, 3)
    s = p["val"]
    y_ref = np.stack([s["x"][i].astype(np.float64) @ (p["W0"].astype(np.float64) + p["deltas"][z].astype(np.float64)) for i, z in enumerate(s["z"])])
    np.testing.assert_allclose(s["y"], y_ref, rtol=1e-4, atol=1e-4)
    for dw in p["deltas"]:
        sv = np.linalg.svd(dw.astype(np.float64), compute_uv=False)
        assert sv[2] > 1e-2 * sv[0] and sv[3] < 1e-5 * sv[0]  # rank 3 up to float32 rounding
        np.testing.assert_allclose(np.linalg.norm(dw), 0.5 * np.linalg.norm(p["W0"]), rtol=1e-4)
    # regime means are at distance delta from the origin and delta*sqrt(2) from each other
    np.testing.assert_allclose(np.linalg.norm(p["mus"], axis=1), 2.0, atol=1e-10)
    for i in range(4):
        for j in range(i + 1, 4):
            np.testing.assert_allclose(np.linalg.norm(p["mus"][i] - p["mus"][j]), 2.0 * np.sqrt(2), atol=1e-10)
    assert p["pairwise_mean_distance_over_sigma"] == pytest.approx(2.0 * np.sqrt(2))


def test_splits_are_independent_and_seeded():
    p = lrd.make_latent_problem(CFG, 11)
    tr, te = p["train"]["x"], p["test"]["x"]
    # no duplicated rows across splits
    assert not any(np.any(np.all(np.isclose(tr, row), axis=1)) for row in te)
    q = lrd.make_latent_problem(CFG, 11)
    np.testing.assert_array_equal(p["train"]["x"], q["train"]["x"])
    np.testing.assert_array_equal(p["W0"], q["W0"])
    r = lrd.make_latent_problem(CFG, 12)
    assert not np.array_equal(p["train"]["x"], r["train"]["x"])
    # same seed, different delta: same W0 and corrections, different inputs
    s = lrd.make_latent_problem(CFG, 11, delta=4.0)
    np.testing.assert_array_equal(s["W0"], p["W0"])
    np.testing.assert_array_equal(s["deltas"][0], p["deltas"][0])
    assert not np.array_equal(s["train"]["x"], p["train"]["x"])


def test_identifiability_decreases_with_overlap():
    accs, ambs = [], []
    for delta in (4.0, 2.0, 1.0):
        p = lrd.make_latent_problem(CFG, 5, delta=delta)
        d = lrd.identifiability(p)
        accs.append(d["bayes_accuracy_test"]); ambs.append(d["ambiguous_fraction_test"])
        assert d["nearest_centroid_accuracy_test"] <= d["bayes_accuracy_test"] + 0.1
    assert accs[0] > 0.97 and accs[0] > accs[1] > accs[2] > 0.25
    assert ambs[0] < ambs[1] < ambs[2]


def test_bayes_posterior_is_normalised_and_oracles_ordered():
    p = lrd.make_latent_problem(CFG, 9, delta=1.5)
    post = lrd.bayes_posterior(p["test"]["x"], p["mus"], p["sigma"])
    np.testing.assert_allclose(post.sum(1), 1.0, atol=1e-12)
    o = lrd.oracle_mses(p)
    assert o["true_regime_oracle_mse"] < 1e-8  # noise-free: planted map reproduces y
    assert 0 <= o["soft_bayes_oracle_mse"] <= o["hard_bayes_oracle_mse"] * 1.5 + 1e-12
    assert o["hard_bayes_oracle_mse"] < o["frozen_backbone_mse"]
    m = lrd.ambiguity_mask(p, "test", 0.9)
    assert m.dtype == bool and m.shape == (80,)
