"""Phase 4A — protocol guarantees: stable seeds, paired initialisation, statistics."""
import math
import os
import sys

import numpy as np
import pytest
import yaml

import needle.nn as nn
from molora import DenseMoLoRALinear, LoRALinear, SparseMoLoRALinear
from molora.protocol import binomial_ci, derive_seed, is_failure, paired_bootstrap_ci, summary_stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))


def test_derive_seed_is_stable_and_tag_dependent():
    assert derive_seed(1000, "init") == derive_seed(1000, "init")
    assert derive_seed(1000, "init") != derive_seed(1000, "shuffle")
    assert derive_seed(1000, "init") != derive_seed(1001, "init")
    assert 0 <= derive_seed(2**40, "x") < 2**32
    import zlib
    assert derive_seed(1000, "init") == (1000 * 1_000_003 + zlib.crc32(b"init")) % 2**32


def test_no_python_hash_in_experiment_code():
    """No call to the process-randomised builtin hash() anywhere in molora/ or experiments/."""
    import ast

    bad = []
    for d in ("molora", "experiments"):
        for root, _, files in os.walk(os.path.join(REPO, d)):
            for f in files:
                if f.endswith(".py"):
                    tree = ast.parse(open(os.path.join(root, f)).read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash":
                            bad.append(f)
    assert not bad, f"process-randomised hash() used in {bad}"


def test_paired_initialisation_across_moe_methods():
    """Same init seed -> identical initial experts and router for dense / top-2 / top-1."""
    np.random.seed(0)
    base = nn.Linear(16, 12, bias=False)
    builders = [
        lambda: DenseMoLoRALinear.from_linear(base, n_experts=4, rank=3),
        lambda: SparseMoLoRALinear.from_linear(base, n_experts=4, rank=3, top_k=2),
        lambda: SparseMoLoRALinear.from_linear(base, n_experts=4, rank=3, top_k=1, router_grad_mode="straight_through"),
    ]
    inits = []
    for b in builders:
        np.random.seed(derive_seed(1000, "init"))
        m = b()
        inits.append([p.numpy().copy() for p in m.adapter_parameters() + m.router_parameters()])
    for other in inits[1:]:
        for a, c in zip(inits[0], other):
            np.testing.assert_array_equal(a, c)
    np.random.seed(derive_seed(1001, "init"))
    m2 = builders[0]()
    assert not np.array_equal(m2.experts[0].lora_A.numpy(), inits[0][0])
    assert np.shares_memory(m2.weight.cached_data, base.weight.cached_data)


def test_study_builder_shares_data_backbone_and_pairs_init():
    import robustness_study as rs

    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "phase4_robustness.yaml")))
    cfg["data"].update(n_train=64, n_val=16, n_test=16)
    problem = rs.make_problem(cfg, 1000)
    models = {name: rs.build_model(name, cfg["methods"][name], problem, cfg, 1000) for name in cfg["methods"]}
    W0 = problem["base"].weight.numpy()
    for m in models.values():
        assert np.shares_memory(m.weight.cached_data, problem["base"].weight.cached_data)
        np.testing.assert_array_equal(m.weight.numpy(), W0)
    for a, b in zip(models["dense_molora"].adapter_parameters(), models["sparse_top2"].adapter_parameters()):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    for a, b in zip(models["dense_molora"].router_parameters(), models["sparse_top1_ste"].router_parameters()):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    assert isinstance(models["single_lora"], LoRALinear) and models["matched_lora"].rank == 34


def test_failure_criterion():
    assert not is_failure(0.0005, 0.01)
    assert is_failure(0.02, 0.01)
    assert is_failure(float("nan"), 0.01) and is_failure(float("inf"), 0.01)


def test_binomial_ci_known_values():
    lo, hi = binomial_ci(0, 30)
    assert lo == 0.0 and abs(hi - 0.1157) < 1e-3  # 3/n rule neighbourhood: exact 0.1157
    lo, hi = binomial_ci(30, 30)
    assert hi == 1.0 and abs(lo - 0.8843) < 1e-3
    lo, hi = binomial_ci(3, 30)
    assert 0.02 < lo < 0.03 and 0.26 < hi < 0.27  # Clopper-Pearson 3/30: (0.0211, 0.2653)
    with pytest.raises(ValueError):
        binomial_ci(5, 3)


def test_paired_bootstrap_ci_and_summary():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 40)
    b = a - 0.5  # constant paired difference
    r = paired_bootstrap_ci(a, b, n_boot=2000, seed=1)
    assert abs(r["estimate"] - 0.5) < 1e-12 and r["ci_low"] <= 0.5 <= r["ci_high"]
    assert r["ci_high"] - r["ci_low"] < 1e-9  # zero-variance differences -> degenerate CI
    r2 = paired_bootstrap_ci(a, b, n_boot=2000, seed=1)
    assert r == r2  # deterministic
    s = summary_stats([1, 2, 3, 4, 100])
    assert s["median"] == 3 and s["min"] == 1 and s["max"] == 100 and math.isclose(s["iqr"], 2.0)
