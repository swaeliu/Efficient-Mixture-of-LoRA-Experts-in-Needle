"""Phase 3A — forward correctness of SparseMoLoRALinear (reference execution mode)."""
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from molora import DenseMoLoRALinear, LoRALinear, SparseMoLoRALinear


def _np_softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _t(a, dtype="float32", requires_grad=False):
    return ndl.Tensor(np.asarray(a).astype(np.dtype(dtype)), dtype=dtype, requires_grad=requires_grad)


def np_reference(m: SparseMoLoRALinear, x: np.ndarray):
    """Independent float64 oracle for the sparse mixture; returns (y, sparse weights, mask)."""
    x = x.astype(np.float64)
    lead = x.shape[:-1]
    xf = x.reshape(-1, x.shape[-1])
    W0 = m.weight.numpy().astype(np.float64)
    y = xf @ W0
    if m.bias is not None:
        y = y + m.bias.numpy().astype(np.float64)
    p = _np_softmax((xf @ m.router.weight.numpy().astype(np.float64) + m.router.bias.numpy().astype(np.float64)) / m.router_temperature)
    idx = np.argsort(-p, axis=-1, kind="stable")[:, : m.top_k]
    mask = np.zeros_like(p)
    np.put_along_axis(mask, idx, 1, axis=-1)
    w = p * mask
    w = w / w.sum(-1, keepdims=True)
    for i, e in enumerate(m.experts):
        dy = m.scaling * (xf @ e.lora_A.numpy().astype(np.float64) @ e.lora_B.numpy().astype(np.float64))
        y = y + w[:, i : i + 1] * dy
    return y.reshape(lead + (m.out_features,)), w.reshape(lead + (m.n_experts,)), mask.reshape(lead + (m.n_experts,))


def randomise(m, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(m.n_experts):
        m.load_expert(i, A=rng.standard_normal((m.in_features, m.rank)), B=rng.standard_normal((m.rank, m.out_features)))
    m.load_router(weight=rng.standard_normal((m.in_features, m.n_experts)), bias=rng.standard_normal(m.n_experts))
    return rng


@pytest.mark.parametrize("batch_shape", [(1,), (5,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_output_shapes_and_exactly_k_selected(batch_shape, k):
    m = SparseMoLoRALinear(6, 4, n_experts=3, rank=2, top_k=k)
    randomise(m)
    y, w = m.forward_with_routing(_t(np.random.default_rng(0).standard_normal(batch_shape + (6,))))
    assert y.shape == batch_shape + (4,) and w.shape == batch_shape + (3,)
    wn = w.numpy()
    np.testing.assert_array_equal((wn > 0).sum(-1), k)
    np.testing.assert_allclose(wn.sum(-1), 1.0, atol=1e-6)
    assert m.last_stats["n_tokens"] == int(np.prod(batch_shape)) and sum(m.last_stats["tokens_per_expert"]) == k * int(np.prod(batch_shape))


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_shape", [(7,), (2, 3)])
def test_initial_output_equals_frozen_backbone(bias, batch_shape):
    m = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=2, bias=bias)
    x = np.random.default_rng(1).standard_normal(batch_shape + (6,)).astype(np.float32)
    base = x @ m.weight.numpy() + (m.bias.numpy() if bias else 0)
    np.testing.assert_allclose(m(_t(x)).numpy(), base, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_shape", [(5,), (2, 3)])
@pytest.mark.parametrize("E,k,rank", [(3, 1, 2), (4, 2, 2), (5, 3, 1), (4, 4, 3)])
def test_matches_independent_numpy_reference(bias, batch_shape, E, k, rank):
    m = SparseMoLoRALinear(6, 4, n_experts=E, rank=rank, alpha=2.5 * rank, top_k=k, bias=bias, router_temperature=0.7,
                           dtype="float64")
    rng = randomise(m, seed=E * 10 + k)
    x = rng.standard_normal(batch_shape + (6,))
    y, w = m.forward_with_routing(_t(x, dtype="float64"))
    y_ref, w_ref, mask_ref = np_reference(m, x)
    np.testing.assert_allclose(w.numpy(), w_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y.numpy(), y_ref, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(w.numpy()[mask_ref == 0], 0.0)


def test_k_equals_E_matches_dense_molora():
    dense = DenseMoLoRALinear(6, 4, n_experts=4, rank=2, alpha=4.0, router_temperature=0.9, dtype="float64")
    sparse = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, alpha=4.0, top_k=4, router_temperature=0.9, dtype="float64",
                                base_weight=dense.weight, base_bias=dense.bias)
    rng = randomise(dense, seed=7)
    for i in range(4):
        sparse.load_expert(i, A=dense.experts[i].lora_A.numpy(), B=dense.experts[i].lora_B.numpy())
    sparse.load_router(weight=dense.router.weight.numpy(), bias=dense.router.bias.numpy())
    x = _t(rng.standard_normal((2, 3, 6)), dtype="float64")
    yd, gd = dense.forward_with_routing(x)
    ys, gs = sparse.forward_with_routing(x)
    np.testing.assert_allclose(gs.numpy(), gd.numpy(), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(ys.numpy(), yd.numpy(), rtol=1e-13, atol=1e-13)
    assert sparse.last_stats["theoretical_flop_fraction"] == 1.0


@pytest.mark.parametrize("bias", [True, False])
def test_single_expert_top1_matches_lora_linear(bias):
    lin = nn.Linear(6, 4, bias=bias)
    lora = LoRALinear.from_linear(lin, rank=2, alpha=5.0)
    sp = SparseMoLoRALinear.from_linear(lin, n_experts=1, rank=2, alpha=5.0, top_k=1)
    rng = np.random.default_rng(4)
    A, B = rng.standard_normal((6, 2)), rng.standard_normal((2, 4))
    lora.load_adapter(A=A, B=B)
    sp.load_expert(0, A=A, B=B)
    x = _t(rng.standard_normal((2, 3, 6)))
    y, w = sp.forward_with_routing(x)
    np.testing.assert_array_equal(w.numpy(), 1.0)
    np.testing.assert_allclose(y.numpy(), lora(x).numpy(), rtol=1e-6, atol=1e-6)
    assert sp.num_trainable_parameters() == lora.num_adapter_parameters() + 6 + 1


def test_from_linear_shares_storage_and_inherits_accounting():
    lin = nn.Linear(128, 128, bias=False)
    m = SparseMoLoRALinear.from_linear(lin, n_experts=4, rank=8, alpha=16.0, top_k=2, execution_mode="reference")
    assert np.shares_memory(m.weight.cached_data, lin.weight.cached_data)
    s = m.parameter_summary()
    assert (s["adapter_parameters"], s["router_parameters"], s["trainable_parameters"], s["top_k"]) == (8192, 516, 8708, 2)
    assert len(m.parameters()) == 1 + 2 + 8 and len({id(p) for p in m.parameters()}) == 11
    assert {id(p) for p in m.trainable_parameters()} == {id(p) for p in m.adapter_parameters() + m.router_parameters()}
    assert id(m.weight) not in {id(p) for p in m.trainable_parameters()}
    x = _t(np.random.default_rng(5).standard_normal((3, 128)))
    np.testing.assert_allclose(m(x).numpy(), lin(x).numpy(), rtol=1e-6, atol=1e-6)


def test_routing_diagnostics_consistency():
    m = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=2, dtype="float64")
    rng = randomise(m, seed=8)
    x = rng.standard_normal((2, 5, 6))
    d = m.routing_diagnostics(_t(x, dtype="float64"))
    _, w_ref, mask_ref = np_reference(m, x)
    np.testing.assert_allclose(d["sparse_weights"], w_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(d["mask"], mask_ref)
    assert d["n_tokens"] == 10 and sum(d["tokens_per_expert"]) == 20
    np.testing.assert_allclose(sum(d["utilization"]), 2.0)
    assert 0 <= d["entropy_nats_mean"] <= d["max_entropy_nats"] + 1e-12
    np.testing.assert_allclose(d["dense_probs"].sum(-1), 1.0, atol=1e-12)


def test_extreme_inputs_are_finite():
    m = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=2)
    randomise(m)
    x = _t(np.random.default_rng(9).standard_normal((5, 6)) * 1e3)
    y, w = m.forward_with_routing(x)
    assert np.all(np.isfinite(y.numpy())) and np.all(np.isfinite(w.numpy()))


def test_tied_router_outputs_break_ties_deterministically():
    m = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=2, router_init="zeros")  # uniform probabilities
    x = _t(np.random.default_rng(10).standard_normal((3, 6)))
    _, w = m.forward_with_routing(x)
    np.testing.assert_allclose(w.numpy(), np.tile([0.5, 0.5, 0.0, 0.0], (3, 1)), atol=1e-7)


@pytest.mark.parametrize("kwargs, exc", [
    (dict(top_k=0), ValueError), (dict(top_k=5), ValueError), (dict(top_k=1.0), TypeError),
    (dict(top_k=2, execution_mode="fast"), ValueError), (dict(top_k=2, router_grad_mode="ste"), ValueError),
    (dict(top_k=2, n_experts=0), ValueError), (dict(top_k=2, router_temperature=-1.0), ValueError),
])
def test_invalid_configuration_raises(kwargs, exc):
    base = dict(in_features=6, out_features=4, n_experts=4, rank=2)
    base.update(kwargs)
    with pytest.raises(exc):
        SparseMoLoRALinear(**base)


def test_wrong_input_width_raises():
    m = SparseMoLoRALinear(6, 4, n_experts=2, rank=2, top_k=1)
    with pytest.raises(ValueError):
        m(_t(np.zeros((3, 5))))
    with pytest.raises(ValueError):
        m(_t(np.zeros(6)))
