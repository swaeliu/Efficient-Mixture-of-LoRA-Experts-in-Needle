"""Phase 2 — forward correctness of DenseMoLoRALinear."""
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from molora import DenseMoLoRALinear, LoRALinear


def _np_softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _t(a, dtype="float32", requires_grad=False):
    return ndl.Tensor(np.asarray(a).astype(np.dtype(dtype)), dtype=dtype, requires_grad=requires_grad)


def _reference(m: DenseMoLoRALinear, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent float64 NumPy oracle: per-expert outputs, softmax gates, weighted sum."""
    x = x.astype(np.float64)
    W0 = m.weight.numpy().astype(np.float64)
    y = x @ W0
    if m.bias is not None:
        y = y + m.bias.numpy().astype(np.float64)
    logits = (x @ m.router.weight.numpy().astype(np.float64) + m.router.bias.numpy().astype(np.float64)) / m.router_temperature
    g = _np_softmax(logits)  # (..., E)
    for i, e in enumerate(m.experts):
        dy = (m.alpha / m.rank) * (x @ e.lora_A.numpy().astype(np.float64) @ e.lora_B.numpy().astype(np.float64))
        y = y + g[..., i : i + 1] * dy
    return y, g


def _randomise(m: DenseMoLoRALinear, seed=0):
    """Deterministic non-zero experts and router so every path is exercised."""
    rng = np.random.default_rng(seed)
    for i in range(m.n_experts):
        m.load_expert(i, A=rng.standard_normal((m.in_features, m.rank)), B=rng.standard_normal((m.rank, m.out_features)))
    m.load_router(weight=rng.standard_normal((m.in_features, m.n_experts)), bias=rng.standard_normal(m.n_experts))
    return rng


# --------------------------------------------------------------------------- #
# shapes and routing probabilities
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch_shape", [(1,), (5,), (2, 3), (2, 3, 4)])
def test_output_and_gate_shapes(batch_shape):
    m = DenseMoLoRALinear(6, 4, n_experts=3, rank=2)
    y, g = m.forward_with_routing(_t(np.random.default_rng(0).standard_normal(batch_shape + (6,))))
    assert y.shape == batch_shape + (4,)
    assert g.shape == batch_shape + (3,)
    assert m(_t(np.zeros(batch_shape + (6,)))).shape == batch_shape + (4,)


def test_routing_probabilities_sum_to_one_and_are_finite():
    m = DenseMoLoRALinear(6, 4, n_experts=4, rank=2)
    _randomise(m)
    x = np.random.default_rng(1).standard_normal((2, 3, 6)) * 20  # large inputs -> large logits
    p = m.router_probabilities(_t(x))
    assert p.shape == (2, 3, 4)
    assert np.all(np.isfinite(p)) and np.all(p >= 0) and np.all(p <= 1)
    np.testing.assert_allclose(p.sum(-1), 1.0, atol=1e-5)  # float32: E * eps_f32 rounding


# --------------------------------------------------------------------------- #
# initial equivalence to the frozen backbone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_shape", [(7,), (2, 3)])
def test_initial_output_equals_frozen_backbone(bias, batch_shape):
    m = DenseMoLoRALinear(6, 4, n_experts=4, rank=2, bias=bias)
    x = np.random.default_rng(2).standard_normal(batch_shape + (6,)).astype(np.float32)
    base = x @ m.weight.numpy() + (m.bias.numpy() if bias else 0)
    np.testing.assert_allclose(m(_t(x)).numpy(), base, rtol=1e-6, atol=1e-6)
    for e in m.experts:
        np.testing.assert_array_equal(e.lora_B.numpy(), 0.0)
        assert np.any(e.lora_A.numpy() != 0.0)


@pytest.mark.parametrize("bias", [True, False])
def test_from_linear_preserves_base_exactly_and_shares_storage(bias):
    lin = nn.Linear(9, 5, bias=bias)
    m = DenseMoLoRALinear.from_linear(lin, n_experts=3, rank=2, alpha=4.0, router_temperature=2.0)
    assert (m.in_features, m.out_features, m.n_experts, m.rank, m.scaling, m.router_temperature) == (9, 5, 3, 2, 2.0, 2.0)
    assert np.shares_memory(m.weight.cached_data, lin.weight.cached_data)
    if bias:
        assert np.shares_memory(m.bias.cached_data, lin.bias.cached_data)
    else:
        assert m.bias is None
    x = _t(np.random.default_rng(3).standard_normal((4, 9)))
    np.testing.assert_allclose(m(x).numpy(), lin(x).numpy(), rtol=1e-6, atol=1e-7)
    assert lin.weight.requires_grad is True and m.weight.requires_grad is False


def test_from_linear_rejects_bad_inputs():
    with pytest.raises(TypeError):
        DenseMoLoRALinear.from_linear(nn.ReLU())
    with pytest.raises(TypeError):
        DenseMoLoRALinear.from_linear(nn.Linear(4, 3), in_features=4)


# --------------------------------------------------------------------------- #
# independent NumPy reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_shape", [(5,), (2, 3)])
@pytest.mark.parametrize("n_experts,rank", [(1, 2), (3, 1), (4, 2), (5, 3)])
def test_matches_numpy_reference(bias, batch_shape, n_experts, rank):
    m = DenseMoLoRALinear(6, 4, n_experts=n_experts, rank=rank, alpha=2.5 * rank, bias=bias, router_temperature=0.7,
                          dtype="float64")
    rng = _randomise(m, seed=n_experts * 10 + rank)
    x = rng.standard_normal(batch_shape + (6,))
    y, g = m.forward_with_routing(_t(x, dtype="float64"))
    y_ref, g_ref = _reference(m, x)
    np.testing.assert_allclose(g.numpy(), g_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y.numpy(), y_ref, rtol=1e-11, atol=1e-11)


def test_matches_numpy_reference_float32():
    m = DenseMoLoRALinear(16, 8, n_experts=4, rank=4, alpha=8.0)
    rng = _randomise(m, seed=9)
    x = rng.standard_normal((3, 16)).astype(np.float32)
    y, _ = m.forward_with_routing(_t(x))
    y_ref, _ = _reference(m, x)
    np.testing.assert_allclose(y.numpy(), y_ref, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# one-expert equivalence with LoRALinear
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bias", [True, False])
def test_single_expert_equals_lora_linear(bias):
    lin = nn.Linear(6, 4, bias=bias)
    lora = LoRALinear.from_linear(lin, rank=2, alpha=5.0)
    mo = DenseMoLoRALinear.from_linear(lin, n_experts=1, rank=2, alpha=5.0)
    rng = np.random.default_rng(4)
    A, B = rng.standard_normal((6, 2)), rng.standard_normal((2, 4))
    lora.load_adapter(A=A, B=B)
    mo.load_expert(0, A=A, B=B)
    x = _t(rng.standard_normal((2, 3, 6)))
    y, g = mo.forward_with_routing(x)
    np.testing.assert_array_equal(g.numpy(), 1.0)  # softmax over a single expert is exactly 1
    np.testing.assert_allclose(y.numpy(), lora(x).numpy(), rtol=1e-6, atol=1e-6)
    assert mo.num_adapter_parameters() == lora.num_adapter_parameters()


# --------------------------------------------------------------------------- #
# ranks, expert counts, accounting, temperature
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_experts", [1, 2, 4, 8])
@pytest.mark.parametrize("rank", [1, 2, 4])
def test_parameter_accounting(n_experts, rank):
    m = DenseMoLoRALinear(16, 8, n_experts=n_experts, rank=rank, bias=True)
    adapter, router, base = n_experts * rank * (16 + 8), 16 * n_experts + n_experts, 16 * 8 + 8
    assert m.num_adapter_parameters() == adapter == sum(int(np.prod(p.shape)) for p in m.adapter_parameters())
    assert m.num_router_parameters() == router == sum(int(np.prod(p.shape)) for p in m.router_parameters())
    assert m.num_base_parameters() == base
    assert m.num_trainable_parameters() == adapter + router
    assert len(m.trainable_parameters()) == 2 * n_experts + 2
    assert m.num_trainable_parameters() == sum(int(np.prod(p.shape)) for p in m.trainable_parameters())
    assert m.num_total_parameters() == base + adapter + router == sum(int(np.prod(p.shape)) for p in m.parameters())
    assert len(m.parameters()) == 2 + 2 + 2 * n_experts
    assert m.trainable_ratio() == pytest.approx((adapter + router) / base)


def test_default_configuration_counts():
    m = DenseMoLoRALinear(128, 128, n_experts=4, rank=8, alpha=16.0, bias=False)
    s = m.parameter_summary()
    assert s["adapter_parameters"] == 8192
    assert s["router_parameters"] == 516
    assert s["trainable_parameters"] == 8708
    assert s["base_parameters"] == 16384
    assert s["total_parameters"] == 16384 + 8708


def test_temperature_changes_gates_but_not_experts():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, 6))
    outs = {}
    for T in (0.5, 1.0, 4.0):
        m = DenseMoLoRALinear(6, 4, n_experts=3, rank=2, router_temperature=T, dtype="float64")
        _randomise(m, seed=11)
        y, g = m.forward_with_routing(_t(x, dtype="float64"))
        outs[T] = (y.numpy(), g.numpy())
    ent = {T: -(g * np.log(g)).sum(-1).mean() for T, (_, g) in outs.items()}
    assert ent[0.5] < ent[1.0] < ent[4.0]
    assert not np.allclose(outs[0.5][0], outs[4.0][0])


def test_invalid_input_width_raises():
    m = DenseMoLoRALinear(6, 4, n_experts=2, rank=2)
    with pytest.raises(ValueError):
        m(_t(np.zeros((3, 5))))
    with pytest.raises(ValueError):
        m(_t(np.zeros(6)))


@pytest.mark.parametrize("kwargs, exc", [
    (dict(in_features=6, out_features=4, n_experts=0, rank=2), ValueError),
    (dict(in_features=6, out_features=4, n_experts=-1, rank=2), ValueError),
    (dict(in_features=6, out_features=4, n_experts=2.0, rank=2), TypeError),
    (dict(in_features=6, out_features=4, n_experts=2, rank=0), ValueError),
    (dict(in_features=6, out_features=4, n_experts=2, rank=5), ValueError),
    (dict(in_features=6, out_features=4, n_experts=2, rank=2, alpha=0.0), ValueError),
    (dict(in_features=6, out_features=4, n_experts=2, rank=2, router_temperature=0.0), ValueError),
    (dict(in_features=6, out_features=4, n_experts=2, rank=2, router_temperature=-2.0), ValueError),
    (dict(in_features=6, out_features=4, n_experts=2, rank=2, router_init="bad"), ValueError),
    (dict(in_features=6, out_features=4, n_experts=2, rank=2, b_init_std=-0.1), ValueError),
    (dict(in_features=0, out_features=4, n_experts=2, rank=2), ValueError),
    (dict(in_features=6, out_features=4, n_experts=2, rank=2, bias=False, base_bias=np.zeros(4)), ValueError),
])
def test_invalid_configuration_raises(kwargs, exc):
    with pytest.raises(exc):
        DenseMoLoRALinear(**kwargs)


def test_nonzero_b_init_variant_is_not_base_equivalent_but_small():
    """Experimental symmetry-breaking init: documented to break exact base equivalence."""
    m = DenseMoLoRALinear(6, 4, n_experts=2, rank=2, b_init_std=1e-3, bias=False)
    x = np.random.default_rng(6).standard_normal((5, 6)).astype(np.float32)
    diff = np.abs(m(_t(x)).numpy() - x @ m.weight.numpy()).max()
    assert 0 < diff < 1e-1
