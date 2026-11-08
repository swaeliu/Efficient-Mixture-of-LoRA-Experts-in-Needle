"""Phase 3B — gather/scatter operators and reference-vs-dispatch equivalence."""
import numpy as np
import pytest

import needle as ndl
from molora import DenseMoLoRALinear, SparseMoLoRALinear
from molora.ops import gather_rows, scatter_add_rows


def _t(a, dtype="float64", requires_grad=False):
    return ndl.Tensor(np.asarray(a), dtype=dtype, requires_grad=requires_grad)


# --------------------------------------------------------------------------- #
# gather_rows / scatter_add_rows
# --------------------------------------------------------------------------- #
def test_gather_rows_forward_and_backward_with_repeated_indices():
    x_np = np.arange(12, dtype=np.float64).reshape(4, 3)
    idx = np.array([2, 0, 2, 2, 1])
    x = _t(x_np, requires_grad=True)
    g = gather_rows(x, idx)
    np.testing.assert_array_equal(g.numpy(), x_np[idx])
    up = np.arange(15, dtype=np.float64).reshape(5, 3) + 1
    (g * _t(up)).sum().backward()
    expected = np.zeros_like(x_np)
    np.add.at(expected, idx, up)  # repeated rows accumulate
    np.testing.assert_array_equal(x.grad.numpy(), expected)
    np.testing.assert_array_equal(x.grad.numpy()[3], 0.0)  # row never gathered gets zero


def test_scatter_add_rows_forward_multi_input_and_backward():
    y1, y2 = np.ones((3, 2)), 2 * np.ones((2, 2))
    i1, i2 = np.array([0, 0, 4]), np.array([4, 1])
    t1, t2 = _t(y1, requires_grad=True), _t(y2, requires_grad=True)
    out = scatter_add_rows([t1, t2], [i1, i2], 5)
    expected = np.zeros((5, 2))
    np.add.at(expected, i1, y1)
    np.add.at(expected, i2, y2)
    np.testing.assert_array_equal(out.numpy(), expected)
    np.testing.assert_array_equal(out.numpy()[0], 2.0)  # repeated destination accumulates
    np.testing.assert_array_equal(out.numpy()[2], 0.0)  # untouched row is zero
    up = np.arange(10, dtype=np.float64).reshape(5, 2)
    (out * _t(up)).sum().backward()
    np.testing.assert_array_equal(t1.grad.numpy(), up[i1])  # gradient gathered at destinations
    np.testing.assert_array_equal(t2.grad.numpy(), up[i2])


def test_gather_and_scatter_are_adjoint():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((6, 3)), rng.standard_normal((4, 3))
    idx = np.array([5, 0, 0, 3])
    lhs = np.sum(gather_rows(_t(x), idx).numpy() * y)
    rhs = np.sum(x * scatter_add_rows([_t(y)], [idx], 6).numpy())
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_gather_scatter_roundtrip_gradient_chain():
    """scatter(gather(x)) with a permutation is the identity; its gradient must be too."""
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((5, 2))
    perm = rng.permutation(5)
    x = _t(x_np, requires_grad=True)
    out = scatter_add_rows([gather_rows(x, perm)], [perm], 5)
    np.testing.assert_allclose(out.numpy(), x_np)
    up = rng.standard_normal((5, 2))
    (out * _t(up)).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), up)


def test_empty_gather_and_scatter_input_errors():
    x = _t(np.zeros((3, 2)))
    with pytest.raises(IndexError):
        gather_rows(x, np.array([3]))
    with pytest.raises(ValueError):
        gather_rows(x, np.array([[0, 1]]))
    with pytest.raises(TypeError):
        gather_rows(x, np.array([0.5]))
    with pytest.raises(ValueError):
        scatter_add_rows([x], [np.array([0, 1])], 3)  # 3 rows but 2 indices
    with pytest.raises(IndexError):
        scatter_add_rows([x], [np.array([0, 1, 3])], 3)
    with pytest.raises(ValueError):
        scatter_add_rows([x], [np.array([0, 1, 2]), np.array([0])], 3)


def test_scatter_float32_dtype_preserved():
    out = scatter_add_rows([_t(np.ones((2, 2), np.float32), dtype="float32")], [np.array([1, 1])], 3)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out.numpy()[1], 2.0)


# --------------------------------------------------------------------------- #
# reference vs dispatch equivalence
# --------------------------------------------------------------------------- #
def _pair(E=4, k=2, in_f=6, out_f=4, rank=2, bias=True, seed=0, mode="renormalized", dtype="float64"):
    rng = np.random.default_rng(seed)
    ref = SparseMoLoRALinear(in_f, out_f, n_experts=E, rank=rank, alpha=3.0, bias=bias, top_k=k, router_temperature=0.8,
                             execution_mode="reference", router_grad_mode=mode, dtype=dtype)
    dis = SparseMoLoRALinear(in_f, out_f, n_experts=E, rank=rank, alpha=3.0, bias=bias, top_k=k, router_temperature=0.8,
                             execution_mode="dispatch", router_grad_mode=mode, dtype=dtype,
                             base_weight=ref.weight, base_bias=ref.bias)
    for i in range(E):
        A, B = rng.standard_normal((in_f, rank)), rng.standard_normal((rank, out_f))
        ref.load_expert(i, A=A, B=B)
        dis.load_expert(i, A=A, B=B)
    W, b = rng.standard_normal((in_f, E)), rng.standard_normal(E)
    ref.load_router(weight=W, bias=b)
    dis.load_router(weight=W, bias=b)
    return ref, dis, rng


def _grads(m, x, target, dtype="float64"):
    xt = _t(x, dtype=dtype, requires_grad=True)
    loss = ((m(xt) - _t(target, dtype=dtype)) ** 2).sum()
    loss.backward()
    grads = [p.grad.numpy() if p.grad is not None else None for p in m.trainable_parameters()]
    return float(loss.numpy()), grads, xt.grad.numpy()


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_shape", [(7,), (2, 3), (2, 3, 2)])
@pytest.mark.parametrize("E,k", [(4, 1), (4, 2), (4, 3), (4, 4), (3, 2)])
def test_dispatch_matches_reference_forward_and_all_gradients(bias, batch_shape, E, k):
    ref, dis, rng = _pair(E=E, k=k, bias=bias, seed=E * 10 + k)
    x = rng.standard_normal(batch_shape + (6,))
    target = rng.standard_normal(batch_shape + (4,))
    y_ref, w_ref = ref.forward_with_routing(_t(x))
    y_dis, w_dis = dis.forward_with_routing(_t(x))
    np.testing.assert_allclose(y_dis.numpy(), y_ref.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(w_dis.numpy(), w_ref.numpy())
    l_ref, g_ref, xg_ref = _grads(ref, x, target)
    l_dis, g_dis, xg_dis = _grads(dis, x, target)
    np.testing.assert_allclose(l_dis, l_ref, rtol=1e-12)
    for a, b in zip(g_ref, g_dis):
        if a is None or b is None:
            assert a is None and b is None  # k == 1 exact: router has no gradient in either mode
            continue
        np.testing.assert_allclose(b, a, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(xg_dis, xg_ref, rtol=1e-11, atol=1e-11)
    # bookkeeping
    assert dis.last_stats["expert_flop_fraction_vs_dense"] == pytest.approx(k / E)
    assert ref.last_stats["expert_flop_fraction_vs_dense"] == 1.0
    assert sum(dis.last_stats["tokens_per_expert"]) == k * int(np.prod(batch_shape))


def test_dispatch_matches_reference_straight_through_top1():
    ref, dis, rng = _pair(E=4, k=1, mode="straight_through", seed=5)
    x, target = rng.standard_normal((9, 6)), rng.standard_normal((9, 4))
    l_ref, g_ref, xg_ref = _grads(ref, x, target)
    l_dis, g_dis, xg_dis = _grads(dis, x, target)
    np.testing.assert_allclose(l_dis, l_ref, rtol=1e-12)
    for a, b in zip(g_ref, g_dis):
        np.testing.assert_allclose(b, a, rtol=1e-11, atol=1e-11)
    assert np.linalg.norm(g_dis[-2]) > 1e-6  # router weight gradient is live under the surrogate
    np.testing.assert_allclose(xg_dis, xg_ref, rtol=1e-11, atol=1e-11)


def test_identical_optimizer_updates_in_both_modes():
    ref, dis, rng = _pair(E=4, k=2, seed=6)
    x, target = rng.standard_normal((16, 6)), rng.standard_normal((16, 4))
    xt, tt = _t(x), _t(target)
    opts = [ndl.optim.Adam(m.trainable_parameters(), lr=1e-2) for m in (ref, dis)]
    for _ in range(5):
        losses = []
        for m, opt in zip((ref, dis), opts):
            opt.reset_grad()
            loss = ((m(xt) - tt) ** 2).sum() / 64
            loss.backward()
            opt.step()
            losses.append(float(loss.numpy()))
        np.testing.assert_allclose(losses[1], losses[0], rtol=1e-12)
    for a, b in zip(ref.trainable_parameters(), dis.trainable_parameters()):
        np.testing.assert_allclose(b.numpy(), a.numpy(), rtol=1e-11, atol=1e-11)


def test_switching_execution_mode_on_one_model_is_consistent():
    ref, _, rng = _pair(E=4, k=2, seed=7)
    x = _t(rng.standard_normal((2, 4, 6)))
    y_ref = ref(x).numpy()
    ref.execution_mode = "dispatch"
    y_dis = ref(x).numpy()
    np.testing.assert_allclose(y_dis, y_ref, rtol=1e-12, atol=1e-12)
    assert ref.last_stats["execution_mode"] == "dispatch"


def test_k_equals_E_dispatch_matches_dense_molora():
    dense = DenseMoLoRALinear(6, 4, n_experts=4, rank=2, alpha=4.0, router_temperature=0.9, dtype="float64")
    sp = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, alpha=4.0, top_k=4, router_temperature=0.9, dtype="float64",
                            execution_mode="dispatch", base_weight=dense.weight, base_bias=dense.bias)
    rng = np.random.default_rng(8)
    for i in range(4):
        A, B = rng.standard_normal((6, 2)), rng.standard_normal((2, 4))
        dense.load_expert(i, A=A, B=B)
        sp.load_expert(i, A=A, B=B)
    W, b = rng.standard_normal((6, 4)), rng.standard_normal(4)
    dense.load_router(weight=W, bias=b)
    sp.load_router(weight=W, bias=b)
    x, target = rng.standard_normal((2, 3, 6)), rng.standard_normal((2, 3, 4))
    yd = dense(_t(x)).numpy()
    ys = sp(_t(x)).numpy()
    np.testing.assert_allclose(ys, yd, rtol=1e-12, atol=1e-12)
    ld, gd, xgd = _grads(dense, x, target)
    ls, gs, xgs = _grads(sp, x, target)
    for a, b in zip(gd, gs):
        np.testing.assert_allclose(b, a, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(xgs, xgd, rtol=1e-11, atol=1e-11)


# --------------------------------------------------------------------------- #
# actual expert skipping
# --------------------------------------------------------------------------- #
def test_inactive_expert_is_skipped_and_accounted():
    ref, dis, rng = _pair(E=4, k=2, seed=9)
    b = dis.router.bias.numpy().copy()
    b[1] = -100.0  # expert 2 can never be selected
    dis.load_router(bias=b)
    ref.load_router(bias=b)
    x, target = rng.standard_normal((20, 6)), rng.standard_normal((20, 4))
    _grads(dis, x, target)
    st = dis.last_stats
    assert st["tokens_per_expert"][1] == 0 and st["expert_rows_executed"][1] == 0
    assert st["experts_skipped"] == 1 and st["expert_matmuls_executed"] == 6  # 3 active experts x 2 matmuls
    assert st["expert_flop_fraction_vs_dense"] == pytest.approx(0.5)  # k/E, independent of balance
    assert sum(st["expert_rows_executed"]) == 40
    # the skipped expert never enters the graph in dispatch mode: its gradient stays None (reference
    # mode produces an explicit zero array instead); outputs still match the reference
    assert dis.experts[1].lora_A.grad is None and dis.experts[1].lora_B.grad is None
    _grads(ref, x, target)
    np.testing.assert_array_equal(ref.experts[1].lora_A.grad.numpy(), 0.0)
    np.testing.assert_array_equal(ref.experts[1].lora_B.grad.numpy(), 0.0)
    np.testing.assert_allclose(dis(_t(x)).numpy(), ref(_t(x)).numpy(), rtol=1e-12, atol=1e-12)


def test_top1_statistics_and_imbalanced_load():
    _, dis, rng = _pair(E=4, k=1, seed=10)
    x = rng.standard_normal((32, 6))
    dis(_t(x))
    st = dis.last_stats
    assert sum(st["tokens_per_expert"]) == 32 and st["selected_experts_per_token"] == 1
    assert st["expert_flop_fraction_vs_dense"] == pytest.approx(0.25) == st["theoretical_flop_fraction"]
    assert st["expert_matmuls_executed"] == 2 * sum(1 for r in st["expert_rows_executed"] if r > 0)
    assert st["expert_flops_executed"] * 4 == st["expert_flops_dense"]


def test_profile_timing_is_recorded_when_enabled():
    _, dis, rng = _pair(E=4, k=2, seed=11)
    dis.profile = True
    dis(_t(rng.standard_normal((8, 6))))
    t = dis.last_timing
    assert set(t) >= {"backbone_s", "router_s", "expert_s", "dispatch_s", "forward_total_s"}
    assert all(v >= 0 for v in t.values()) and t["forward_total_s"] >= t["expert_s"]
    dis.profile = False
    dis(_t(rng.standard_normal((8, 6))))
