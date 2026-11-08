"""Phase 3B — training behaviour of SparseMoLoRALinear in both execution modes."""
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from molora import SparseMoLoRALinear
from molora.sparse_molora import load_balance_loss


def _data(n=32, in_f=6, out_f=4, seed=0, dtype="float32"):
    rng = np.random.default_rng(seed)
    x = ndl.Tensor(rng.standard_normal((n, in_f)).astype(dtype), dtype=dtype, requires_grad=False)
    y = ndl.Tensor(rng.standard_normal((n, out_f)).astype(dtype), dtype=dtype, requires_grad=False)
    return x, y


def _mse(pred, target):
    return ((pred - target) ** 2).sum() / int(np.prod(pred.shape))


def _snapshot(m):
    return {"W0": m.weight.numpy().copy(), "b": None if m.bias is None else m.bias.numpy().copy(),
            "A": [e.lora_A.numpy().copy() for e in m.experts], "B": [e.lora_B.numpy().copy() for e in m.experts],
            "Wg": m.router.weight.numpy().copy(), "bg": m.router.bias.numpy().copy()}


@pytest.mark.parametrize("mode", ["reference", "dispatch"])
@pytest.mark.parametrize("opt_name,lr", [("Adam", 1e-2), ("SGD", 1e-1)])
@pytest.mark.parametrize("bias", [True, False])
def test_backbone_frozen_experts_and_router_update_top2(mode, opt_name, lr, bias):
    m = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, alpha=4.0, bias=bias, top_k=2, execution_mode=mode)
    x, y = _data()
    s0 = _snapshot(m)
    params = m.trainable_parameters()
    assert id(m.weight) not in {id(p) for p in params} and len({id(p) for p in params}) == len(params) == 10
    opt = getattr(ndl.optim, opt_name)(params, lr=lr)
    losses = []
    for step in range(8):
        opt.reset_grad()
        loss = _mse(m(x), y)
        loss.backward()
        assert np.isfinite(loss.numpy())
        for p in params:
            if p.grad is not None:
                assert np.all(np.isfinite(p.grad.numpy()))
        opt.step()
        losses.append(float(loss.numpy()))
        np.testing.assert_array_equal(m.weight.numpy(), s0["W0"])
        if bias:
            np.testing.assert_array_equal(m.bias.numpy(), s0["b"])
    assert any(not np.array_equal(e.lora_B.numpy(), s0["B"][i]) for i, e in enumerate(m.experts))
    assert not np.array_equal(m.router.weight.numpy(), s0["Wg"])
    assert not np.array_equal(m.router.bias.numpy(), s0["bg"])
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("mode", ["reference", "dispatch"])
def test_top1_exact_router_never_moves_but_straight_through_does(mode):
    x, y = _data(seed=1)
    exact = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=1, execution_mode=mode, router_grad_mode="renormalized")
    s0 = _snapshot(exact)
    opt = ndl.optim.Adam(exact.trainable_parameters(), lr=1e-2)
    for _ in range(5):
        opt.reset_grad()
        _mse(exact(x), y).backward()
        opt.step()
    np.testing.assert_array_equal(exact.router.weight.numpy(), s0["Wg"])  # grad is None -> optimizer skips it
    np.testing.assert_array_equal(exact.router.bias.numpy(), s0["bg"])
    assert any(not np.array_equal(e.lora_B.numpy(), s0["B"][i]) for i, e in enumerate(exact.experts))

    np.random.seed(0)
    ste = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=1, execution_mode=mode, router_grad_mode="straight_through")
    s1 = _snapshot(ste)
    opt = ndl.optim.Adam(ste.trainable_parameters(), lr=1e-2)
    for _ in range(5):
        opt.reset_grad()
        _mse(ste(x), y).backward()
        opt.step()
    assert not np.array_equal(ste.router.weight.numpy(), s1["Wg"])
    np.testing.assert_array_equal(ste.weight.numpy(), s1["W0"])


def test_modes_give_identical_training_trajectories():
    np.random.seed(3)
    a = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=2, execution_mode="reference", dtype="float64")
    b = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=2, execution_mode="dispatch", dtype="float64",
                           base_weight=a.weight, base_bias=a.bias)
    for i in range(4):
        b.load_expert(i, A=a.experts[i].lora_A.numpy(), B=a.experts[i].lora_B.numpy())
    b.load_router(weight=a.router.weight.numpy(), bias=a.router.bias.numpy())
    x, y = _data(seed=4, dtype="float64")
    oa, ob = (ndl.optim.Adam(m.trainable_parameters(), lr=1e-2) for m in (a, b))
    for _ in range(10):
        la = _mse(a(x), y); oa.reset_grad(); la.backward(); oa.step()
        lb = _mse(b(x), y); ob.reset_grad(); lb.backward(); ob.step()
        np.testing.assert_allclose(float(lb.numpy()), float(la.numpy()), rtol=1e-12)
    for p, q in zip(a.trainable_parameters(), b.trainable_parameters()):
        np.testing.assert_allclose(q.numpy(), p.numpy(), rtol=1e-11, atol=1e-11)


def test_from_linear_training_leaves_original_untouched():
    lin = nn.Linear(6, 4)
    W, bb = lin.weight.numpy().copy(), lin.bias.numpy().copy()
    m = SparseMoLoRALinear.from_linear(lin, n_experts=3, rank=2, top_k=2, execution_mode="dispatch")
    x, y = _data(seed=5)
    opt = ndl.optim.Adam(m.trainable_parameters(), lr=1e-2)
    for _ in range(4):
        opt.reset_grad()
        _mse(m(x), y).backward()
        opt.step()
    np.testing.assert_array_equal(lin.weight.numpy(), W)
    np.testing.assert_array_equal(lin.bias.numpy(), bb)


def test_stacked_model_with_gradient_requiring_input_dispatch():
    first = nn.Linear(6, 6)
    m = SparseMoLoRALinear(6, 4, n_experts=3, rank=2, top_k=2, execution_mode="dispatch")
    model = nn.Sequential(first, nn.ReLU(), m)
    x, y = _data(seed=6)
    W0 = m.weight.numpy().copy()
    opt = ndl.optim.Adam(first.parameters() + m.trainable_parameters(), lr=1e-2)
    for _ in range(3):
        opt.reset_grad()
        _mse(model(x), y).backward()
        opt.step()
    assert first.weight.grad is not None and np.all(np.isfinite(first.weight.grad.numpy()))
    np.testing.assert_array_equal(m.weight.numpy(), W0)


def test_deterministic_under_fixed_seed_dispatch():
    def run():
        np.random.seed(11)
        m = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=2, execution_mode="dispatch")
        x, y = _data(seed=7)
        opt = ndl.optim.Adam(m.trainable_parameters(), lr=1e-2)
        for _ in range(4):
            opt.reset_grad()
            _mse(m(x), y).backward()
            opt.step()
        return np.concatenate([p.numpy().ravel() for p in m.trainable_parameters()])

    np.testing.assert_array_equal(run(), run())


def test_load_balance_loss_value_and_gradient():
    m = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=1, dtype="float64", router_init="zeros")
    x = ndl.Tensor(np.random.default_rng(8).standard_normal((16, 6)), dtype="float64", requires_grad=False)
    sel = m.router.route(x)
    # zero router -> uniform probabilities, ties -> everything goes to expert 0: f = (1,0,0,0), P = 1/4
    lb = load_balance_loss(sel)
    np.testing.assert_allclose(float(lb.numpy()), 4 * (1.0 * 0.25), rtol=1e-12)
    lb.backward()
    assert m.router.weight.grad is not None and np.all(np.isfinite(m.router.weight.grad.numpy()))
    # a perfectly balanced hard assignment with uniform probabilities gives the minimum value 1
    m2 = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, top_k=1, dtype="float64")
    rng = np.random.default_rng(9)
    m2.load_router(weight=rng.standard_normal((6, 4)) * 5, bias=np.zeros(4))
    sel2 = m2.router.route(ndl.Tensor(rng.standard_normal((400, 6)), dtype="float64", requires_grad=False))
    val = float(load_balance_loss(sel2).numpy())
    assert val >= 1.0 - 1e-9
