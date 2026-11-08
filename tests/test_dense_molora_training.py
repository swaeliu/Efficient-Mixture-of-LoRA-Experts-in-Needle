"""Phase 2 — frozen-backbone guarantees and optimizer behaviour of DenseMoLoRALinear."""
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from molora import DenseMoLoRALinear, trainable_parameters


def _data(n=32, in_f=6, out_f=4, seed=0):
    rng = np.random.default_rng(seed)
    x = ndl.Tensor(rng.standard_normal((n, in_f)).astype(np.float32), dtype="float32", requires_grad=False)
    y = ndl.Tensor(rng.standard_normal((n, out_f)).astype(np.float32), dtype="float32", requires_grad=False)
    return x, y


def _mse(pred, target):
    return ((pred - target) ** 2).sum() / int(np.prod(pred.shape))


def _snapshot(m):
    return {
        "W0": m.weight.numpy().copy(),
        "b": None if m.bias is None else m.bias.numpy().copy(),
        "A": [e.lora_A.numpy().copy() for e in m.experts],
        "B": [e.lora_B.numpy().copy() for e in m.experts],
        "Wg": m.router.weight.numpy().copy(),
        "bg": m.router.bias.numpy().copy(),
    }


@pytest.mark.parametrize("opt_name,lr", [("Adam", 1e-2), ("SGD", 1e-1)])
@pytest.mark.parametrize("bias", [True, False])
def test_backbone_frozen_experts_and_router_update(opt_name, lr, bias):
    m = DenseMoLoRALinear(6, 4, n_experts=3, rank=2, alpha=4.0, bias=bias)
    x, y = _data()
    s0 = _snapshot(m)
    params = m.trainable_parameters()
    opt = getattr(ndl.optim, opt_name)(params, lr=lr)
    assert all(id(p) not in {id(q) for q in m.base_parameters()} for p in opt.params)
    assert len({id(p) for p in opt.params}) == len(opt.params) == 2 * 3 + 2

    losses = []
    for step in range(6):
        opt.reset_grad()
        loss = _mse(m(x), y)
        loss.backward()
        assert np.isfinite(loss.numpy())
        for p in params:
            assert p.grad is not None and np.all(np.isfinite(p.grad.numpy()))
        if step == 0:  # zero-B init: router gets no signal on the very first step
            np.testing.assert_array_equal(m.router.weight.grad.numpy(), 0.0)
        opt.step()
        losses.append(float(loss.numpy()))
        np.testing.assert_array_equal(m.weight.numpy(), s0["W0"])
        if bias:
            np.testing.assert_array_equal(m.bias.numpy(), s0["b"])

    for i, e in enumerate(m.experts):
        assert not np.array_equal(e.lora_A.numpy(), s0["A"][i])
        assert not np.array_equal(e.lora_B.numpy(), s0["B"][i])
    # once expert outputs differ (after step 1) the router receives gradient and moves
    assert not np.array_equal(m.router.weight.numpy(), s0["Wg"])
    assert not np.array_equal(m.router.bias.numpy(), s0["bg"])
    assert losses[-1] < losses[0]


def test_router_frozen_in_place_when_experts_identical():
    """Characterisation: identical experts -> router gradient 0 -> Adam leaves the router unchanged."""
    m = DenseMoLoRALinear(6, 4, n_experts=3, rank=2, alpha=4.0, dtype="float64")
    rng = np.random.default_rng(1)
    A, B = rng.standard_normal((6, 2)), rng.standard_normal((2, 4))
    for i in range(3):
        m.load_expert(i, A=A, B=B)
    x = ndl.Tensor(rng.standard_normal((16, 6)), dtype="float64", requires_grad=False)
    y = ndl.Tensor(rng.standard_normal((16, 4)), dtype="float64", requires_grad=False)
    Wg0 = m.router.weight.numpy().copy()
    opt = ndl.optim.Adam(m.trainable_parameters(), lr=1e-2)
    _mse(m(x), y).backward()
    assert np.linalg.norm(m.router.weight.grad.numpy()) < 1e-12
    opt.step()
    # Adam normalises by sqrt(v)+eps, so a ~1e-16 gradient could still move the weight by ~lr * g/(|g|+eps).
    # With |g| << eps the update is ~ lr * g / eps ~ 1e-6 at most; assert it is negligible.
    assert np.abs(m.router.weight.numpy() - Wg0).max() < 1e-5


def test_trainable_parameter_list_matches_counts_and_excludes_backbone():
    m = DenseMoLoRALinear(128, 128, n_experts=4, rank=8, alpha=16.0, bias=False)
    params = m.trainable_parameters()
    ids = {id(p) for p in params}
    assert id(m.weight) not in ids
    assert sum(int(np.prod(p.shape)) for p in params) == 8708
    assert {id(p) for p in trainable_parameters(m)} == ids  # generic helper agrees with the explicit list
    assert len(m.frozen_parameters()) == 1


def test_from_linear_training_leaves_original_untouched():
    lin = nn.Linear(6, 4)
    W, b = lin.weight.numpy().copy(), lin.bias.numpy().copy()
    m = DenseMoLoRALinear.from_linear(lin, n_experts=2, rank=2)
    x, y = _data(seed=2)
    opt = ndl.optim.Adam(m.trainable_parameters(), lr=1e-2)
    for _ in range(4):
        opt.reset_grad()
        _mse(m(x), y).backward()
        opt.step()
    np.testing.assert_array_equal(lin.weight.numpy(), W)
    np.testing.assert_array_equal(lin.bias.numpy(), b)
    np.testing.assert_array_equal(m.weight.numpy(), W)


def test_training_in_a_stack_with_gradient_requiring_input():
    """MoLoRA layer after a trainable layer: backbone gets a .grad (Needle) but is never updated."""
    first = nn.Linear(6, 6)
    m = DenseMoLoRALinear(6, 4, n_experts=2, rank=2)
    model = nn.Sequential(first, nn.ReLU(), m)
    x, y = _data(seed=3)
    W0 = m.weight.numpy().copy()
    params = first.parameters() + m.trainable_parameters()
    opt = ndl.optim.Adam(params, lr=1e-2)
    for _ in range(3):
        opt.reset_grad()
        _mse(model(x), y).backward()
        opt.step()
    assert m.weight.grad is not None  # populated by Needle because its input requires grad
    np.testing.assert_array_equal(m.weight.numpy(), W0)  # ... but never handed to the optimizer


def test_deterministic_under_fixed_seed():
    def run():
        np.random.seed(7)
        m = DenseMoLoRALinear(6, 4, n_experts=3, rank=2)
        x, y = _data(seed=4)
        opt = ndl.optim.Adam(m.trainable_parameters(), lr=1e-2)
        for _ in range(4):
            opt.reset_grad()
            _mse(m(x), y).backward()
            opt.step()
        return np.concatenate([p.numpy().ravel() for p in m.trainable_parameters()])

    np.testing.assert_array_equal(run(), run())
