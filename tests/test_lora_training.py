"""Phase 1 — frozen-parameter guarantees and optimizer behaviour."""
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from molora import LoRALinear, count_parameters, freeze_module, frozen_parameters, trainable_parameters


def _data(n=16, in_f=6, out_f=4, seed=0):
    rng = np.random.default_rng(seed)
    x = ndl.Tensor(rng.standard_normal((n, in_f)).astype(np.float32), dtype="float32", requires_grad=False)
    y = ndl.Tensor(rng.standard_normal((n, out_f)).astype(np.float32), dtype="float32", requires_grad=False)
    return x, y


def _mse(pred, target):
    n = int(np.prod(pred.shape))
    return ((pred - target) ** 2).sum() / n


@pytest.mark.parametrize("opt_name,lr", [("Adam", 1e-2), ("SGD", 1e-1)])
@pytest.mark.parametrize("bias", [True, False])
def test_only_adapters_change_over_several_steps(opt_name, lr, bias):
    layer = LoRALinear(6, 4, rank=2, alpha=4.0, bias=bias)
    x, y = _data()
    W0 = layer.weight.numpy().copy()
    b0 = None if layer.bias is None else layer.bias.numpy().copy()
    A0, B0 = layer.lora_A.numpy().copy(), layer.lora_B.numpy().copy()

    opt = getattr(ndl.optim, opt_name)(layer.adapter_parameters(), lr=lr)
    losses = []
    for _ in range(5):
        opt.reset_grad()
        loss = _mse(layer(x), y)
        loss.backward()
        assert np.isfinite(loss.numpy())
        for p in layer.adapter_parameters():
            assert p.grad is not None and np.all(np.isfinite(p.grad.numpy()))
        opt.step()
        losses.append(float(loss.numpy()))
        # frozen base identical (bitwise) after every step
        np.testing.assert_array_equal(layer.weight.numpy(), W0)
        if bias:
            np.testing.assert_array_equal(layer.bias.numpy(), b0)

    assert not np.array_equal(layer.lora_B.numpy(), B0)  # B moves on step 1 already
    assert not np.array_equal(layer.lora_A.numpy(), A0)  # A moves once B != 0
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_optimizer_receives_exactly_the_adapter_parameters():
    layer = LoRALinear(10, 7, rank=3)
    params = layer.adapter_parameters()
    opt = ndl.optim.Adam(params, lr=1e-3)
    assert [id(p) for p in opt.params] == [id(layer.lora_A), id(layer.lora_B)]
    assert count_parameters(opt.params) == layer.num_adapter_parameters() == 3 * (10 + 7)
    assert [id(p) for p in trainable_parameters(layer)] == [id(p) for p in params]
    assert [id(p) for p in layer.trainable_parameters()] == [id(p) for p in params]
    assert {id(p) for p in frozen_parameters(layer)} == {id(layer.weight), id(layer.bias)}


def test_parameter_counts_match_tensor_shapes_and_formula():
    for in_f, out_f, r, bias in [(128, 128, 8, False), (64, 32, 4, True), (3, 5, 2, True)]:
        layer = LoRALinear(in_f, out_f, rank=r, bias=bias)
        base = in_f * out_f + (out_f if bias else 0)
        adapter = r * (in_f + out_f)
        assert layer.num_base_parameters() == base == sum(int(np.prod(p.shape)) for p in layer.base_parameters())
        assert layer.num_adapter_parameters() == adapter == sum(int(np.prod(p.shape)) for p in layer.adapter_parameters())
        assert layer.num_total_parameters() == base + adapter == sum(int(np.prod(p.shape)) for p in layer.parameters())
        assert layer.trainable_ratio() == pytest.approx(adapter / base)
        s = layer.parameter_summary()
        assert (s["base_parameters"], s["adapter_parameters"], s["total_parameters"]) == (base, adapter, base + adapter)


def test_passing_all_parameters_to_optimizer_would_update_base_when_input_requires_grad():
    """Characterises the Needle pitfall the design guards against (see module docstring in lora.py)."""
    layer = LoRALinear(6, 4, rank=2)
    x, y = _data()
    x_req = ndl.Tensor(x.numpy(), dtype="float32", requires_grad=True)
    W0 = layer.weight.numpy().copy()
    opt = ndl.optim.SGD(layer.parameters(), lr=0.1)  # WRONG usage
    _mse(layer(x_req), y).backward()
    opt.step()
    assert not np.array_equal(layer.weight.numpy(), W0)
    # correct usage keeps it frozen in the same setting
    layer2 = LoRALinear(6, 4, rank=2)
    W0 = layer2.weight.numpy().copy()
    opt2 = ndl.optim.SGD(layer2.adapter_parameters(), lr=0.1)
    _mse(layer2(x_req), y).backward()
    opt2.step()
    np.testing.assert_array_equal(layer2.weight.numpy(), W0)


def test_from_linear_training_leaves_original_linear_untouched():
    lin = nn.Linear(6, 4)
    W_orig, b_orig = lin.weight.numpy().copy(), lin.bias.numpy().copy()
    layer = LoRALinear.from_linear(lin, rank=2)
    x, y = _data(seed=3)
    opt = ndl.optim.Adam(layer.adapter_parameters(), lr=1e-2)
    for _ in range(3):
        opt.reset_grad()
        _mse(layer(x), y).backward()
        opt.step()
    np.testing.assert_array_equal(lin.weight.numpy(), W_orig)
    np.testing.assert_array_equal(lin.bias.numpy(), b_orig)
    np.testing.assert_array_equal(layer.weight.numpy(), W_orig)
    assert not np.array_equal(layer.lora_B.numpy(), 0.0)


def test_freeze_module_and_trainable_parameters_on_mixed_model():
    """A frozen backbone Linear followed by a LoRA layer: only the adapter is trainable."""
    backbone = freeze_module(nn.Linear(6, 6))
    lora = LoRALinear(6, 4, rank=2)
    model = nn.Sequential(backbone, nn.ReLU(), lora)
    assert len(model.parameters()) == 6
    train = trainable_parameters(model)
    assert [id(p) for p in train] == [id(lora.lora_A), id(lora.lora_B)]
    x, y = _data(seed=4)
    W_bb = backbone.weight.numpy().copy()
    opt = ndl.optim.Adam(train, lr=1e-2)
    for _ in range(3):
        opt.reset_grad()
        _mse(model(x), y).backward()
        opt.step()
    np.testing.assert_array_equal(backbone.weight.numpy(), W_bb)
    np.testing.assert_array_equal(lora.weight.numpy(), lora.weight.numpy())  # trivially; checks finite
    assert not np.array_equal(lora.lora_B.numpy(), 0.0)


def test_training_is_deterministic_under_fixed_seed():
    def run():
        np.random.seed(123)
        layer = LoRALinear(6, 4, rank=2)
        x, y = _data(seed=5)
        opt = ndl.optim.Adam(layer.adapter_parameters(), lr=1e-2)
        for _ in range(4):
            opt.reset_grad()
            _mse(layer(x), y).backward()
            opt.step()
        return layer.lora_A.numpy().copy(), layer.lora_B.numpy().copy()

    A1, B1 = run()
    A2, B2 = run()
    np.testing.assert_array_equal(A1, A2)
    np.testing.assert_array_equal(B1, B2)
