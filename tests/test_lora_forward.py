"""Phase 1 — forward-pass correctness of LoRALinear."""
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from molora import LoRALinear


def _x(shape, seed=0, dtype=np.float32, requires_grad=False):
    rng = np.random.default_rng(seed)
    return ndl.Tensor(rng.standard_normal(shape).astype(dtype), dtype=str(np.dtype(dtype)), requires_grad=requires_grad)


def _reference(layer: LoRALinear, x_np: np.ndarray) -> np.ndarray:
    """x W0 + b + (alpha / r) x A B computed with numpy in float64."""
    W0 = layer.weight.numpy().astype(np.float64)
    A = layer.lora_A.numpy().astype(np.float64)
    B = layer.lora_B.numpy().astype(np.float64)
    out = x_np.astype(np.float64) @ W0 + layer.scaling * (x_np.astype(np.float64) @ A @ B)
    if layer.bias is not None:
        out = out + layer.bias.numpy().astype(np.float64)
    return out


# --------------------------------------------------------------------------- #
# 1. shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch_shape", [(1,), (5,), (2, 3), (2, 3, 4)])
def test_output_shape_with_arbitrary_leading_dims(batch_shape):
    layer = LoRALinear(6, 4, rank=2)
    x = _x(batch_shape + (6,))
    y = layer(x)
    assert y.shape == batch_shape + (4,)
    assert y.dtype == np.float32


def test_parameter_shapes_follow_needle_orientation():
    layer = LoRALinear(6, 4, rank=3, bias=True)
    assert layer.weight.shape == (6, 4)
    assert layer.bias.shape == (4,)
    assert layer.lora_A.shape == (6, 3)
    assert layer.lora_B.shape == (3, 4)


# --------------------------------------------------------------------------- #
# 2/3. zero update at init & equivalence to the frozen base linear layer
# --------------------------------------------------------------------------- #
def test_lora_update_is_zero_at_initialisation():
    layer = LoRALinear(8, 5, rank=4)
    np.testing.assert_array_equal(layer.lora_B.numpy(), 0.0)
    assert np.any(layer.lora_A.numpy() != 0.0)
    np.testing.assert_array_equal(layer.delta_weight(), 0.0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_shape", [(7,), (2, 3)])
def test_initial_output_equals_base_linear(bias, batch_shape):
    layer = LoRALinear(6, 4, rank=2, bias=bias)
    x = _x(batch_shape + (6,), seed=1)
    base = x.numpy() @ layer.weight.numpy()
    if bias:
        base = base + layer.bias.numpy()
    np.testing.assert_allclose(layer(x).numpy(), base, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bias", [True, False])
def test_from_linear_matches_wrapped_layer_exactly(bias):
    lin = nn.Linear(9, 5, bias=bias)
    layer = LoRALinear.from_linear(lin, rank=3, alpha=6.0)
    assert layer.in_features == 9 and layer.out_features == 5 and layer.rank == 3 and layer.scaling == 2.0
    # weights preserved exactly, and storage is shared (no copy)
    np.testing.assert_array_equal(layer.weight.numpy(), lin.weight.numpy())
    assert np.shares_memory(layer.weight.cached_data, lin.weight.cached_data)
    if bias:
        np.testing.assert_array_equal(layer.bias.numpy(), lin.bias.numpy())
        assert np.shares_memory(layer.bias.cached_data, lin.bias.cached_data)
    else:
        assert layer.bias is None
    x = _x((4, 9), seed=2)
    np.testing.assert_allclose(layer(x).numpy(), lin(x).numpy(), rtol=1e-6, atol=1e-7)
    # the wrapper's base params are frozen while the original module is untouched
    assert layer.weight.requires_grad is False
    assert lin.weight.requires_grad is True


def test_from_linear_rejects_non_linear():
    with pytest.raises(TypeError):
        LoRALinear.from_linear(nn.ReLU())


# --------------------------------------------------------------------------- #
# 4. equivalence to explicit matrix multiplication with W_eff = W0 + s A B
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("in_f,out_f,rank", [(6, 4, 2), (4, 6, 1), (16, 16, 8), (5, 3, 3)])
def test_matches_effective_weight_projection_with_nonzero_B(in_f, out_f, rank):
    layer = LoRALinear(in_f, out_f, rank=rank, alpha=2.5 * rank)
    rng = np.random.default_rng(42)
    layer.load_adapter(B=rng.standard_normal((rank, out_f)).astype(np.float32))
    x_np = rng.standard_normal((5, in_f)).astype(np.float32)
    y = layer(ndl.Tensor(x_np, dtype="float32", requires_grad=False)).numpy()

    W_eff = layer.weight.numpy() + layer.scaling * layer.lora_A.numpy() @ layer.lora_B.numpy()
    np.testing.assert_allclose(layer.effective_weight(), W_eff, rtol=1e-6, atol=1e-6)
    expected = x_np @ W_eff + layer.bias.numpy()
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
    # and against the float64 numpy reference of the exact formula
    np.testing.assert_allclose(y, _reference(layer, x_np), rtol=1e-5, atol=1e-5)


def test_batched_input_matches_reference_with_nonzero_adapter():
    layer = LoRALinear(6, 4, rank=2, alpha=4.0)
    rng = np.random.default_rng(7)
    layer.load_adapter(A=rng.standard_normal((6, 2)), B=rng.standard_normal((2, 4)))
    x_np = rng.standard_normal((2, 3, 6)).astype(np.float32)
    y = layer(ndl.Tensor(x_np, dtype="float32", requires_grad=False)).numpy()
    np.testing.assert_allclose(y, _reference(layer, x_np), rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# 5/6/7. dimensions, ranks, bias handling, scaling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rank", [1, 2, 4, 8])
def test_multiple_ranks_counts_and_scaling(rank):
    layer = LoRALinear(16, 8, rank=rank, alpha=16.0)
    assert layer.scaling == pytest.approx(16.0 / rank)
    assert layer.num_adapter_parameters() == rank * (16 + 8)


def test_scaling_is_applied():
    """Doubling alpha doubles the LoRA correction."""
    rng = np.random.default_rng(3)
    A, B = rng.standard_normal((6, 2)), rng.standard_normal((2, 4))
    x = _x((5, 6), seed=4)
    l1 = LoRALinear(6, 4, rank=2, alpha=2.0, bias=False)
    l2 = LoRALinear(6, 4, rank=2, alpha=4.0, bias=False, base_weight=l1.weight)
    l1.load_adapter(A=A, B=B)
    l2.load_adapter(A=A, B=B)
    d1 = l1(x).numpy() - x.numpy() @ l1.weight.numpy()
    d2 = l2(x).numpy() - x.numpy() @ l2.weight.numpy()
    np.testing.assert_allclose(d2, 2 * d1, rtol=1e-5, atol=1e-5)


def test_bias_false_has_no_bias_parameter():
    layer = LoRALinear(6, 4, rank=2, bias=False)
    assert layer.bias is None
    assert len(layer.parameters()) == 3
    assert layer.num_base_parameters() == 24
    assert layer.effective_bias() is None


def test_bias_is_frozen_and_counted_as_base_not_adapter():
    layer = LoRALinear(6, 4, rank=2, bias=True)
    assert layer.bias.requires_grad is False
    assert layer.num_base_parameters() == 6 * 4 + 4
    assert layer.num_adapter_parameters() == 2 * (6 + 4)
    assert layer.num_total_parameters() == layer.num_base_parameters() + layer.num_adapter_parameters()


def test_base_weight_can_be_imported_from_numpy():
    W = np.random.default_rng(5).standard_normal((6, 4)).astype(np.float32)
    b = np.arange(4, dtype=np.float32)
    layer = LoRALinear(6, 4, rank=2, base_weight=W, base_bias=b)
    np.testing.assert_array_equal(layer.weight.numpy(), W)
    np.testing.assert_array_equal(layer.bias.numpy(), b)


def test_float64_layer():
    layer = LoRALinear(6, 4, rank=2, dtype="float64")
    assert layer.weight.dtype == np.float64 and layer.lora_A.dtype == np.float64
    y = layer(_x((3, 6), dtype=np.float64))
    assert y.dtype == np.float64


# --------------------------------------------------------------------------- #
# 8. invalid configurations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs, exc",
    [
        (dict(in_features=0, out_features=4), ValueError),
        (dict(in_features=6, out_features=-1), ValueError),
        (dict(in_features=6, out_features=4, rank=0), ValueError),
        (dict(in_features=6, out_features=4, rank=-2), ValueError),
        (dict(in_features=6, out_features=4, rank=5), ValueError),  # rank > min(in, out)
        (dict(in_features=6, out_features=4, rank=2.5), TypeError),
        (dict(in_features=6.0, out_features=4), TypeError),
        (dict(in_features=6, out_features=4, rank=2, alpha=0.0), ValueError),
        (dict(in_features=6, out_features=4, rank=2, alpha=-1.0), ValueError),
        (dict(in_features=6, out_features=4, rank=2, alpha=float("nan")), ValueError),
        (dict(in_features=6, out_features=4, rank=2, alpha="16"), TypeError),
        (dict(in_features=6, out_features=4, bias=False, base_bias=np.zeros(4)), ValueError),
        (dict(in_features=6, out_features=4, base_weight=np.zeros((4, 6))), ValueError),
    ],
)
def test_invalid_configuration_raises(kwargs, exc):
    with pytest.raises(exc):
        LoRALinear(**kwargs)


def test_wrong_input_width_raises():
    layer = LoRALinear(6, 4, rank=2)
    with pytest.raises(ValueError):
        layer(_x((3, 5)))
    with pytest.raises(ValueError):
        layer(_x((6,)))


def test_load_adapter_shape_check():
    layer = LoRALinear(6, 4, rank=2)
    with pytest.raises(ValueError):
        layer.load_adapter(A=np.zeros((6, 3)))
    with pytest.raises(ValueError):
        layer.load_adapter(B=np.zeros((3, 4)))
