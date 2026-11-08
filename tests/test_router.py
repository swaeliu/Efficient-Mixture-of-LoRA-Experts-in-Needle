"""Phase 2 — DenseSoftmaxRouter and the Needle-native softmax."""
import numpy as np
import pytest

import needle as ndl
from molora import DenseSoftmaxRouter, softmax


def _np_softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _t(a, dtype="float64", requires_grad=False):
    return ndl.Tensor(np.asarray(a), dtype=dtype, requires_grad=requires_grad)


@pytest.mark.parametrize("shape", [(5, 4), (2, 3, 4), (2, 3, 5, 4)])
def test_softmax_matches_numpy_on_last_axis(shape):
    z = np.random.default_rng(0).standard_normal(shape) * 3
    p = softmax(_t(z)).numpy()
    np.testing.assert_allclose(p, _np_softmax(z), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p.sum(-1), 1.0, rtol=0, atol=1e-12)


def test_softmax_is_stable_for_large_logits():
    z = np.array([[1000.0, 1000.0, -1000.0], [50.0, 0.0, -50.0]])
    p = softmax(_t(z)).numpy()
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p[0], [0.5, 0.5, 0.0], atol=1e-12)


def test_softmax_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((2, 3, 4))
    R = rng.standard_normal((2, 3, 4))
    zt = _t(z, requires_grad=True)
    (softmax(zt) * _t(R)).sum().backward()
    eps = 1e-6
    fd = np.zeros_like(z)
    for idx in np.ndindex(z.shape):
        zp, zm = z.copy(), z.copy()
        zp[idx] += eps
        zm[idx] -= eps
        fd[idx] = (np.sum(_np_softmax(zp) * R) - np.sum(_np_softmax(zm) * R)) / (2 * eps)
    np.testing.assert_allclose(zt.grad.numpy(), fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("batch_shape", [(6,), (2, 3)])
def test_router_output_shape_and_normalisation(batch_shape):
    r = DenseSoftmaxRouter(7, 4)
    x = _t(np.random.default_rng(2).standard_normal(batch_shape + (7,)), dtype="float32")
    p = r(x).numpy()
    assert p.shape == batch_shape + (4,)
    assert np.all(np.isfinite(p)) and np.all(p >= 0)
    np.testing.assert_allclose(p.sum(-1), 1.0, atol=1e-6)
    # normalisation is over the expert axis only: not over batch / sequence
    assert not np.allclose(p.sum(axis=0), 1.0)


def test_router_matches_numpy_reference_with_temperature():
    rng = np.random.default_rng(3)
    for T in (0.5, 1.0, 2.0):
        r = DenseSoftmaxRouter(5, 3, temperature=T, dtype="float64")
        r.weight.data = _t(rng.standard_normal((5, 3)))
        r.bias.data = _t(rng.standard_normal(3))
        x = rng.standard_normal((2, 4, 5))
        ref = _np_softmax((x @ r.weight.numpy() + r.bias.numpy()) / T)
        np.testing.assert_allclose(r(_t(x)).numpy(), ref, rtol=1e-12, atol=1e-12)


def test_higher_temperature_increases_entropy_and_approaches_uniform():
    rng = np.random.default_rng(4)
    W, b, x = rng.standard_normal((5, 3)), rng.standard_normal(3), rng.standard_normal((8, 5))

    def entropy(T):
        r = DenseSoftmaxRouter(5, 3, temperature=T, dtype="float64")
        r.weight.data, r.bias.data = _t(W), _t(b)
        p = r(_t(x)).numpy()
        return -(p * np.log(p + 1e-300)).sum(-1).mean(), p

    h_low, _ = entropy(0.1)
    h_mid, _ = entropy(1.0)
    h_high, p_high = entropy(1000.0)
    assert h_low < h_mid < h_high
    np.testing.assert_allclose(p_high, 1 / 3, atol=5e-3)  # logits / 1000 are O(1e-3)


def test_router_parameter_count_and_init():
    r = DenseSoftmaxRouter(128, 4)
    assert r.num_parameters() == 128 * 4 + 4 == sum(int(np.prod(p.shape)) for p in r.parameters())
    assert r.weight.shape == (128, 4) and r.bias.shape == (4,)
    np.testing.assert_array_equal(r.bias.numpy(), 0.0)
    assert np.any(r.weight.numpy() != 0.0)
    rz = DenseSoftmaxRouter(6, 3, weight_init="zeros")
    p = rz(_t(np.random.default_rng(5).standard_normal((4, 6)), dtype="float32")).numpy()
    np.testing.assert_allclose(p, 1 / 3, atol=1e-7)


@pytest.mark.parametrize("kwargs, exc", [
    (dict(input_dim=5, n_experts=3, temperature=0.0), ValueError),
    (dict(input_dim=5, n_experts=3, temperature=-1.0), ValueError),
    (dict(input_dim=5, n_experts=3, temperature=float("inf")), ValueError),
    (dict(input_dim=5, n_experts=3, temperature="1"), TypeError),
    (dict(input_dim=5, n_experts=0), ValueError),
    (dict(input_dim=0, n_experts=3), ValueError),
    (dict(input_dim=5, n_experts=2.0), TypeError),
    (dict(input_dim=5, n_experts=3, weight_init="random"), ValueError),
])
def test_router_invalid_arguments(kwargs, exc):
    with pytest.raises(exc):
        DenseSoftmaxRouter(**kwargs)


def test_router_rejects_wrong_input_width():
    r = DenseSoftmaxRouter(5, 3)
    with pytest.raises(ValueError):
        r(_t(np.zeros((2, 4)), dtype="float32"))


def test_router_gradients_flow_to_weight_bias_and_input():
    rng = np.random.default_rng(6)
    r = DenseSoftmaxRouter(5, 3, dtype="float64")
    x = _t(rng.standard_normal((4, 5)), requires_grad=True)
    R = _t(rng.standard_normal((4, 3)))
    (r(x) * R).sum().backward()
    for p in (r.weight, r.bias, x):
        assert p.grad is not None and np.all(np.isfinite(p.grad.numpy())) and np.any(p.grad.numpy() != 0)
