"""Phase 3A — SparseTopKRouter (masked, renormalised top-k over a dense softmax router)."""
import numpy as np
import pytest

import needle as ndl
from molora import DenseSoftmaxRouter, SparseTopKRouter


def _np_softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _np_route(x, W, b, T, k):
    p = _np_softmax((x @ W + b) / T)
    idx = np.argsort(-p, axis=-1, kind="stable")[..., :k]
    m = np.zeros_like(p)
    np.put_along_axis(m, idx, 1, axis=-1)
    s = p * m
    return s / s.sum(-1, keepdims=True), p, idx, m


def _t(a, dtype="float64", requires_grad=False):
    return ndl.Tensor(np.asarray(a), dtype=dtype, requires_grad=requires_grad)


def _router(k, E=5, d=6, T=1.0, mode="renormalized", seed=0):
    rng = np.random.default_rng(seed)
    r = SparseTopKRouter(d, E, k, temperature=T, dtype="float64", router_grad_mode=mode)
    r.dense.weight.data = _t(rng.standard_normal((d, E)))
    r.dense.bias.data = _t(rng.standard_normal(E))
    return r, rng


@pytest.mark.parametrize("batch_shape", [(7,), (2, 3)])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_shapes_metadata_and_numpy_reference(batch_shape, k):
    r, rng = _router(k, T=0.7)
    x = rng.standard_normal(batch_shape + (6,))
    sel = r.route(_t(x))
    w_ref, p_ref, idx_ref, m_ref = _np_route(x, r.weight.numpy(), r.bias.numpy(), 0.7, k)
    assert sel.weights.shape == batch_shape + (5,) and sel.indices.shape == batch_shape + (k,)
    np.testing.assert_allclose(sel.weights.numpy(), w_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sel.dense_probs.numpy(), p_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(sel.indices, idx_ref)
    np.testing.assert_array_equal(sel.mask, m_ref)
    np.testing.assert_allclose(r(_t(x)).numpy(), w_ref, rtol=1e-12, atol=1e-12)


def test_k_equals_E_matches_dense_router_exactly():
    r, rng = _router(5)
    dense = DenseSoftmaxRouter(6, 5, dtype="float64")
    dense.weight.data, dense.bias.data = _t(r.weight.numpy()), _t(r.bias.numpy())
    x = _t(rng.standard_normal((4, 6)))
    np.testing.assert_array_equal(r(x).numpy(), dense(x).numpy())


def test_k1_weights_are_one_hot_at_argmax():
    r, rng = _router(1)
    x = rng.standard_normal((8, 6))
    sel = r.route(_t(x))
    w = sel.weights.numpy()
    np.testing.assert_array_equal(w.sum(-1), 1.0)
    np.testing.assert_array_equal(w.argmax(-1), sel.dense_probs.numpy().argmax(-1))
    np.testing.assert_array_equal(np.sort(w, axis=-1)[:, -1], 1.0)


def test_parameters_are_only_the_dense_router_parameters():
    r, _ = _router(2)
    params = r.parameters()
    assert [id(p) for p in params] == [id(r.dense.weight), id(r.dense.bias)]
    assert r.num_parameters() == 6 * 5 + 5
    assert r.weight is r.dense.weight and r.bias is r.dense.bias


@pytest.mark.parametrize("kwargs, exc", [
    (dict(input_dim=6, n_experts=5, top_k=0), ValueError),
    (dict(input_dim=6, n_experts=5, top_k=6), ValueError),
    (dict(input_dim=6, n_experts=5, top_k=1.5), TypeError),
    (dict(input_dim=6, n_experts=5, top_k=2, temperature=0.0), ValueError),
    (dict(input_dim=6, n_experts=5, top_k=2, router_grad_mode="magic"), ValueError),
])
def test_invalid_arguments(kwargs, exc):
    with pytest.raises(exc):
        SparseTopKRouter(**kwargs)


def test_extreme_inputs_are_finite():
    r, rng = _router(2)
    x = rng.standard_normal((5, 6)) * 1e4
    w = r(_t(x)).numpy()
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-12)


def _fd(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        ap, am = arr.copy(), arr.copy()
        ap[idx] += eps
        am[idx] -= eps
        g[idx] = (f(ap) - f(am)) / (2 * eps)
    return g


def test_router_gradients_match_finite_differences_away_from_boundaries():
    r, rng = _router(2, T=0.8, seed=11)
    W, b = r.weight.numpy(), r.bias.numpy()
    cand = rng.standard_normal((300, 6))
    srt = np.sort(_np_softmax((cand @ W + b) / 0.8), axis=-1)
    x = cand[srt[:, -2] - srt[:, -3] > 0.05][:6]  # away from the selection boundary
    assert len(x) == 6
    R = rng.standard_normal((6, 5))
    xt = _t(x, requires_grad=True)
    (r(xt) * _t(R)).sum().backward()

    def loss(W_=W, b_=b, x_=x):
        return np.sum(_np_route(x_, W_, b_, 0.8, 2)[0] * R)

    np.testing.assert_allclose(r.weight.grad.numpy(), _fd(lambda a: loss(W_=a), W), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r.bias.grad.numpy(), _fd(lambda a: loss(b_=a), b), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(xt.grad.numpy(), _fd(lambda a: loss(x_=a), x), rtol=1e-6, atol=1e-8)


def test_top1_exact_mode_has_zero_router_gradient_but_straight_through_does_not():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((6, 6))
    R = rng.standard_normal((6, 5))
    r_exact, _ = _router(1, mode="renormalized", seed=12)
    w_exact = r_exact(_t(x))
    assert w_exact.requires_grad is False  # constant one-hot: no path to the router parameters
    assert r_exact.weight.grad is None and r_exact.bias.grad is None

    r_ste, _ = _router(1, mode="straight_through", seed=12)
    sel = r_ste.route(_t(x))
    np.testing.assert_array_equal(sel.weights.numpy(), r_exact(_t(x)).numpy())  # same forward value
    (sel.weights * _t(R)).sum().backward()
    assert np.linalg.norm(r_ste.weight.grad.numpy()) > 1e-3
    # surrogate = gradient of sum(m * softmax(z) * R) wrt (W, b): compare with the analytic softmax Jacobian
    p = sel.dense_probs.numpy()
    upstream = sel.mask * R  # dL/dp on the selection
    dz = p * (upstream - (upstream * p).sum(-1, keepdims=True))  # softmax Jacobian^T, T = 1
    np.testing.assert_allclose(r_ste.weight.grad.numpy(), x.T @ dz, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(r_ste.bias.grad.numpy(), dz.sum(0), rtol=1e-10, atol=1e-12)
