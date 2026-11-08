"""Phase 3A — top-k selection and renormalised sparse gates."""
import numpy as np
import pytest

import needle as ndl
from molora.ops.topk import sparse_renormalized_gates, topk_indices, topk_mask


def _np_softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _np_sparse_gates(p, k):
    idx = np.argsort(-p, axis=-1, kind="stable")[..., :k]
    m = np.zeros_like(p)
    np.put_along_axis(m, idx, 1, axis=-1)
    s = p * m
    return s / s.sum(-1, keepdims=True), idx, m


def _t(a, dtype="float64", requires_grad=False):
    return ndl.Tensor(np.asarray(a), dtype=dtype, requires_grad=requires_grad)


@pytest.mark.parametrize("shape", [(6,), (5, 6), (2, 3, 6), (2, 2, 3, 6)])
@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_topk_indices_membership_and_order(shape, k):
    v = np.random.default_rng(0).standard_normal(shape)
    idx = topk_indices(v, k)
    assert idx.shape == shape[:-1] + (k,)
    top_vals = np.take_along_axis(v, idx, axis=-1)
    # descending and equal to the k largest values
    assert np.all(np.diff(top_vals, axis=-1) <= 0)
    np.testing.assert_array_equal(np.sort(top_vals, axis=-1), np.sort(v, axis=-1)[..., -k:])
    # exactly k distinct experts per row
    assert all(len(set(row)) == k for row in idx.reshape(-1, k))


def test_topk_mask_has_exactly_k_ones():
    v = np.random.default_rng(1).standard_normal((4, 5, 7))
    for k in (1, 3, 7):
        m = topk_mask(v, k)
        assert m.shape == v.shape and m.dtype == v.dtype
        np.testing.assert_array_equal(m.sum(-1), k)
        assert set(np.unique(m)) <= {0.0, 1.0}


def test_tie_breaking_is_deterministic_lowest_index_first():
    v = np.array([[1.0, 3.0, 3.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(topk_indices(v, 1), [[1], [0]])
    np.testing.assert_array_equal(topk_indices(v, 2), [[1, 2], [0, 1]])
    np.testing.assert_array_equal(topk_indices(v, 3), [[1, 2, 4], [0, 1, 2]])
    # repeated calls agree bitwise
    np.testing.assert_array_equal(topk_indices(v, 3), topk_indices(v.copy(), 3))


@pytest.mark.parametrize("k", [0, -1, 7, 2.0, True])
def test_invalid_k_raises(k):
    v = np.zeros((3, 6))
    with pytest.raises((ValueError, TypeError)):
        topk_indices(v, k)
    with pytest.raises((ValueError, TypeError)):
        sparse_renormalized_gates(_t(_np_softmax(v)), k)


@pytest.mark.parametrize("shape", [(5, 6), (2, 3, 6)])
@pytest.mark.parametrize("k", [1, 2, 4, 6])
def test_sparse_gates_match_numpy_reference(shape, k):
    z = np.random.default_rng(2).standard_normal(shape) * 2
    p = _np_softmax(z)
    sel = sparse_renormalized_gates(_t(p), k)
    w_ref, idx_ref, m_ref = _np_sparse_gates(p, k)
    np.testing.assert_allclose(sel.weights.numpy(), w_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(sel.indices, idx_ref)
    np.testing.assert_array_equal(sel.mask, m_ref)
    w = sel.weights.numpy()
    np.testing.assert_array_equal((w > 0).sum(-1), k)  # exactly k selected
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-12)  # renormalised
    np.testing.assert_array_equal(w[m_ref == 0], 0.0)  # unselected are exactly zero
    assert sel.weights.shape == shape and sel.dense_probs is not None


def test_k_equals_E_returns_dense_probabilities_unchanged():
    p = _np_softmax(np.random.default_rng(3).standard_normal((4, 5)))
    sel = sparse_renormalized_gates(_t(p), 5)
    np.testing.assert_allclose(sel.weights.numpy(), p, rtol=0, atol=0)


def test_extreme_logits_no_nan():
    z = np.array([[1e4, -1e4, 0.0, 5.0], [700.0, 700.0, -700.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    from molora import softmax

    p = softmax(_t(z))
    for k in (1, 2, 4):
        w = sparse_renormalized_gates(p, k).weights.numpy()
        assert np.all(np.isfinite(w))
        np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-12)
    # k=1 on a two-way tie picks the lower index
    np.testing.assert_array_equal(sparse_renormalized_gates(p, 1).indices[1], [0])


def _fd(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        ap, am = arr.copy(), arr.copy()
        ap[idx] += eps
        am[idx] -= eps
        g[idx] = (f(ap) - f(am)) / (2 * eps)
    return g


@pytest.mark.parametrize("k", [2, 3])
def test_gradient_of_renormalized_gates_matches_finite_differences(k):
    """FD on the logits -> softmax -> top-k renormalisation chain, away from selection boundaries."""
    rng = np.random.default_rng(4)
    # keep only logit rows whose k-th / (k+1)-th probability margin is clearly away from the boundary
    cand = rng.standard_normal((200, 5)) * 2
    srt = np.sort(_np_softmax(cand), axis=-1)
    z = cand[srt[:, -k] - srt[:, -k - 1] > 0.05][:6]
    assert len(z) == 6
    R = rng.standard_normal((6, 5))
    from molora import softmax

    zt = _t(z, requires_grad=True)
    (sparse_renormalized_gates(softmax(zt), k).weights * _t(R)).sum().backward()

    def f(zz):
        w, _, _ = _np_sparse_gates(_np_softmax(zz), k)
        return np.sum(w * R)

    np.testing.assert_allclose(zt.grad.numpy(), _fd(f, z), rtol=1e-6, atol=1e-8)


def test_gradient_is_zero_on_unselected_entries_and_zero_for_k1():
    rng = np.random.default_rng(5)
    p = _np_softmax(rng.standard_normal((6, 5)) * 2)
    R = rng.standard_normal((6, 5))
    pt = _t(p, requires_grad=True)
    sel = sparse_renormalized_gates(pt, 2)
    (sel.weights * _t(R)).sum().backward()
    g = pt.grad.numpy()
    np.testing.assert_array_equal(g[sel.mask == 0], 0.0)
    assert np.any(g[sel.mask == 1] != 0)
    # k = 1: the renormalised weight is identically 1 on the selection -> the weights are a constant
    # and the probabilities receive no gradient at all
    pt1 = _t(p, requires_grad=True)
    w1 = sparse_renormalized_gates(pt1, 1).weights
    assert w1.requires_grad is False
    np.testing.assert_array_equal(w1.numpy(), sparse_renormalized_gates(_t(p), 1).mask)
    assert getattr(pt1, "grad", None) is None


def test_straight_through_value_and_surrogate_gradient():
    rng = np.random.default_rng(6)
    p = _np_softmax(rng.standard_normal((6, 5)) * 2)
    R = rng.standard_normal((6, 5))
    pt = _t(p, requires_grad=True)
    sel = sparse_renormalized_gates(pt, 1, straight_through=True)
    w_exact = sparse_renormalized_gates(_t(p), 1).weights.numpy()
    np.testing.assert_array_equal(sel.weights.numpy(), w_exact)  # same forward value
    (sel.weights * _t(R)).sum().backward()
    # surrogate: gradient of sum(m * p * R) wrt p = m * R
    np.testing.assert_allclose(pt.grad.numpy(), sel.mask * R, rtol=0, atol=1e-15)
