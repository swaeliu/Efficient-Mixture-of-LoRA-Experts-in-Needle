"""Phase 3A — gradient correctness of SparseMoLoRALinear (reference mode, float64).

Finite differences are evaluated only at inputs away from top-k selection
boundaries (asserted via the probability margin between the k-th and (k+1)-th
expert), where the function is smooth with a fixed selection.
"""
import numpy as np
import pytest

import needle as ndl
from molora import SparseMoLoRALinear

TOL = dict(rtol=1e-6, atol=1e-7)


def _np_softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _t(a, requires_grad=False, dtype="float64"):
    return ndl.Tensor(np.asarray(a), dtype=dtype, requires_grad=requires_grad)


def _make(E=3, k=2, in_f=5, out_f=4, rank=2, bias=True, batch_shape=(4,), seed=0, T=0.8, mode="renormalized",
          margin=0.03):
    rng = np.random.default_rng(seed)
    m = SparseMoLoRALinear(in_f, out_f, n_experts=E, rank=rank, alpha=3.0, bias=bias, top_k=k, router_temperature=T,
                           dtype="float64", router_grad_mode=mode)
    for i in range(E):
        m.load_expert(i, A=rng.standard_normal((in_f, rank)), B=rng.standard_normal((rank, out_f)))
    m.load_router(weight=rng.standard_normal((in_f, E)), bias=rng.standard_normal(E))
    # keep only tokens whose k-th / (k+1)-th probability margin is away from the selection boundary
    n = int(np.prod(batch_shape))
    cand = rng.standard_normal((200 * n, in_f))
    p = _np_softmax((cand @ m.router.weight.numpy() + m.router.bias.numpy()) / T)
    srt = np.sort(p, axis=-1)
    ok = np.ones(len(cand), bool) if k == E else (srt[:, -k] - srt[:, -k - 1] > margin)
    x = cand[ok][:n].reshape(batch_shape + (in_f,))
    assert x.shape[:-1] == batch_shape
    target = rng.standard_normal(batch_shape + (out_f,))
    return m, x, target


def _params(m):
    d = {"W0": m.weight.numpy(), "Wg": m.router.weight.numpy(), "bg": m.router.bias.numpy()}
    if m.bias is not None:
        d["b"] = m.bias.numpy()
    for i, e in enumerate(m.experts):
        d[f"A{i}"], d[f"B{i}"] = e.lora_A.numpy(), e.lora_B.numpy()
    return d


def _forward_np(m, p, x):
    xf = x.reshape(-1, x.shape[-1])
    y = xf @ p["W0"]
    if "b" in p:
        y = y + p["b"]
    g = _np_softmax((xf @ p["Wg"] + p["bg"]) / m.router_temperature)
    idx = np.argsort(-g, axis=-1, kind="stable")[:, : m.top_k]
    mask = np.zeros_like(g)
    np.put_along_axis(mask, idx, 1, axis=-1)
    w = g * mask
    w = w / w.sum(-1, keepdims=True)
    dys = []
    for i in range(m.n_experts):
        dy = m.scaling * (xf @ p[f"A{i}"] @ p[f"B{i}"])
        dys.append(dy)
        y = y + w[:, i : i + 1] * dy
    return y.reshape(x.shape[:-1] + (m.out_features,)), g, w, mask, dys


def _loss_np(m, p, x, target):
    y = _forward_np(m, p, x)[0]
    return np.sum((y - target) ** 2)


def _fd(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        ap, am = arr.copy(), arr.copy()
        ap[idx] += eps
        am[idx] -= eps
        g[idx] = (f(ap) - f(am)) / (2 * eps)
    return g


def _backward(m, x, target, x_requires_grad=True):
    xt = _t(x, requires_grad=x_requires_grad)
    loss = ((m(xt) - _t(target)) ** 2).sum()
    loss.backward()
    return xt, float(loss.numpy())


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)])
@pytest.mark.parametrize("E,k", [(3, 2), (4, 2), (4, 3)])
def test_fd_all_parameters_and_input(bias, batch_shape, E, k):
    m, x, target = _make(E=E, k=k, bias=bias, batch_shape=batch_shape, seed=E * 7 + k)
    p = _params(m)
    xt, loss = _backward(m, x, target)
    assert np.isclose(loss, _loss_np(m, p, x, target))

    def fd_for(name):
        def f(arr):
            q = dict(p)
            q[name] = arr
            return _loss_np(m, q, x, target)
        return _fd(f, p[name])

    for i, e in enumerate(m.experts):
        np.testing.assert_allclose(e.lora_A.grad.numpy(), fd_for(f"A{i}"), **TOL)
        np.testing.assert_allclose(e.lora_B.grad.numpy(), fd_for(f"B{i}"), **TOL)
    np.testing.assert_allclose(m.router.weight.grad.numpy(), fd_for("Wg"), **TOL)
    np.testing.assert_allclose(m.router.bias.grad.numpy(), fd_for("bg"), **TOL)
    assert np.any(m.router.weight.grad.numpy() != 0)
    np.testing.assert_allclose(xt.grad.numpy(), _fd(lambda xx: _loss_np(m, p, xx, target), x), **TOL)


def test_unselected_experts_get_no_gradient_from_tokens_that_did_not_select_them():
    """dL/dB_i must equal the closed form restricted to the tokens that selected expert i."""
    m, x, target = _make(E=4, k=1, batch_shape=(12,), seed=3)
    p = _params(m)
    _backward(m, x, target, x_requires_grad=False)
    y, g, w, mask, dys = _forward_np(m, p, x)
    G = 2 * (y - target)  # dL/dy
    for i, e in enumerate(m.experts):
        sel = mask[:, i] == 1
        if not sel.any():
            np.testing.assert_array_equal(e.lora_A.grad.numpy(), 0.0)
            np.testing.assert_array_equal(e.lora_B.grad.numpy(), 0.0)
            continue
        xs, Gs, ws = x[sel], G[sel], w[sel, i : i + 1]
        dB = m.scaling * (xs @ p[f"A{i}"]).T @ (ws * Gs)
        dA = m.scaling * xs.T @ ((ws * Gs) @ p[f"B{i}"].T)
        np.testing.assert_allclose(e.lora_B.grad.numpy(), dB, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(e.lora_A.grad.numpy(), dA, rtol=1e-10, atol=1e-10)


def test_expert_never_selected_receives_exactly_zero_gradient():
    m, x, target = _make(E=4, k=2, batch_shape=(8,), seed=4)
    b = m.router.bias.numpy().copy()
    b[3] = -50.0  # expert 4 can never be in the top-2
    m.load_router(bias=b)
    _backward(m, x, target, x_requires_grad=False)
    assert m.last_stats["tokens_per_expert"][3] == 0
    np.testing.assert_array_equal(m.experts[3].lora_A.grad.numpy(), 0.0)
    np.testing.assert_array_equal(m.experts[3].lora_B.grad.numpy(), 0.0)
    for i in range(3):
        assert np.any(m.experts[i].lora_B.grad.numpy() != 0)


def test_top1_exact_mode_router_gradient_is_zero_and_straight_through_is_not():
    m, x, target = _make(E=4, k=1, batch_shape=(8,), seed=5, mode="renormalized")
    _backward(m, x, target, x_requires_grad=False)
    assert m.router.weight.grad is None and m.router.bias.grad is None  # constant gates: no router path
    for e in m.experts:
        if np.any(e.lora_B.grad.numpy() != 0):
            break
    else:
        raise AssertionError("some expert must receive gradient")

    m2, x2, target2 = _make(E=4, k=1, batch_shape=(8,), seed=5, mode="straight_through")
    _backward(m2, x2, target2, x_requires_grad=False)
    gW = m2.router.weight.grad.numpy()
    assert np.linalg.norm(gW) > 1e-6
    # surrogate check: dL/dg_i = <G, dy_i> on the selected expert, pushed through the softmax Jacobian
    p = _params(m2)
    y, g, w, mask, dys = _forward_np(m2, p, x2)
    G = 2 * (y - target2)
    up = np.stack([(G * dy).sum(-1) for dy in dys], axis=-1) * mask  # (N, E)
    dz = g * (up - (up * g).sum(-1, keepdims=True)) / m2.router_temperature
    np.testing.assert_allclose(gW, x2.T @ dz, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(m2.router.bias.grad.numpy(), dz.sum(0), rtol=1e-9, atol=1e-10)


def test_gradients_at_zero_B_initialisation():
    m = SparseMoLoRALinear(5, 4, n_experts=3, rank=2, alpha=3.0, top_k=2, dtype="float64")
    rng = np.random.default_rng(6)
    x, target = rng.standard_normal((6, 5)), rng.standard_normal((6, 4))
    _backward(m, x, target, x_requires_grad=False)
    for e in m.experts:
        np.testing.assert_array_equal(e.lora_A.grad.numpy(), 0.0)
    assert any(np.any(e.lora_B.grad.numpy() != 0) for e in m.experts)
    np.testing.assert_array_equal(m.router.weight.grad.numpy(), 0.0)
    assert m.weight.grad is None and m.bias.grad is None


def test_float32_matches_float64_oracle():
    rng = np.random.default_rng(7)
    m = SparseMoLoRALinear(6, 4, n_experts=4, rank=2, alpha=3.0, top_k=2, dtype="float32")
    for i in range(4):
        m.load_expert(i, A=rng.standard_normal((6, 2)), B=rng.standard_normal((2, 4)))
    m.load_router(weight=rng.standard_normal((6, 4)), bias=rng.standard_normal(4))
    x = rng.standard_normal((5, 6)).astype(np.float32)
    target = rng.standard_normal((5, 4)).astype(np.float32)
    xt = ndl.Tensor(x, dtype="float32", requires_grad=True)
    ((m(xt) - ndl.Tensor(target, dtype="float32", requires_grad=False)) ** 2).sum().backward()
    p = {k: v.astype(np.float64) for k, v in _params(m).items()}
    x64, t64 = x.astype(np.float64), target.astype(np.float64)
    for i, e in enumerate(m.experts):
        for name, par in ((f"A{i}", e.lora_A), (f"B{i}", e.lora_B)):
            def f(arr, name=name):
                q = dict(p); q[name] = arr
                return _loss_np(m, q, x64, t64)
            np.testing.assert_allclose(par.grad.numpy(), _fd(f, p[name]), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(xt.grad.numpy(), _fd(lambda a: _loss_np(m, p, a, t64), x64), rtol=1e-3, atol=1e-3)
