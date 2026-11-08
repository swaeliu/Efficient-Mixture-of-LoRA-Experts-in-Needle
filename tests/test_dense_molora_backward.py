"""Phase 2 — gradient correctness of DenseMoLoRALinear.

Finite-difference checks (float64, central differences) for every expert's A
and B, the router weight and bias, and the input; both at a deterministic
non-zero adapter state (where the router actually receives signal) and at the
default zero-B initialisation (where router and A gradients must be exactly 0).
"""
import numpy as np
import pytest

import needle as ndl
from molora import DenseMoLoRALinear

# atol 1e-7: central differences with eps=1e-6 on a loss of magnitude ~1e2 carry ~1e-8 rounding noise
TOL = dict(rtol=1e-6, atol=1e-7)


def _np_softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _t(a, requires_grad=False, dtype="float64"):
    return ndl.Tensor(np.asarray(a), dtype=dtype, requires_grad=requires_grad)


def _make(n_experts=3, in_f=5, out_f=4, rank=2, bias=True, batch_shape=(3,), seed=0, randomise=True, T=0.8):
    rng = np.random.default_rng(seed)
    m = DenseMoLoRALinear(in_f, out_f, n_experts=n_experts, rank=rank, alpha=3.0, bias=bias, router_temperature=T,
                          dtype="float64")
    if randomise:
        for i in range(n_experts):
            m.load_expert(i, A=rng.standard_normal((in_f, rank)), B=rng.standard_normal((rank, out_f)))
        m.load_router(weight=rng.standard_normal((in_f, n_experts)), bias=rng.standard_normal(n_experts))
    x = rng.standard_normal(batch_shape + (in_f,))
    target = rng.standard_normal(batch_shape + (out_f,))
    return m, x, target


def _params(m):
    """Flat dict name -> numpy array of every parameter the oracle needs."""
    d = {"W0": m.weight.numpy(), "Wg": m.router.weight.numpy(), "bg": m.router.bias.numpy()}
    if m.bias is not None:
        d["b"] = m.bias.numpy()
    for i, e in enumerate(m.experts):
        d[f"A{i}"], d[f"B{i}"] = e.lora_A.numpy(), e.lora_B.numpy()
    return d


def _loss_np(m, p, x, target):
    y = x @ p["W0"]
    if "b" in p:
        y = y + p["b"]
    g = _np_softmax((x @ p["Wg"] + p["bg"]) / m.router_temperature)
    for i in range(m.n_experts):
        y = y + g[..., i : i + 1] * (m.scaling * (x @ p[f"A{i}"] @ p[f"B{i}"]))
    return np.sum((y - target) ** 2)


def _fd(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        ap, am = arr.copy(), arr.copy()
        ap[idx] += eps
        am[idx] -= eps
        g[idx] = (f(ap) - f(am)) / (2 * eps)
    return g


def _needle_backward(m, x, target, x_requires_grad=True):
    xt = _t(x, requires_grad=x_requires_grad)
    loss = ((m(xt) - _t(target)) ** 2).sum()
    loss.backward()
    return xt, float(loss.numpy())


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
def test_fd_all_parameters_and_input_nonzero_state(bias, batch_shape):
    m, x, target = _make(bias=bias, batch_shape=batch_shape, seed=1)
    p = _params(m)
    xt, loss = _needle_backward(m, x, target)
    assert np.isfinite(loss) and np.isclose(loss, _loss_np(m, p, x, target))

    def fd_for(name):
        def f(arr):
            q = dict(p)
            q[name] = arr
            return _loss_np(m, q, x, target)
        return _fd(f, p[name])

    for i, e in enumerate(m.experts):
        gA, gB = e.lora_A.grad.numpy(), e.lora_B.grad.numpy()
        assert np.any(gA != 0) and np.any(gB != 0)
        np.testing.assert_allclose(gA, fd_for(f"A{i}"), **TOL)
        np.testing.assert_allclose(gB, fd_for(f"B{i}"), **TOL)
    gW, gb = m.router.weight.grad.numpy(), m.router.bias.grad.numpy()
    assert np.any(gW != 0) and np.any(gb != 0)
    np.testing.assert_allclose(gW, fd_for("Wg"), **TOL)
    np.testing.assert_allclose(gb, fd_for("bg"), **TOL)
    fd_x = _fd(lambda xx: _loss_np(m, p, xx, target), x)
    np.testing.assert_allclose(xt.grad.numpy(), fd_x, **TOL)
    # frozen backbone receives a gradient only because x requires grad (Needle behaviour, Phase 1 §4)
    assert m.weight.requires_grad is False


def test_gradients_at_default_initialisation():
    """B = 0 for every expert: dL/dB_i != 0 (and matches FD), dL/dA_i == 0 and router grads == 0 exactly."""
    m, x, target = _make(randomise=False, seed=2)
    p = _params(m)
    _needle_backward(m, x, target, x_requires_grad=False)
    for i, e in enumerate(m.experts):
        np.testing.assert_array_equal(e.lora_A.grad.numpy(), 0.0)
        np.testing.assert_array_equal(e.lora_B.numpy(), 0.0)

        def f(arr, i=i):
            q = dict(p)
            q[f"B{i}"] = arr
            return _loss_np(m, q, x, target)

        gB = e.lora_B.grad.numpy()
        assert np.any(gB != 0)
        np.testing.assert_allclose(gB, _fd(f, p[f"B{i}"]), **TOL)
    np.testing.assert_array_equal(m.router.weight.grad.numpy(), 0.0)
    np.testing.assert_array_equal(m.router.bias.grad.numpy(), 0.0)
    assert m.weight.grad is None and m.bias.grad is None


def test_router_gradient_is_zero_when_all_experts_are_identical():
    """If every expert computes the same dy, the mixture is independent of the gates."""
    m, x, target = _make(randomise=False, seed=3)
    rng = np.random.default_rng(3)
    A, B = rng.standard_normal((m.in_features, m.rank)), rng.standard_normal((m.rank, m.out_features))
    for i in range(m.n_experts):
        m.load_expert(i, A=A, B=B)
    m.load_router(weight=rng.standard_normal((m.in_features, m.n_experts)), bias=rng.standard_normal(m.n_experts))
    _needle_backward(m, x, target, x_requires_grad=False)
    np.testing.assert_allclose(m.router.weight.grad.numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(m.router.bias.grad.numpy(), 0.0, atol=1e-12)
    for e in m.experts:
        assert np.any(e.lora_A.grad.numpy() != 0) and np.any(e.lora_B.grad.numpy() != 0)


def test_router_gradient_becomes_nonzero_once_experts_differ():
    m, x, target = _make(randomise=False, seed=4)
    rng = np.random.default_rng(4)
    for i in range(m.n_experts):
        m.load_expert(i, B=rng.standard_normal((m.rank, m.out_features)))
    _needle_backward(m, x, target, x_requires_grad=False)
    assert np.linalg.norm(m.router.weight.grad.numpy()) > 1e-6
    assert np.linalg.norm(m.router.bias.grad.numpy()) > 1e-6


def test_single_expert_router_gradient_is_zero_and_experts_match_lora():
    m, x, target = _make(n_experts=1, seed=5)
    _needle_backward(m, x, target, x_requires_grad=False)
    np.testing.assert_allclose(m.router.weight.grad.numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(m.router.bias.grad.numpy(), 0.0, atol=1e-12)


def test_float32_gradients_match_float64_oracle():
    rng = np.random.default_rng(6)
    m = DenseMoLoRALinear(6, 4, n_experts=3, rank=2, alpha=3.0, bias=True, dtype="float32")
    for i in range(3):
        m.load_expert(i, A=rng.standard_normal((6, 2)), B=rng.standard_normal((2, 4)))
    m.load_router(weight=rng.standard_normal((6, 3)), bias=rng.standard_normal(3))
    x = rng.standard_normal((4, 6)).astype(np.float32)
    target = rng.standard_normal((4, 4)).astype(np.float32)
    xt = ndl.Tensor(x, dtype="float32", requires_grad=True)
    ((m(xt) - ndl.Tensor(target, dtype="float32", requires_grad=False)) ** 2).sum().backward()
    p = {k: v.astype(np.float64) for k, v in _params(m).items()}
    x64, t64 = x.astype(np.float64), target.astype(np.float64)
    tol = dict(rtol=1e-3, atol=1e-3)
    for i, e in enumerate(m.experts):
        for name, par in ((f"A{i}", e.lora_A), (f"B{i}", e.lora_B)):
            def f(arr, name=name):
                q = dict(p); q[name] = arr
                return _loss_np(m, q, x64, t64)
            np.testing.assert_allclose(par.grad.numpy(), _fd(f, p[name]), **tol)
    for name, par in (("Wg", m.router.weight), ("bg", m.router.bias)):
        def f(arr, name=name):
            q = dict(p); q[name] = arr
            return _loss_np(m, q, x64, t64)
        np.testing.assert_allclose(par.grad.numpy(), _fd(f, p[name]), **tol)
    np.testing.assert_allclose(xt.grad.numpy(), _fd(lambda xx: _loss_np(m, p, xx, t64), x64), **tol)
