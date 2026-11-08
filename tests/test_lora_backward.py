"""Phase 1 — gradient correctness of LoRALinear.

Three complementary checks:

* finite differences in float64 (tight tolerance) — the primary check,
* Needle float32 autograd against the analytic numpy gradient,
* finite differences in float32 (loose tolerance) to document precision.

Loss used throughout: L = sum((y - T)^2) for a fixed random target T, which is
non-linear in y so dL/dy depends on the forward output.
"""
import numpy as np
import pytest

import needle as ndl
from molora import LoRALinear

F64_TOL = dict(rtol=1e-6, atol=1e-8)
F32_ANALYTIC_TOL = dict(rtol=1e-4, atol=1e-4)
F32_FD_TOL = dict(rtol=2e-2, atol=2e-2)


def _make_layer(in_f=5, out_f=4, rank=2, dtype="float64", bias=True, batch_shape=(3,), seed=0):
    """A layer with deliberately NON-zero B (otherwise dL/dA == 0 trivially)."""
    rng = np.random.default_rng(seed)
    np_dtype = np.dtype(dtype)
    layer = LoRALinear(in_f, out_f, rank=rank, alpha=3.0, bias=bias, dtype=dtype)
    layer.load_adapter(
        A=rng.standard_normal((in_f, rank)).astype(np_dtype),
        B=rng.standard_normal((rank, out_f)).astype(np_dtype),
    )
    x = rng.standard_normal(batch_shape + (in_f,)).astype(np_dtype)
    target = rng.standard_normal(batch_shape + (out_f,)).astype(np_dtype)
    return layer, x, target


def _loss_np(layer, x, target, A=None, B=None, W=None):
    A = layer.lora_A.numpy() if A is None else A
    B = layer.lora_B.numpy() if B is None else B
    W = layer.weight.numpy() if W is None else W
    y = x @ W + layer.scaling * (x @ A @ B)
    if layer.bias is not None:
        y = y + layer.bias.numpy()
    return np.sum((y - target) ** 2)


def _needle_loss(layer, x_t, target_np):
    t = ndl.Tensor(target_np, dtype=str(target_np.dtype), requires_grad=False)
    return ((layer(x_t) - t) ** 2).sum()


def _needle_grads(layer, x, target, x_requires_grad=False):
    x_t = ndl.Tensor(x, dtype=str(x.dtype), requires_grad=x_requires_grad)
    loss = _needle_loss(layer, x_t, target)
    loss.backward()
    grads = {"A": layer.lora_A.grad.numpy(), "B": layer.lora_B.grad.numpy(), "loss": float(loss.numpy())}
    if x_requires_grad:
        grads["x"] = x_t.grad.numpy()
    return grads


def _fd_grad(f, p, eps):
    """Central finite-difference gradient of scalar f wrt array p."""
    g = np.zeros_like(p)
    it = np.nditer(p, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        p_plus, p_minus = p.copy(), p.copy()
        p_plus[idx] += eps
        p_minus[idx] -= eps
        g[idx] = (f(p_plus) - f(p_minus)) / (2 * eps)
    return g


def _analytic_grads(layer, x, target):
    """dL/dA, dL/dB, dL/dx for L = sum((y - T)^2), y = x W0 + b + s x A B."""
    A, B, W = layer.lora_A.numpy(), layer.lora_B.numpy(), layer.weight.numpy()
    s = layer.scaling
    y = x @ W + s * (x @ A @ B)
    if layer.bias is not None:
        y = y + layer.bias.numpy()
    g = 2 * (y - target)  # dL/dy
    x2 = x.reshape(-1, x.shape[-1])
    g2 = g.reshape(-1, g.shape[-1])
    dA = s * x2.T @ (g2 @ B.T)
    dB = s * (x2 @ A).T @ g2
    dx = g @ W.T + s * (g @ B.T) @ A.T
    return dA, dB, dx


# --------------------------------------------------------------------------- #
# float64 finite differences (primary)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
def test_fd_gradients_float64(bias, batch_shape):
    layer, x, target = _make_layer(dtype="float64", bias=bias, batch_shape=batch_shape)
    got = _needle_grads(layer, x, target, x_requires_grad=True)
    eps = 1e-6
    fd_A = _fd_grad(lambda A: _loss_np(layer, x, target, A=A), layer.lora_A.numpy(), eps)
    fd_B = _fd_grad(lambda B: _loss_np(layer, x, target, B=B), layer.lora_B.numpy(), eps)
    fd_x = _fd_grad(lambda xx: _loss_np(layer, xx, target), x, eps)
    assert np.all(np.isfinite(got["A"])) and np.any(got["A"] != 0)
    np.testing.assert_allclose(got["A"], fd_A, **F64_TOL)
    np.testing.assert_allclose(got["B"], fd_B, **F64_TOL)
    np.testing.assert_allclose(got["x"], fd_x, **F64_TOL)


def test_fd_gradient_B_at_default_init_float64():
    """At init B = 0, dL/dB is still non-trivial (depends on x A) while dL/dA is exactly 0."""
    layer = LoRALinear(5, 4, rank=2, alpha=3.0, dtype="float64")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 5))
    target = rng.standard_normal((3, 4))
    got = _needle_grads(layer, x, target)
    fd_B = _fd_grad(lambda B: _loss_np(layer, x, target, B=B), layer.lora_B.numpy(), 1e-6)
    np.testing.assert_allclose(got["B"], fd_B, **F64_TOL)
    np.testing.assert_array_equal(got["A"], 0.0)


# --------------------------------------------------------------------------- #
# float32 autograd vs analytic gradient
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
def test_autograd_matches_analytic_float32(batch_shape):
    layer, x, target = _make_layer(dtype="float32", batch_shape=batch_shape, seed=3)
    got = _needle_grads(layer, x, target, x_requires_grad=True)
    dA, dB, dx = _analytic_grads(layer, x, target)
    np.testing.assert_allclose(got["A"], dA, **F32_ANALYTIC_TOL)
    np.testing.assert_allclose(got["B"], dB, **F32_ANALYTIC_TOL)
    np.testing.assert_allclose(got["x"], dx, **F32_ANALYTIC_TOL)


def test_autograd_matches_analytic_float64():
    layer, x, target = _make_layer(dtype="float64", seed=4)
    got = _needle_grads(layer, x, target, x_requires_grad=True)
    dA, dB, dx = _analytic_grads(layer, x, target)
    np.testing.assert_allclose(got["A"], dA, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got["B"], dB, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got["x"], dx, rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------- #
# float32 finite differences (documents precision limits)
# --------------------------------------------------------------------------- #
def test_fd_gradients_float32_loose():
    layer, x, target = _make_layer(dtype="float32", seed=5)
    got = _needle_grads(layer, x, target)
    # FD evaluated in float64 on float32-valued parameters to isolate autograd's own f32 error
    x64, t64 = x.astype(np.float64), target.astype(np.float64)
    A64, B64 = layer.lora_A.numpy().astype(np.float64), layer.lora_B.numpy().astype(np.float64)
    W64 = layer.weight.numpy().astype(np.float64)
    b64 = layer.bias.numpy().astype(np.float64)
    s = layer.scaling

    def loss64(A=A64, B=B64):
        return np.sum((x64 @ W64 + b64 + s * (x64 @ A @ B) - t64) ** 2)

    fd_A = _fd_grad(lambda A: loss64(A=A), A64, 1e-6)
    fd_B = _fd_grad(lambda B: loss64(B=B), B64, 1e-6)
    np.testing.assert_allclose(got["A"], fd_A, **F32_ANALYTIC_TOL)
    np.testing.assert_allclose(got["B"], fd_B, **F32_ANALYTIC_TOL)
    # pure float32 central differences are much noisier: only a loose agreement is expected
    fd_A32 = _fd_grad(lambda A: _loss_np(layer, x, target, A=A.astype(np.float32)), layer.lora_A.numpy(), 1e-2)
    np.testing.assert_allclose(got["A"], fd_A32, **F32_FD_TOL)


# --------------------------------------------------------------------------- #
# input gradient flow and frozen-parameter gradient behaviour
# --------------------------------------------------------------------------- #
def test_constant_input_keeps_frozen_base_out_of_the_graph():
    """With a requires_grad=False input the product x @ W0 is detached, so W0 (and the bias)
    receive no gradient at all, while both adapter matrices do. (Needle itself still writes
    .grad on the constant input leaf because it is reached via the x @ A path; that is a
    property of Needle's engine, not of this layer, and is harmless.)"""
    layer, x, target = _make_layer()
    x_t = ndl.Tensor(x, dtype="float64", requires_grad=False)
    _needle_loss(layer, x_t, target).backward()
    assert layer.weight.grad is None
    assert layer.bias.grad is None
    assert layer.lora_A.grad is not None and layer.lora_B.grad is not None
    assert getattr(x_t, "grad", None) is not None  # documents Needle behaviour


def test_input_gradient_flows_through_frozen_base_and_adapter():
    """When x requires grad, dL/dx must include both the W0 path and the A B path."""
    layer, x, target = _make_layer(dtype="float64", seed=6)
    got = _needle_grads(layer, x, target, x_requires_grad=True)
    _, _, dx = _analytic_grads(layer, x, target)
    only_base = 2 * (x @ layer.weight.numpy() + layer.bias.numpy()
                     + layer.scaling * (x @ layer.lora_A.numpy() @ layer.lora_B.numpy()) - target) @ layer.weight.numpy().T
    np.testing.assert_allclose(got["x"], dx, rtol=1e-10, atol=1e-10)
    assert not np.allclose(got["x"], only_base)  # adapter path contributes


def test_gradients_through_stacked_lora_layers():
    """Composability: two LoRA layers in sequence, gradient check on the first layer's A."""
    rng = np.random.default_rng(8)
    l1 = LoRALinear(5, 6, rank=2, alpha=2.0, dtype="float64")
    l2 = LoRALinear(6, 3, rank=2, alpha=2.0, dtype="float64")
    for l in (l1, l2):
        l.load_adapter(A=rng.standard_normal(l.lora_A.shape), B=rng.standard_normal(l.lora_B.shape))
    x = rng.standard_normal((4, 5))
    target = rng.standard_normal((4, 3))

    def loss_np(A1):
        y1 = x @ l1.weight.numpy() + l1.bias.numpy() + l1.scaling * (x @ A1 @ l1.lora_B.numpy())
        y2 = y1 @ l2.weight.numpy() + l2.bias.numpy() + l2.scaling * (y1 @ l2.lora_A.numpy() @ l2.lora_B.numpy())
        return np.sum((y2 - target) ** 2)

    x_t = ndl.Tensor(x, dtype="float64", requires_grad=False)
    t = ndl.Tensor(target, dtype="float64", requires_grad=False)
    ((l2(l1(x_t)) - t) ** 2).sum().backward()
    fd = _fd_grad(loss_np, l1.lora_A.numpy(), 1e-6)
    np.testing.assert_allclose(l1.lora_A.grad.numpy(), fd, **F64_TOL)
    # Needle pitfall, documented: the second layer's frozen weight receives a .grad because
    # its input (output of l1) requires grad. It must simply never be handed to an optimizer.
    assert getattr(l2.weight, "grad", None) is not None
    assert l2.weight.requires_grad is False
