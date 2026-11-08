"""Phase 0 — verify the Needle framework is importable and functional on CPU."""
import sys

import numpy as np
import pytest

import molora  # noqa: F401
import needle as ndl
import needle.nn as nn
from needle.nn import Parameter


def _t(arr, requires_grad=False, dtype="float32"):
    return ndl.Tensor(np.asarray(arr), dtype=dtype, requires_grad=requires_grad)


def test_needle_imports_and_reports_cpu_backend():
    info = molora.needle_info()
    assert info["array_api"] == "numpy"
    assert "needle.cpu()" in info["devices"]
    assert ndl.Tensor is not None and nn.Linear is not None and ndl.optim.Adam is not None


def test_no_gpu_required():
    """The numpy backend exposes only a CPU device, and it is enabled."""
    devices = ndl.all_devices()
    assert devices == [ndl.cpu()]
    assert ndl.cpu().enabled()


def test_pytorch_is_not_used():
    """Framework fidelity: molora must not pull in torch."""
    assert "torch" not in sys.modules


def test_tensor_creation_roundtrip():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = _t(x)
    assert t.shape == (2, 3)
    assert t.dtype == np.float32
    assert t.device == ndl.cpu()
    np.testing.assert_array_equal(t.numpy(), x)


def test_matmul_matches_numpy():
    np.random.seed(1)
    a = np.random.randn(4, 5).astype(np.float32)
    b = np.random.randn(5, 3).astype(np.float32)
    out = (_t(a) @ _t(b)).numpy()
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)


def test_batched_matmul_with_2d_rhs():
    np.random.seed(2)
    a = np.random.randn(2, 4, 5).astype(np.float32)
    b = np.random.randn(5, 3).astype(np.float32)
    out = (_t(a) @ _t(b)).numpy()
    assert out.shape == (2, 4, 3)
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)


def test_backward_matches_analytic_gradient():
    """f(x) = sum(3 * x^2 + 2 * x)  =>  df/dx = 6 x + 2."""
    np.random.seed(3)
    x_np = np.random.randn(3, 4).astype(np.float32)
    x = _t(x_np, requires_grad=True)
    f = ((x ** 2) * 3.0 + x * 2.0).sum()
    f.backward()
    np.testing.assert_allclose(f.numpy(), np.sum(3 * x_np**2 + 2 * x_np), rtol=1e-5)
    np.testing.assert_allclose(x.grad.numpy(), 6 * x_np + 2, rtol=1e-5, atol=1e-6)


def test_matmul_backward_matches_analytic_gradient():
    """L = sum(x @ W) => dL/dW = x^T 1, dL/dx = 1 W^T."""
    np.random.seed(4)
    x_np = np.random.randn(5, 3).astype(np.float32)
    w_np = np.random.randn(3, 2).astype(np.float32)
    x, w = _t(x_np, True), _t(w_np, True)
    (x @ w).sum().backward()
    np.testing.assert_allclose(w.grad.numpy(), x_np.T @ np.ones((5, 2), np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x.grad.numpy(), np.ones((5, 2), np.float32) @ w_np.T, rtol=1e-5, atol=1e-5)


def test_linear_layer_instantiation_and_forward():
    lin = nn.Linear(7, 3)
    assert lin.weight.shape == (7, 3)  # Needle convention: (in, out)
    assert lin.bias.shape == (3,)
    x = _t(np.random.randn(4, 7).astype(np.float32))
    y = lin(x)
    assert y.shape == (4, 3)
    np.testing.assert_allclose(y.numpy(), x.numpy() @ lin.weight.numpy() + lin.bias.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("opt_name", ["SGD", "Adam"])
def test_optimizer_step_updates_parameters(opt_name):
    lin = nn.Linear(4, 2)
    before = lin.weight.numpy().copy()
    opt = getattr(ndl.optim, opt_name)(lin.parameters(), lr=0.1)
    x = _t(np.random.randn(8, 4).astype(np.float32))
    loss = (lin(x) ** 2).sum()
    loss.backward()
    assert lin.weight.grad is not None
    opt.step()
    after = lin.weight.numpy()
    assert not np.array_equal(before, after)
    assert np.all(np.isfinite(after))
    opt.reset_grad()
    assert lin.weight.grad is None


def test_parameters_can_be_identified_and_controlled():
    lin = nn.Linear(4, 2)
    params = lin.parameters()
    assert len(params) == 2 and all(isinstance(p, Parameter) for p in params)
    # requires_grad is a plain attribute we can flip to mark a parameter frozen
    lin.weight.requires_grad = False
    assert [p.requires_grad for p in lin.parameters()] == [False, True]
    # a Parameter built from a Tensor shares storage (no copy) and can be created frozen
    frozen = Parameter(lin.bias, requires_grad=False)
    assert frozen.requires_grad is False
    assert np.shares_memory(frozen.cached_data, lin.bias.cached_data)


def test_frozen_weight_receives_no_grad_when_input_is_constant():
    """If neither operand requires grad, Needle detaches the result: W.grad stays unset."""
    w = _t(np.random.randn(3, 2).astype(np.float32), requires_grad=False)
    a = _t(np.random.randn(3, 2).astype(np.float32), requires_grad=True)
    x = _t(np.random.randn(4, 3).astype(np.float32))
    (x @ w + x @ a).sum().backward()
    assert getattr(w, "grad", None) is None
    assert a.grad is not None


def test_requires_grad_false_is_not_sufficient_in_needle():
    """Documents the pitfall motivating adapter_parameters(): when the *input* requires
    grad, Needle still writes .grad on a requires_grad=False weight, and its optimizers
    would update it if handed that tensor."""
    w = _t(np.random.randn(3, 2).astype(np.float32), requires_grad=False)
    x = _t(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
    (x @ w).sum().backward()
    assert getattr(w, "grad", None) is not None
    before = w.numpy().copy()
    ndl.optim.SGD([w], lr=0.1).step()
    assert not np.array_equal(before, w.numpy())


def test_float64_tensors_supported():
    x = ndl.Tensor(np.random.randn(2, 3), dtype="float64", requires_grad=True)
    w = ndl.Tensor(np.random.randn(3, 2), dtype="float64", requires_grad=True)
    (x @ w).sum().backward()
    assert x.grad.dtype == np.float64 and w.grad.dtype == np.float64
