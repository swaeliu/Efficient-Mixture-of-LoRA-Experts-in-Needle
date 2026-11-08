"""Dense softmax router for mixture-of-LoRA layers (Phase 2).

``g(x) = softmax((x W_g + b_g) / T)`` over the expert axis, built only from
Needle ops. Needle's ``logsoftmax`` op normalises over axis 1 unconditionally,
so for inputs with arbitrary leading dimensions the softmax is written as
``exp(z - logsumexp(z, axes=(-1,)))``, which is numerically stable (max is
subtracted inside ``logsumexp``) and differentiable through existing ops.
"""
from __future__ import annotations

import math

import numpy as np

import needle as ndl
import needle.nn as nn
from needle import init, ops
from needle.autograd import Tensor
from needle.nn import Parameter

__all__ = ["softmax", "DenseSoftmaxRouter"]


def softmax(z: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax over ``axis`` using Needle ops (differentiable)."""
    ndim = len(z.shape)
    axis = axis % ndim
    keep_shape = tuple(1 if i == axis else s for i, s in enumerate(z.shape))
    lse = ops.logsumexp(z, axes=(axis,))
    lse = ops.broadcast_to(ops.reshape(lse, keep_shape), z.shape)
    return ops.exp(z - lse)


class DenseSoftmaxRouter(nn.Module):
    """Linear projection followed by a temperature-scaled softmax over experts.

    Input ``(..., input_dim)`` -> probabilities ``(..., n_experts)`` that sum to
    one along the last axis for every routed position (token granularity).

    Parameters
    ----------
    input_dim, n_experts:
        Projection size ``W_g: (input_dim, n_experts)``, ``b_g: (n_experts,)``.
    temperature:
        Positive scalar ``T``; logits are divided by ``T`` before the softmax.
    weight_init:
        ``"xavier"`` (default; Needle ``xavier_uniform``) or ``"zeros"`` (uniform
        routing at initialisation).
    """

    def __init__(self, input_dim: int, n_experts: int, temperature: float = 1.0,
                 device=None, dtype="float32", weight_init: str = "xavier") -> None:
        super().__init__()
        for name, val in (("input_dim", input_dim), ("n_experts", n_experts)):
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
                raise TypeError(f"{name} must be an int, got {type(val).__name__}")
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float, np.floating, np.integer)):
            raise TypeError(f"temperature must be a number, got {type(temperature).__name__}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be a positive finite number, got {temperature}")
        if weight_init not in ("xavier", "zeros"):
            raise ValueError(f"weight_init must be 'xavier' or 'zeros', got {weight_init!r}")

        self.input_dim = int(input_dim)
        self.n_experts = int(n_experts)
        self.temperature = float(temperature)
        self.device = device if device is not None else ndl.cpu()
        self.dtype = str(np.dtype(dtype))

        if weight_init == "xavier":
            w = init.xavier_uniform(self.input_dim, self.n_experts, device=self.device, dtype=self.dtype)
        else:
            w = init.zeros(self.input_dim, self.n_experts, device=self.device, dtype=self.dtype)
        self.weight: Parameter = Parameter(w)
        self.bias: Parameter = Parameter(init.zeros(self.n_experts, device=self.device, dtype=self.dtype))
        self.weight.grad = None
        self.bias.grad = None

    def logits(self, x: Tensor) -> Tensor:
        if len(x.shape) < 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input (..., {self.input_dim}) with >= 2 dims, got {tuple(x.shape)}")
        z = x @ self.weight
        bias_shape = (1,) * (len(z.shape) - 1) + (self.n_experts,)
        z = z + ops.broadcast_to(ops.reshape(self.bias, bias_shape), z.shape)
        if self.temperature != 1.0:
            z = z / self.temperature
        return z

    def forward(self, x: Tensor) -> Tensor:
        """Routing probabilities ``(..., n_experts)``; graph-bearing, nothing is cached."""
        return softmax(self.logits(x), axis=-1)

    def num_parameters(self) -> int:
        return self.input_dim * self.n_experts + self.n_experts

    def __repr__(self) -> str:
        return (f"DenseSoftmaxRouter(input_dim={self.input_dim}, n_experts={self.n_experts}, "
                f"temperature={self.temperature}, dtype={self.dtype})")
