"""Dense Mixture-of-LoRA linear layer (Phase 2).

    y = x W0 + b + sum_i g_i(x) * (alpha / r) * x A_i B_i,      g(x) = softmax((x W_g + b_g) / T)

One frozen backbone ``W0`` (and optional frozen ``b``) is shared by ``E``
independent low-rank experts; a :class:`~molora.router.DenseSoftmaxRouter`
produces per-position probabilities over experts and every expert is evaluated
(dense routing). Freezing follows Phase 1 exactly (see ``molora/lora.py``):
frozen base tensors carry ``requires_grad=False`` and ``grad=None`` and the
optimizer must be built from :meth:`DenseMoLoRALinear.trainable_parameters`.

Gate extraction without indexing ops
------------------------------------
Needle has no slicing / gather op, so the i-th gate column is obtained as
``gates @ e_i`` with a constant one-hot ``(E, 1)`` matrix, which is
differentiable through ``matmul`` and broadcasts over any leading dims.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

import needle as ndl
import needle.nn as nn
from needle import init, ops
from needle.autograd import Tensor
from needle.nn import Parameter

from .lora import LoRALinear, _frozen_parameter, _to_tensor, count_parameters
from .router import DenseSoftmaxRouter

__all__ = ["LoRAExpert", "DenseMoLoRALinear"]


class LoRAExpert(nn.Module):
    """A single low-rank update ``(alpha / r) * x A B`` without its own backbone.

    Initialisation matches :class:`molora.LoRALinear`: ``A`` Kaiming-uniform,
    ``B`` zeros (or ``N(0, b_init_std)`` for the *experimental* non-zero variant).
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float,
                 device=None, dtype="float32", b_init_std: float = 0.0) -> None:
        super().__init__()
        self.in_features, self.out_features, self.rank, self.alpha = int(in_features), int(out_features), int(rank), float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A: Parameter = Parameter(init.kaiming_uniform(self.in_features, self.rank, device=device, dtype=dtype))
        if b_init_std > 0:
            B = init.randn(self.rank, self.out_features, std=b_init_std, device=device, dtype=dtype)
        else:
            B = init.zeros(self.rank, self.out_features, device=device, dtype=dtype)
        self.lora_B: Parameter = Parameter(B)
        self.lora_A.grad = None
        self.lora_B.grad = None

    def forward(self, x: Tensor) -> Tensor:
        return ((x @ self.lora_A) @ self.lora_B) * self.scaling

    def delta_weight(self) -> np.ndarray:
        return self.scaling * (self.lora_A.numpy() @ self.lora_B.numpy())


class DenseMoLoRALinear(nn.Module):
    """Frozen linear layer with ``n_experts`` LoRA experts mixed by a softmax router.

    Parameters
    ----------
    in_features, out_features, rank, alpha, bias, device, dtype:
        As in :class:`molora.LoRALinear`.
    n_experts:
        Number of experts ``E >= 1``.
    router_temperature:
        Softmax temperature ``T > 0``.
    router_init:
        ``"xavier"`` (default) or ``"zeros"`` (uniform gates at init).
    b_init_std:
        0.0 (default) gives exact zero adapter contribution at init. A positive
        value is an *experimental* symmetry-breaking variant (Task 2.10) and
        breaks the exact base-equivalence guarantee.
    base_weight, base_bias:
        Optional pretrained base tensors; prefer :meth:`from_linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_experts: int = 4,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        router_temperature: float = 1.0,
        device=None,
        dtype="float32",
        *,
        router_init: str = "xavier",
        b_init_std: float = 0.0,
        base_weight=None,
        base_bias=None,
    ) -> None:
        super().__init__()
        LoRALinear._validate_config(in_features, out_features, rank, alpha)
        if isinstance(n_experts, bool) or not isinstance(n_experts, (int, np.integer)):
            raise TypeError(f"n_experts must be an int, got {type(n_experts).__name__}")
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts}")
        if isinstance(b_init_std, bool) or not isinstance(b_init_std, (int, float, np.floating, np.integer)):
            raise TypeError("b_init_std must be a number")
        if not math.isfinite(b_init_std) or b_init_std < 0:
            raise ValueError(f"b_init_std must be a finite non-negative number, got {b_init_std}")

        self.in_features, self.out_features = int(in_features), int(out_features)
        self.n_experts, self.rank, self.alpha = int(n_experts), int(rank), float(alpha)
        self.scaling = self.alpha / self.rank
        self.router_temperature = float(router_temperature)
        self.b_init_std = float(b_init_std)
        self.device = device if device is not None else ndl.cpu()
        self.dtype = str(np.dtype(dtype))

        # ---- frozen shared backbone ---------------------------------------- #
        if base_weight is None:
            base_weight = init.kaiming_uniform(self.in_features, self.out_features, device=self.device, dtype=self.dtype)
        base_weight = _to_tensor(base_weight, shape=(self.in_features, self.out_features), name="base_weight",
                                 device=self.device, dtype=self.dtype)
        self.weight: Parameter = _frozen_parameter(base_weight)
        if bias:
            if base_bias is None:
                base_bias = init.kaiming_uniform(self.out_features, 1, device=self.device, dtype=self.dtype).reshape(
                    (self.out_features,))
            base_bias = _to_tensor(base_bias, shape=(self.out_features,), name="base_bias",
                                   device=self.device, dtype=self.dtype)
            self.bias: Optional[Parameter] = _frozen_parameter(base_bias)
        else:
            if base_bias is not None:
                raise ValueError("base_bias was given but bias=False")
            self.bias = None

        # ---- router (validates temperature) and experts --------------------- #
        self.router = DenseSoftmaxRouter(self.in_features, self.n_experts, temperature=self.router_temperature,
                                         device=self.device, dtype=self.dtype, weight_init=router_init)
        self.experts = [
            LoRAExpert(self.in_features, self.out_features, self.rank, self.alpha,
                       device=self.device, dtype=self.dtype, b_init_std=self.b_init_std)
            for _ in range(self.n_experts)
        ]
        # constant one-hot selectors e_i : (E, 1), used to read gate column i via matmul
        self._selectors = [
            ndl.Tensor(np.eye(self.n_experts, dtype=np.dtype(self.dtype))[:, i : i + 1], device=self.device,
                       dtype=self.dtype, requires_grad=False)
            for i in range(self.n_experts)
        ]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "DenseMoLoRALinear":
        """Wrap an existing :class:`needle.nn.Linear`, sharing its weight/bias storage."""
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"from_linear expects a needle.nn.Linear, got {type(linear).__name__}")
        weight = linear.weight
        if len(weight.shape) != 2:
            raise ValueError(f"linear.weight must be 2-D, got shape {tuple(weight.shape)}")
        for k in ("in_features", "out_features", "bias", "base_weight", "base_bias", "device", "dtype"):
            if k in kwargs:
                raise TypeError(f"from_linear derives {k!r} from the wrapped layer; do not pass it")
        in_features, out_features = (int(s) for s in weight.shape)
        return cls(in_features, out_features, bias=linear.bias is not None, device=weight.device,
                   dtype=weight.dtype, base_weight=weight, base_bias=linear.bias, **kwargs)

    # ------------------------------------------------------------------ #
    def _base(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            bias_shape = (1,) * (len(out.shape) - 1) + (self.out_features,)
            out = out + ops.broadcast_to(ops.reshape(self.bias, bias_shape), out.shape)
        return out

    def forward_with_routing(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(y, gates)`` where ``gates`` is the graph-bearing ``(..., E)`` router output."""
        if len(x.shape) < 2 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input with shape (..., {self.in_features}) and at least 2 dims, got {tuple(x.shape)}")
        out = self._base(x)
        gates = self.router(x)  # (..., E)
        for i, expert in enumerate(self.experts):
            g_i = gates @ self._selectors[i]  # (..., 1)
            dy_i = expert(x)  # (..., out)
            out = out + ops.broadcast_to(g_i, dy_i.shape) * dy_i
        return out, gates

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_routing(x)[0]

    def router_probabilities(self, x: Tensor) -> np.ndarray:
        """Routing probabilities as a detached numpy array ``(..., E)`` (diagnostics only)."""
        return self.router(x).numpy()

    # ------------------------------------------------------------------ #
    # Parameter access (no duplicates; frozen backbone never included below)
    # ------------------------------------------------------------------ #
    def adapter_parameters(self) -> list[Parameter]:
        """``[A_1, B_1, ..., A_E, B_E]``."""
        return [p for e in self.experts for p in (e.lora_A, e.lora_B)]

    def router_parameters(self) -> list[Parameter]:
        return [self.router.weight, self.router.bias]

    def base_parameters(self) -> list[Parameter]:
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def trainable_parameters(self) -> list[Parameter]:
        params = self.adapter_parameters() + self.router_parameters()
        assert len({id(p) for p in params}) == len(params)
        assert all(p.requires_grad for p in params)
        return params

    def frozen_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters() if not p.requires_grad]

    # ------------------------------------------------------------------ #
    # Accounting
    # ------------------------------------------------------------------ #
    def num_base_parameters(self) -> int:
        return count_parameters(self.base_parameters())

    def num_adapter_parameters(self) -> int:
        """``E * r * (in + out)``."""
        return count_parameters(self.adapter_parameters())

    def num_router_parameters(self) -> int:
        """``in * E + E``."""
        return count_parameters(self.router_parameters())

    def num_trainable_parameters(self) -> int:
        return self.num_adapter_parameters() + self.num_router_parameters()

    def num_total_parameters(self) -> int:
        return count_parameters(self.parameters())

    def trainable_ratio(self) -> float:
        return self.num_trainable_parameters() / self.num_base_parameters()

    def parameter_summary(self) -> dict:
        return {
            "in_features": self.in_features, "out_features": self.out_features, "n_experts": self.n_experts,
            "rank": self.rank, "alpha": self.alpha, "scaling": self.scaling, "router_temperature": self.router_temperature,
            "has_bias": self.bias is not None, "base_parameters": self.num_base_parameters(),
            "adapter_parameters": self.num_adapter_parameters(), "router_parameters": self.num_router_parameters(),
            "trainable_parameters": self.num_trainable_parameters(), "total_parameters": self.num_total_parameters(),
            "trainable_ratio": self.trainable_ratio(),
        }

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def load_expert(self, i: int, A=None, B=None) -> None:
        e = self.experts[i]
        if A is not None:
            e.lora_A.data = _to_tensor(A, shape=(self.in_features, self.rank), name="A", device=self.device, dtype=self.dtype)
        if B is not None:
            e.lora_B.data = _to_tensor(B, shape=(self.rank, self.out_features), name="B", device=self.device, dtype=self.dtype)

    def load_router(self, weight=None, bias=None) -> None:
        if weight is not None:
            self.router.weight.data = _to_tensor(weight, shape=(self.in_features, self.n_experts), name="router weight",
                                                 device=self.device, dtype=self.dtype)
        if bias is not None:
            self.router.bias.data = _to_tensor(bias, shape=(self.n_experts,), name="router bias",
                                               device=self.device, dtype=self.dtype)

    def expert_delta_weights(self) -> list[np.ndarray]:
        """``[(alpha / r) A_i B_i]`` for every expert, as numpy arrays."""
        return [e.delta_weight() for e in self.experts]

    def __repr__(self) -> str:
        return (f"DenseMoLoRALinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"n_experts={self.n_experts}, rank={self.rank}, alpha={self.alpha}, bias={self.bias is not None}, "
                f"router_temperature={self.router_temperature}, dtype={self.dtype})")
