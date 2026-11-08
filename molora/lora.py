"""Native LoRA (Low-Rank Adaptation) linear layer for Needle.

Needle's :class:`needle.nn.Linear` stores its weight with shape
``(in_features, out_features)`` and computes ``y = x @ W + b``. This matches the
row-vector convention used in the LoRA paper, so the adapted layer is

    y = x W0 + b + (alpha / r) * x A B

with ``W0 : (in, out)`` frozen, ``A : (in, r)`` trainable, ``B : (r, out)``
trainable and ``b : (out,)`` frozen. ``B`` is zero-initialised so the layer
reproduces the base layer exactly at construction time.

Freezing semantics in Needle
----------------------------
Needle's optimizers update every tensor in the list they are given, and its
autograd engine writes ``.grad`` on every node it visits regardless of the
node's ``requires_grad`` flag (the flag only controls whether a *new* node is
detached from the graph). Setting ``requires_grad=False`` on ``W0`` therefore
keeps it out of the graph only when the layer input does not itself require a
gradient; inside a deeper network ``W0.grad`` will still be populated. The
only robust guarantee is to give the optimizer *only* the adapter parameters,
which is what :meth:`LoRALinear.adapter_parameters` /
:func:`trainable_parameters` are for.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

import needle as ndl
import needle.nn as nn
from needle import init, ops
from needle.autograd import Tensor
from needle.nn import Parameter

__all__ = [
    "LoRALinear",
    "freeze_module",
    "trainable_parameters",
    "frozen_parameters",
    "count_parameters",
]


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
def count_parameters(params: Iterable[Tensor]) -> int:
    """Total number of scalar entries across ``params``."""
    return int(sum(int(np.prod(p.shape)) for p in params))


def freeze_module(module: nn.Module) -> nn.Module:
    """Mark every parameter of ``module`` as frozen (``requires_grad=False``) in place."""
    for p in module.parameters():
        p.requires_grad = False
    return module


def trainable_parameters(module: nn.Module) -> list[Tensor]:
    """Parameters of ``module`` with ``requires_grad=True`` (what an optimizer should get)."""
    return [p for p in module.parameters() if p.requires_grad]


def frozen_parameters(module: nn.Module) -> list[Tensor]:
    """Parameters of ``module`` with ``requires_grad=False``."""
    return [p for p in module.parameters() if not p.requires_grad]


def _frozen_parameter(t: Tensor) -> Parameter:
    """Wrap ``t`` as a frozen Parameter that *shares* its underlying storage (no copy).

    Sharing is safe because Needle optimizers never mutate ``cached_data`` in
    place; they rebind ``p.data`` to a freshly computed tensor.
    """
    p = Parameter(t, device=t.device, dtype=t.dtype, requires_grad=False)
    # Needle's optimizers read ``p.grad`` unconditionally and the attribute only
    # exists once autograd has written it. Define it so a frozen parameter that
    # never enters the graph is skipped (``grad is None``) instead of raising.
    p.grad = None
    return p


def _to_tensor(value, *, shape: tuple[int, ...], name: str, device, dtype) -> Tensor:
    """Coerce a numpy array / Tensor to a Needle Tensor of the given shape and dtype."""
    if isinstance(value, Tensor):
        t = value
        if tuple(t.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {shape}, got {tuple(t.shape)}")
        if np.dtype(t.dtype) != np.dtype(dtype):
            t = ndl.Tensor(t.numpy().astype(np.dtype(dtype)), device=device, dtype=dtype, requires_grad=False)
        return t
    arr = np.asarray(value)
    if arr.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return ndl.Tensor(arr.astype(np.dtype(dtype), copy=False), device=device, dtype=dtype, requires_grad=False)


# --------------------------------------------------------------------------- #
# LoRALinear
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """A frozen Needle linear layer with a trainable low-rank update.

    Parameters
    ----------
    in_features, out_features:
        Layer dimensions. The frozen weight has shape ``(in_features, out_features)``.
    rank:
        LoRA rank ``r`` (``0 < r <= min(in_features, out_features)``).
    alpha:
        LoRA scaling numerator; the update is multiplied by ``alpha / rank``.
    bias:
        Whether the (frozen) base layer has a bias.
    device, dtype:
        Passed to Needle initialisers. CPU / float32 by default.
    base_weight, base_bias:
        Optional pretrained base tensors (numpy arrays or Needle tensors). When
        given they are used *as is* (shared storage for Tensors) instead of a
        fresh Needle initialisation. Prefer :meth:`from_linear` for wrapping an
        existing :class:`needle.nn.Linear`.

    Attributes
    ----------
    weight, bias : Parameter
        Frozen base parameters (``requires_grad=False``).
    lora_A, lora_B : Parameter
        Trainable adapter matrices of shape ``(in, r)`` and ``(r, out)``.
    scaling : float
        ``alpha / rank``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        device=None,
        dtype="float32",
        *,
        base_weight=None,
        base_bias=None,
    ) -> None:
        super().__init__()
        self._validate_config(in_features, out_features, rank, alpha)

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.device = device if device is not None else ndl.cpu()
        self.dtype = str(np.dtype(dtype))

        # ---- frozen base ------------------------------------------------- #
        if base_weight is None:
            # Same initialisation as needle.nn.Linear.
            base_weight = init.kaiming_uniform(self.in_features, self.out_features, device=self.device, dtype=self.dtype)
        base_weight = _to_tensor(base_weight, shape=(self.in_features, self.out_features), name="base_weight",
                                 device=self.device, dtype=self.dtype)
        self.weight: Parameter = _frozen_parameter(base_weight)

        if bias:
            if base_bias is None:
                base_bias = init.kaiming_uniform(self.out_features, 1, device=self.device, dtype=self.dtype).reshape(
                    (self.out_features,)
                )
            base_bias = _to_tensor(base_bias, shape=(self.out_features,), name="base_bias",
                                   device=self.device, dtype=self.dtype)
            self.bias: Optional[Parameter] = _frozen_parameter(base_bias)
        else:
            if base_bias is not None:
                raise ValueError("base_bias was given but bias=False")
            self.bias = None

        # ---- trainable adapter ------------------------------------------- #
        # A: Kaiming-uniform (bound sqrt(6 / in_features)), B: zeros, so the
        # initial LoRA correction A @ B is exactly zero.
        self.lora_A: Parameter = Parameter(
            init.kaiming_uniform(self.in_features, self.rank, device=self.device, dtype=self.dtype)
        )
        self.lora_B: Parameter = Parameter(init.zeros(self.rank, self.out_features, device=self.device, dtype=self.dtype))
        self.lora_A.grad = None
        self.lora_B.grad = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_config(in_features, out_features, rank, alpha) -> None:
        for name, val in (("in_features", in_features), ("out_features", out_features), ("rank", rank)):
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
                raise TypeError(f"{name} must be an int, got {type(val).__name__}")
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank ({rank}) must not exceed min(in_features, out_features) = {min(in_features, out_features)}; "
                "a larger rank is not a low-rank adapter"
            )
        if not isinstance(alpha, (int, float, np.floating, np.integer)) or isinstance(alpha, bool):
            raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be a positive finite number, got {alpha}")

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> "LoRALinear":
        """Wrap an existing :class:`needle.nn.Linear` without copying its weights.

        The base weight and bias tensors share storage with ``linear``'s
        parameters. The returned layer produces exactly the same output as
        ``linear`` until the adapter is trained.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"from_linear expects a needle.nn.Linear, got {type(linear).__name__}")
        weight = linear.weight
        if len(weight.shape) != 2:
            raise ValueError(f"linear.weight must be 2-D, got shape {tuple(weight.shape)}")
        in_features, out_features = (int(s) for s in weight.shape)
        return cls(
            in_features,
            out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            device=weight.device,
            dtype=weight.dtype,
            base_weight=weight,
            base_bias=linear.bias,
        )

    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) < 2 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input with shape (..., {self.in_features}) and at least 2 dims, got {tuple(x.shape)}"
            )
        out = x @ self.weight
        if self.bias is not None:
            bias_shape = (1,) * (len(out.shape) - 1) + (self.out_features,)
            out = out + ops.broadcast_to(ops.reshape(self.bias, bias_shape), out.shape)
        lora = (x @ self.lora_A) @ self.lora_B
        return out + lora * self.scaling

    # ------------------------------------------------------------------ #
    # Parameter access
    # ------------------------------------------------------------------ #
    def adapter_parameters(self) -> list[Parameter]:
        """The trainable adapter matrices ``[A, B]`` (what the optimizer should receive)."""
        return [self.lora_A, self.lora_B]

    def base_parameters(self) -> list[Parameter]:
        """The frozen base parameters (weight and, if present, bias)."""
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def trainable_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters() if not p.requires_grad]

    # ------------------------------------------------------------------ #
    # Parameter accounting
    # ------------------------------------------------------------------ #
    def num_base_parameters(self) -> int:
        """Frozen parameter count: ``in*out`` plus ``out`` if a bias is present."""
        return count_parameters(self.base_parameters())

    def num_adapter_parameters(self) -> int:
        """Trainable LoRA parameter count ``r * (in + out)``; frozen bias excluded."""
        return count_parameters(self.adapter_parameters())

    def num_total_parameters(self) -> int:
        return count_parameters(self.parameters())

    def trainable_ratio(self) -> float:
        """Trainable adapter parameters divided by frozen base parameters."""
        return self.num_adapter_parameters() / self.num_base_parameters()

    def parameter_summary(self) -> dict:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": self.scaling,
            "has_bias": self.bias is not None,
            "base_parameters": self.num_base_parameters(),
            "adapter_parameters": self.num_adapter_parameters(),
            "total_parameters": self.num_total_parameters(),
            "trainable_ratio": self.trainable_ratio(),
        }

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def load_adapter(self, A=None, B=None) -> None:
        """Overwrite the adapter matrices from numpy arrays or Tensors (shape-checked)."""
        if A is not None:
            t = _to_tensor(A, shape=(self.in_features, self.rank), name="A", device=self.device, dtype=self.dtype)
            self.lora_A.data = t
        if B is not None:
            t = _to_tensor(B, shape=(self.rank, self.out_features), name="B", device=self.device, dtype=self.dtype)
            self.lora_B.data = t

    def delta_weight(self) -> np.ndarray:
        """The low-rank update ``(alpha / r) * A @ B`` as a numpy array."""
        return self.scaling * (self.lora_A.numpy() @ self.lora_B.numpy())

    def effective_weight(self) -> np.ndarray:
        """``W0 + (alpha / r) * A @ B`` as a numpy array (useful for tests and merging)."""
        return self.weight.numpy() + self.delta_weight()

    def effective_bias(self) -> Optional[np.ndarray]:
        return None if self.bias is None else self.bias.numpy()

    def __repr__(self) -> str:
        return (
            f"LoRALinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, bias={self.bias is not None}, dtype={self.dtype})"
        )
