"""Masked top-k router (Phase 3A reference).

``SparseTopKRouter`` wraps a :class:`molora.router.DenseSoftmaxRouter`: it
computes all E softmax probabilities, selects the top-k experts per routed
position, masks the rest and renormalises the selected probabilities to sum to
one. It owns the router parameters (``weight``, ``bias`` of ``self.dense``) and
adds none of its own.

``router_grad_mode``:
    ``"renormalized"`` exact gradient of the renormalised selected weights with
        the selection held fixed (for k = 1 this gradient is identically 0);
    ``"straight_through"`` forward value identical, backward pass as if the
        weight were the dense probability of the selected expert (surrogate).
"""
from __future__ import annotations

import numpy as np

import needle.nn as nn
from needle.autograd import Tensor

from .ops.topk import TopKSelection, _validate_k, sparse_renormalized_gates
from .router import DenseSoftmaxRouter

__all__ = ["SparseTopKRouter", "ROUTER_GRAD_MODES"]

ROUTER_GRAD_MODES = ("renormalized", "straight_through")


class SparseTopKRouter(nn.Module):
    def __init__(self, input_dim: int, n_experts: int, top_k: int, temperature: float = 1.0, device=None,
                 dtype="float32", weight_init: str = "xavier", router_grad_mode: str = "renormalized") -> None:
        super().__init__()
        self.dense = DenseSoftmaxRouter(input_dim, n_experts, temperature=temperature, device=device, dtype=dtype,
                                        weight_init=weight_init)
        self.top_k = _validate_k(top_k, self.dense.n_experts)
        if router_grad_mode not in ROUTER_GRAD_MODES:
            raise ValueError(f"router_grad_mode must be one of {ROUTER_GRAD_MODES}, got {router_grad_mode!r}")
        self.router_grad_mode = router_grad_mode

    # convenience accessors so callers can treat it like the dense router
    @property
    def weight(self):
        return self.dense.weight

    @property
    def bias(self):
        return self.dense.bias

    @property
    def n_experts(self) -> int:
        return self.dense.n_experts

    @property
    def input_dim(self) -> int:
        return self.dense.input_dim

    @property
    def temperature(self) -> float:
        return self.dense.temperature

    def dense_probabilities(self, x: Tensor) -> Tensor:
        return self.dense(x)

    def route(self, x: Tensor) -> TopKSelection:
        """Full routing result: sparse weights (graph-bearing), dense probs, indices and mask."""
        return sparse_renormalized_gates(self.dense(x), self.top_k,
                                         straight_through=self.router_grad_mode == "straight_through")

    def forward(self, x: Tensor) -> Tensor:
        """Sparse renormalised routing weights ``(..., E)`` with exactly top_k non-zeros per row."""
        return self.route(x).weights

    def num_parameters(self) -> int:
        return self.dense.num_parameters()

    def __repr__(self) -> str:
        return (f"SparseTopKRouter(input_dim={self.input_dim}, n_experts={self.n_experts}, top_k={self.top_k}, "
                f"temperature={self.temperature}, router_grad_mode={self.router_grad_mode!r})")
