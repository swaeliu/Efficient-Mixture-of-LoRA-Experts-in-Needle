"""Sparse (top-k) Mixture-of-LoRA linear layer (Phase 3).

    y = x W0 + b + sum_{i in S_k(x)} g~_i(x) (alpha / r) x A_i B_i
    S_k(x) = TopK(softmax((x W_g + b_g) / T), k),   g~_i = g_i / sum_{j in S_k} g_j

Two execution modes compute the same function:

``"reference"``  every expert is evaluated on every token and multiplied by
                 the sparse weights (zeros for unselected experts). Trusted
                 numerical baseline; no efficiency.
``"dispatch"``   (Phase 3B) tokens are gathered per expert, only selected
                 experts run on only their tokens, results are scatter-added.

The layer subclasses :class:`molora.DenseMoLoRALinear`, so the frozen shared
backbone, the ``LoRAExpert`` list, ``from_linear``, parameter accounting and
the Phase 1 freezing strategy are inherited unchanged. ``self.router`` is a
:class:`molora.sparse_router.SparseTopKRouter` (which owns the dense router).
Leading input dimensions are flattened to a token axis internally.
"""
from __future__ import annotations

import time

import numpy as np

import needle as ndl
from needle import ops
from needle.autograd import Tensor
from needle.nn import Parameter

from .dense_molora import DenseMoLoRALinear
from .ops.dispatch import gather_rows, scatter_add_rows
from .ops.topk import TopKSelection, _validate_k
from .sparse_router import ROUTER_GRAD_MODES, SparseTopKRouter

__all__ = ["SparseMoLoRALinear", "EXECUTION_MODES", "load_balance_loss"]

EXECUTION_MODES = ("reference", "dispatch")


class SparseMoLoRALinear(DenseMoLoRALinear):
    """Frozen linear layer with ``n_experts`` LoRA experts of which ``top_k`` run per token."""

    def __init__(self, in_features, out_features, n_experts=4, rank=8, alpha=16.0, bias=True, top_k=2,
                 router_temperature=1.0, execution_mode="reference", router_grad_mode="renormalized",
                 device=None, dtype="float32", *, router_init="xavier", b_init_std=0.0, base_weight=None,
                 base_bias=None) -> None:
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"execution_mode must be one of {EXECUTION_MODES}, got {execution_mode!r}")
        if router_grad_mode not in ROUTER_GRAD_MODES:
            raise ValueError(f"router_grad_mode must be one of {ROUTER_GRAD_MODES}, got {router_grad_mode!r}")
        super().__init__(in_features, out_features, n_experts=n_experts, rank=rank, alpha=alpha, bias=bias,
                         router_temperature=router_temperature, device=device, dtype=dtype, router_init=router_init,
                         b_init_std=b_init_std, base_weight=base_weight, base_bias=base_bias)
        self.top_k = _validate_k(top_k, self.n_experts)
        self.execution_mode = execution_mode
        self.router_grad_mode = router_grad_mode
        # replace the dense router (same parameters, same init draw) by the sparse wrapper
        dense_router = self.router
        self.router = SparseTopKRouter(self.in_features, self.n_experts, self.top_k, temperature=self.router_temperature,
                                       device=self.device, dtype=self.dtype, weight_init=router_init,
                                       router_grad_mode=router_grad_mode)
        self.router.dense.weight = dense_router.weight  # keep the Parameter objects created by the parent
        self.router.dense.bias = dense_router.bias
        self.last_stats: dict | None = None  # plain numbers from the most recent forward (never tensors)
        self.profile = False  # when True, forward() records wall-time per stage in self.last_timing
        self.last_timing: dict | None = None

    # ------------------------------------------------------------------ #
    def router_parameters(self) -> list[Parameter]:
        return [self.router.dense.weight, self.router.dense.bias]

    def load_router(self, weight=None, bias=None) -> None:
        from .lora import _to_tensor  # local import to keep module namespace small

        if weight is not None:
            self.router.dense.weight.data = _to_tensor(weight, shape=(self.in_features, self.n_experts),
                                                       name="router weight", device=self.device, dtype=self.dtype)
        if bias is not None:
            self.router.dense.bias.data = _to_tensor(bias, shape=(self.n_experts,), name="router bias",
                                                     device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------ #
    def _flatten(self, x: Tensor):
        if len(x.shape) < 2 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input with shape (..., {self.in_features}) and at least 2 dims, got {tuple(x.shape)}")
        lead = tuple(x.shape[:-1])
        n_tokens = int(np.prod(lead))
        x_flat = x if len(x.shape) == 2 else ops.reshape(x, (n_tokens, self.in_features))
        return x_flat, lead, n_tokens

    def _expert_flops(self, n_tokens: int) -> int:
        """Multiply-add count (x2 for FLOPs) of one expert on n_tokens rows."""
        return 2 * n_tokens * (self.in_features * self.rank + self.rank * self.out_features)

    def _reference_mix(self, x_flat: Tensor, out: Tensor, sel: TopKSelection) -> Tensor:
        for i, expert in enumerate(self.experts):
            w_i = sel.weights @ self._selectors[i]  # (N, 1), zero for tokens that did not select i
            dy_i = expert(x_flat)  # (N, out) -- every expert runs on every token
            out = out + ops.broadcast_to(w_i, dy_i.shape) * dy_i
        return out

    def _dispatch_mix(self, x_flat: Tensor, out: Tensor, sel: TopKSelection, n_tokens: int):
        """Run each expert only on the tokens that selected it; scatter-add the weighted results.

        Returns ``(out, executed_rows)`` where ``executed_rows[i]`` is the number of token
        rows expert i actually processed (0 => its matmuls were skipped).
        """
        contribs, idx_list, executed = [], [], []
        t_gather = t_expert = 0.0
        for i, expert in enumerate(self.experts):
            tok = np.flatnonzero(sel.mask[:, i])
            if tok.size == 0:
                executed.append(0)
                continue  # inactive expert: no gather, no matmul
            t0 = time.perf_counter() if self.profile else 0.0
            x_i = gather_rows(x_flat, tok)  # (M_i, in)
            w_i = gather_rows(sel.weights, tok) @ self._selectors[i]  # (M_i, 1) selected routing weights
            if self.profile:
                t1 = time.perf_counter()
                t_gather += t1 - t0
            dy_i = expert(x_i)  # (M_i, out): the only expert matmuls executed
            if self.profile:
                t2 = time.perf_counter()
                t_expert += t2 - t1
            contribs.append(ops.broadcast_to(w_i, dy_i.shape) * dy_i)
            idx_list.append(tok)
            executed.append(int(tok.size))
        t0 = time.perf_counter() if self.profile else 0.0
        if contribs:
            # each token selects an expert at most once, so indices are unique within each expert
            out = out + scatter_add_rows(contribs, idx_list, n_tokens, assume_unique=True)
        if self.profile:
            self._timing_acc["dispatch_s"] += t_gather + (time.perf_counter() - t0)
            self._timing_acc["expert_s"] += t_expert
        return out, executed

    def forward_with_routing(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(y, sparse_weights)``; ``sparse_weights`` has shape ``(..., E)`` and is graph-bearing."""
        out, sel, lead = self._forward_core(x)
        weights = sel.weights if len(lead) == 1 else ops.reshape(sel.weights, lead + (self.n_experts,))
        return out, weights

    def forward_with_selection(self, x: Tensor) -> tuple[Tensor, TopKSelection]:
        """Return ``(y, selection)`` where ``selection`` is the token-flattened :class:`TopKSelection`
        (graph-bearing weights and dense probabilities, NumPy indices and mask) -- e.g. for an
        auxiliary balance loss. Nothing is cached on the module."""
        out, sel, _ = self._forward_core(x)
        return out, sel

    def _forward_core(self, x: Tensor):
        prof = self.profile
        if prof:
            self._timing_acc = {"backbone_s": 0.0, "router_s": 0.0, "expert_s": 0.0, "dispatch_s": 0.0}
            t0 = time.perf_counter()
        x_flat, lead, n_tokens = self._flatten(x)
        out = self._base(x_flat)
        if prof:
            t1 = time.perf_counter()
            self._timing_acc["backbone_s"] = t1 - t0
        sel = self.router.route(x_flat)
        if prof:
            t2 = time.perf_counter()
            self._timing_acc["router_s"] = t2 - t1
        tokens_per_expert = sel.mask.sum(axis=0).astype(int).tolist()
        if self.execution_mode == "reference":
            t3 = time.perf_counter() if prof else 0.0
            out = self._reference_mix(x_flat, out, sel)
            if prof:
                self._timing_acc["expert_s"] = time.perf_counter() - t3
            executed = [n_tokens] * self.n_experts
        else:
            out, executed = self._dispatch_mix(x_flat, out, sel, n_tokens)
        self.last_stats = self._make_stats(n_tokens, tokens_per_expert, executed)
        if prof:
            self._timing_acc["forward_total_s"] = time.perf_counter() - t0
            self.last_timing = dict(self._timing_acc)
        if len(lead) != 1:
            out = ops.reshape(out, lead + (self.out_features,))
        return out, sel, lead

    def _make_stats(self, n_tokens: int, tokens_per_expert: list[int], executed_rows: list[int]) -> dict:
        """Expert-work accounting (expert matmuls only; backbone, router and dispatch excluded)."""
        active = [r > 0 for r in executed_rows]
        dense_rows = self.n_experts * n_tokens
        executed_total = int(sum(executed_rows))
        return {
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "execution_mode": self.execution_mode,
            "n_tokens": n_tokens,
            "selected_experts_per_token": self.top_k,
            "tokens_per_expert": tokens_per_expert,
            "expert_rows_executed": executed_rows,  # rows each expert's two matmuls were run on
            "expert_matmuls_executed": 2 * int(sum(active)),
            "experts_skipped": int(self.n_experts - sum(active)),
            "expert_flops_executed": 2 * executed_total * (self.in_features * self.rank + self.rank * self.out_features),
            "expert_flops_dense": 2 * dense_rows * (self.in_features * self.rank + self.rank * self.out_features),
            "expert_flop_fraction_vs_dense": executed_total / dense_rows if dense_rows else 0.0,
            "theoretical_flop_fraction": self.top_k / self.n_experts,
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_routing(x)[0]

    # ------------------------------------------------------------------ #
    def router_probabilities(self, x: Tensor) -> np.ndarray:
        """Dense (pre-selection) routing probabilities as numpy, shape ``(..., E)``."""
        x_flat, lead, _ = self._flatten(x)
        p = self.router.dense_probabilities(x_flat).numpy()
        return p.reshape(lead + (self.n_experts,))

    def routing_diagnostics(self, x: Tensor) -> dict:
        """Dense probs, sparse weights, selected indices, utilisation, load and entropy (numpy only)."""
        x_flat, lead, n_tokens = self._flatten(x)
        sel = self.router.route(x_flat)
        p = sel.dense_probs.numpy().astype(np.float64)
        w = sel.weights.numpy().astype(np.float64)
        ent = -(p * np.log(np.clip(p, 1e-300, None))).sum(-1)
        load = sel.mask.sum(axis=0)
        return {
            "dense_probs": p.reshape(lead + (self.n_experts,)),
            "sparse_weights": w.reshape(lead + (self.n_experts,)),
            "indices": sel.indices.reshape(lead + (self.top_k,)),
            "mask": sel.mask.reshape(lead + (self.n_experts,)),
            "tokens_per_expert": load.astype(int).tolist(),
            "utilization": (load / n_tokens).tolist(),  # fraction of tokens that selected each expert
            "mean_sparse_weight": w.mean(0).tolist(),
            "entropy_nats_mean": float(ent.mean()),
            "max_entropy_nats": float(np.log(self.n_experts)),
            "n_tokens": n_tokens,
        }

    def parameter_summary(self) -> dict:
        s = super().parameter_summary()
        s.update({"top_k": self.top_k, "execution_mode": self.execution_mode, "router_grad_mode": self.router_grad_mode})
        return s

    def __repr__(self) -> str:
        return (f"SparseMoLoRALinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"n_experts={self.n_experts}, top_k={self.top_k}, rank={self.rank}, alpha={self.alpha}, "
                f"bias={self.bias is not None}, execution_mode={self.execution_mode!r}, "
                f"router_grad_mode={self.router_grad_mode!r}, dtype={self.dtype})")


def load_balance_loss(sel: TopKSelection) -> Tensor:
    """Switch-Transformer style auxiliary balance loss ``E * sum_i f_i * P_i``.

    ``f_i`` is the (constant) fraction of tokens that selected expert i and ``P_i``
    the mean dense routing probability of expert i (differentiable). The loss is
    minimised (value 1) when both are uniform. ``sel`` must come from a 2-D
    (token-flattened) routing call. Optional; not used unless an experiment
    enables it explicitly.
    """
    probs = sel.dense_probs
    n_tokens, E = probs.shape
    f = sel.mask.sum(axis=0) / (n_tokens * sel.mask.sum(axis=1)[0])  # fraction of assignments per expert
    f_t = ndl.Tensor(f.astype(probs.numpy().dtype), device=probs.device, dtype=str(np.dtype(probs.numpy().dtype)),
                     requires_grad=False)
    P = ops.summation(probs, axes=(0,)) / n_tokens  # (E,)
    return ops.summation(P * f_t) * E
