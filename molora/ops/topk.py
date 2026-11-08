"""Top-k expert selection with renormalised sparse gates.

Selection (which experts) is discrete metadata computed in NumPy from the
*values* of the router output; it is not differentiable and is held fixed for
one forward/backward evaluation. The sparse weights themselves are computed
with Needle ops so that gradients flow through the selected probabilities:

    m        = topk_mask(g, k)                     constant {0,1}, exactly k ones per row
    g_sel    = g * m                               Needle EWiseMul with a constant
    g_tilde  = g_sel / broadcast(sum(g_sel, -1))   Needle ops

Gradient semantics
------------------
For fixed selection S, g_tilde_i = g_i / sum_{j in S} g_j for i in S and 0
otherwise. Autograd yields the exact derivative of this piecewise-smooth
function with respect to g on the open region where the selection does not
change; at selection boundaries the function is non-differentiable and the
gradient is one-sided. Unselected entries receive zero gradient.

Straight-through variant: ``straight_through=True`` keeps the forward value
g_tilde but back-propagates as if the weight were the dense probability g_i on
the selected entries (``g*m + (g_tilde - g*m).detach()``). This is a surrogate,
not an exact derivative; it exists because for k = 1 the exact gradient of
g_tilde (identically 1 on the selected expert) with respect to the router is 0.

Tie-breaking is deterministic: ties are resolved in favour of the lower expert
index (stable sort on the negated values).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

import needle as ndl
from needle import ops
from needle.autograd import Tensor

__all__ = ["TopKSelection", "topk_indices", "topk_mask", "sparse_renormalized_gates"]


def _validate_k(k, n_experts: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if not 1 <= k <= n_experts:
        raise ValueError(f"k must satisfy 1 <= k <= n_experts ({n_experts}), got {k}")
    return int(k)


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest entries along the last axis, descending, ties -> lower index.

    ``values`` has shape ``(..., E)``; the result has shape ``(..., k)``.
    """
    values = np.asarray(values)
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    k = _validate_k(k, values.shape[-1])
    # stable sort of -values: equal values keep their original (ascending index) order
    order = np.argsort(-values, axis=-1, kind="stable")
    return order[..., :k]


def topk_mask(values: np.ndarray, k: int) -> np.ndarray:
    """{0,1} mask of shape ``values.shape`` with exactly k ones per row (same dtype as values)."""
    values = np.asarray(values)
    idx = topk_indices(values, k)
    mask = np.zeros(values.shape, dtype=values.dtype)
    np.put_along_axis(mask, idx, 1, axis=-1)
    return mask


class TopKSelection(NamedTuple):
    """Result of sparse gating."""

    weights: Tensor  # (..., E) renormalised sparse gates, graph-bearing, zeros off the selection
    dense_probs: Tensor  # (..., E) the dense probabilities the selection was taken from
    indices: np.ndarray  # (..., k) selected expert indices, descending probability
    mask: np.ndarray  # (..., E) {0,1}


def sparse_renormalized_gates(dense_probs: Tensor, k: int, *, straight_through: bool = False) -> TopKSelection:
    """Select top-k experts of ``dense_probs`` along the last axis and renormalise their weights."""
    E = dense_probs.shape[-1]
    k = _validate_k(k, E)
    probs_np = dense_probs.numpy()
    indices = topk_indices(probs_np, k)
    mask_np = np.zeros(probs_np.shape, dtype=probs_np.dtype)
    np.put_along_axis(mask_np, indices, 1, axis=-1)

    mask = ndl.Tensor(mask_np, device=dense_probs.device, dtype=str(np.dtype(probs_np.dtype)), requires_grad=False)
    selected = dense_probs * mask  # zeros off the selection, gradient only on selected entries
    if k == E:
        weights = selected  # already normalised: softmax sums to one
    elif k == 1:
        # g_i / g_i == 1 identically on the selection: the exact derivative w.r.t. g is 0, so the
        # weight is the constant mask (avoids 1e-17 round-off "gradients" from computing g/g).
        weights = mask
    else:
        keep_shape = tuple(dense_probs.shape[:-1]) + (1,)
        denom = ops.broadcast_to(ops.reshape(ops.summation(selected, axes=(len(dense_probs.shape) - 1,)), keep_shape),
                                 dense_probs.shape)
        weights = selected / denom
    if straight_through:
        # value: renormalised weights; gradient: that of the dense probabilities on the selection
        weights = selected + (weights - selected).detach()
    return TopKSelection(weights=weights, dense_probs=dense_probs, indices=indices, mask=mask_np)
