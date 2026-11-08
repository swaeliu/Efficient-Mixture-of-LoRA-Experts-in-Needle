"""Project-local Needle-compatible operators (Phase 3).

Needle (hw2) has no indexing, gather, scatter, top-k or stack operation. The
operators here are implemented as ``needle.autograd.TensorOp`` subclasses or
as compositions of existing Needle ops, without modifying the coursework.
"""
from .dispatch import GatherRows, ScatterAddRows, gather_rows, scatter_add_rows
from .topk import TopKSelection, sparse_renormalized_gates, topk_indices, topk_mask

__all__ = ["GatherRows", "ScatterAddRows", "gather_rows", "scatter_add_rows", "TopKSelection",
           "sparse_renormalized_gates", "topk_indices", "topk_mask"]
