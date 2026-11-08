"""Row gather / scatter-add operators for token-to-expert dispatch (Phase 3B).

Needle (hw2) has no indexing op, so these are project-local ``TensorOp``
subclasses. Indices are plain NumPy integer arrays held by the op instance:
they are discrete metadata, never differentiable.

    gather_rows(x, idx)                 (N, D) -> (M, D)     out[m] = x[idx[m]]
    scatter_add_rows([y_1, ...], [idx_1, ...], N)
                                        (M_j, D) -> (N, D)   out[idx_j[m]] += y_j[m]

The two ops are adjoint: the gradient of ``gather_rows`` scatter-adds the
upstream gradient into the source rows (repeated indices accumulate), and the
gradient of ``scatter_add_rows`` gathers the upstream gradient at each
destination row for every input. Both are implemented with NumPy fancy
indexing / ``np.add.at`` on Needle's NumPy CPU backend; a different backend
would need its own kernels. ``scatter_add_rows`` takes any number of inputs so
all experts' contributions can be accumulated into one output without
materialising one dense (N, D) buffer per expert.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from needle.autograd import Tensor, TensorOp

__all__ = ["GatherRows", "ScatterAddRows", "gather_rows", "scatter_add_rows"]


def _as_index(idx, name="idx") -> np.ndarray:
    idx = np.asarray(idx)
    if idx.ndim != 1:
        raise ValueError(f"{name} must be a 1-D integer array, got shape {idx.shape}")
    if idx.size and not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype, got {idx.dtype}")
    return idx.astype(np.intp, copy=False)


class GatherRows(TensorOp):
    """``out[m] = x[idx[m]]`` for a 2-D input ``x``."""

    def __init__(self, idx) -> None:
        self.idx = _as_index(idx)

    def compute(self, a):
        if a.ndim != 2:
            raise ValueError(f"gather_rows expects a 2-D array, got shape {a.shape}")
        if self.idx.size and (self.idx.min() < 0 or self.idx.max() >= a.shape[0]):
            raise IndexError(f"gather_rows index out of range for {a.shape[0]} rows")
        return a[self.idx]

    def gradient(self, out_grad: Tensor, node: Tensor):
        n_rows = node.inputs[0].shape[0]
        return scatter_add_rows([out_grad], [self.idx], n_rows)


class ScatterAddRows(TensorOp):
    """``out[idx_j[m]] += y_j[m]`` over all inputs ``y_j``; output has ``n_rows`` rows."""

    def __init__(self, index_list: Sequence, n_rows: int) -> None:
        self.index_list = [_as_index(i, f"index_list[{j}]") for j, i in enumerate(index_list)]
        if isinstance(n_rows, bool) or not isinstance(n_rows, (int, np.integer)) or n_rows < 0:
            raise ValueError(f"n_rows must be a non-negative int, got {n_rows!r}")
        self.n_rows = int(n_rows)

    def compute(self, *arrays):
        if len(arrays) != len(self.index_list):
            raise ValueError(f"expected {len(self.index_list)} inputs, got {len(arrays)}")
        if not arrays:
            raise ValueError("scatter_add_rows needs at least one input")
        d = arrays[0].shape[1]
        out = np.zeros((self.n_rows, d), dtype=arrays[0].dtype)
        for a, idx in zip(arrays, self.index_list):
            if a.ndim != 2 or a.shape[1] != d or a.shape[0] != idx.shape[0]:
                raise ValueError(f"input of shape {a.shape} does not match {idx.shape[0]} indices with width {d}")
            if idx.size and (idx.min() < 0 or idx.max() >= self.n_rows):
                raise IndexError(f"scatter_add_rows index out of range for {self.n_rows} rows")
            np.add.at(out, idx, a)  # repeated indices accumulate
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        return tuple(gather_rows(out_grad, idx) for idx in self.index_list)


def gather_rows(x: Tensor, idx) -> Tensor:
    """Differentiable row gather ``x[idx]`` (``idx`` is non-differentiable metadata)."""
    return GatherRows(idx)(x)


def scatter_add_rows(tensors: Sequence[Tensor], index_list: Sequence, n_rows: int) -> Tensor:
    """Differentiable multi-input row scatter-add into an ``(n_rows, D)`` output."""
    if len(tensors) != len(index_list):
        raise ValueError("tensors and index_list must have the same length")
    return ScatterAddRows(index_list, n_rows)(*tensors)
