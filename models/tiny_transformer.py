"""A minimal autoregressive transformer built only from hw2 Needle ops (+ molora.softmax).

Design constraints documented in docs/phase4c_transformer.md: no embedding op
(one-hot matmul), LayerNorm1d is 2-D only (reshape around it), no masking op
(additive constant mask), logsoftmax is axis-1 only (logsumexp-based softmax),
all scalars are Python floats so the graph stays float32.
"""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np

import needle as ndl
import needle.nn as nn
from needle import init, ops
from needle.autograd import Tensor
from needle.nn import Parameter

from molora.router import softmax

__all__ = ["LayerNorm", "Embedding", "CausalSelfAttention", "FeedForward", "Block", "TinyTransformer",
           "named_parameters", "save_state", "load_state"]


class LayerNorm(nn.Module):
    """Layer norm over the last axis of an ``(..., d)`` tensor via Needle's 2-D ``LayerNorm1d``."""

    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype="float32") -> None:
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm1d(dim, eps=eps, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        lead = tuple(x.shape[:-1])
        n = int(np.prod(lead))
        return ops.reshape(self.ln(ops.reshape(x, (n, self.dim))), lead + (self.dim,))


class Embedding(nn.Module):
    """Lookup table implemented as one-hot @ weight (Needle has no gather)."""

    def __init__(self, num_embeddings: int, dim: int, std: float = 0.02, device=None, dtype="float32") -> None:
        super().__init__()
        self.num_embeddings, self.dim, self.dtype = int(num_embeddings), int(dim), str(np.dtype(dtype))
        self.device = device if device is not None else ndl.cpu()
        self.weight = Parameter(init.randn(self.num_embeddings, self.dim, std=std, device=self.device, dtype=self.dtype))

    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx)
        if idx.min() < 0 or idx.max() >= self.num_embeddings:
            raise IndexError("embedding index out of range")
        onehot = np.eye(self.num_embeddings, dtype=np.dtype(self.dtype))[idx]  # (..., V)
        return ndl.Tensor(onehot, device=self.device, dtype=self.dtype, requires_grad=False) @ self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, device=None, dtype="float32") -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.dtype = str(np.dtype(dtype))
        self.device = device if device is not None else ndl.cpu()
        self._mask_cache: dict[int, np.ndarray] = {}

    def _mask(self, B: int, T: int) -> Tensor:
        if T not in self._mask_cache:
            self._mask_cache[T] = np.triu(np.full((T, T), -1e9, dtype=np.dtype(self.dtype)), 1)
        m = np.broadcast_to(self._mask_cache[T], (B, self.n_heads, T, T)).copy()
        return ndl.Tensor(m, device=self.device, dtype=self.dtype, requires_grad=False)

    def _heads(self, x: Tensor, B: int, T: int) -> Tensor:
        return ops.transpose(ops.reshape(x, (B, T, self.n_heads, self.d_head)), (1, 2))  # (B, H, T, dh)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q, k, v = (self._heads(p(x), B, T) for p in (self.q_proj, self.k_proj, self.v_proj))
        scores = (q @ ops.transpose(k, (2, 3))) * (1.0 / math.sqrt(self.d_head))  # Python float keeps float32
        att = softmax(scores + self._mask(B, T), axis=-1)  # (B, H, T, T)
        out = ops.reshape(ops.transpose(att @ v, (1, 2)), (B, T, self.d_model))
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype="float32") -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-LayerNorm transformer block with residual connections."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, device=None, dtype="float32") -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalSelfAttention(d_model, n_heads, device=device, dtype=dtype)
        self.ln2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TinyTransformer(nn.Module):
    """Decoder-only LM: tokens ``(B, T)`` ints -> logits ``(B, T, V)``."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, d_ff: int = 128,
                 max_seq_len: int = 32, device=None, dtype="float32") -> None:
        super().__init__()
        self.vocab_size, self.d_model, self.n_heads, self.n_layers, self.d_ff, self.max_seq_len = (
            int(vocab_size), int(d_model), int(n_heads), int(n_layers), int(d_ff), int(max_seq_len))
        self.dtype = str(np.dtype(dtype))
        self.tok_emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.pos_emb = Embedding(max_seq_len, d_model, device=device, dtype=dtype)
        self.blocks = [Block(d_model, n_heads, d_ff, device=device, dtype=dtype) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
        self.loss_fn = nn.SoftmaxLoss()

    def forward(self, tokens: np.ndarray) -> Tensor:
        tokens = np.asarray(tokens)
        B, T = tokens.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        x = self.tok_emb(tokens) + self.pos_emb(np.broadcast_to(np.arange(T), (B, T)))
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))  # (B, T, V)

    def loss(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        B, T, V = logits.shape
        y = ndl.Tensor(np.asarray(targets).reshape(-1), dtype="int32", requires_grad=False)
        return self.loss_fn(ops.reshape(logits, (B * T, V)), y)

    @staticmethod
    def accuracy(logits: Tensor, targets: np.ndarray) -> float:
        return float((logits.numpy().argmax(-1) == np.asarray(targets)).mean())


# --------------------------------------------------------------------------- #
# Named parameters / checkpoints (Needle has neither)
# --------------------------------------------------------------------------- #
def named_parameters(module: nn.Module, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
    """Deterministic (name, Parameter) walk over attributes, lists and child modules."""
    for key, value in module.__dict__.items():
        name = f"{prefix}{key}"
        if isinstance(value, Parameter):
            yield name, value
        elif isinstance(value, nn.Module):
            yield from named_parameters(value, name + ".")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                if isinstance(v, Parameter):
                    yield f"{name}.{i}", v
                elif isinstance(v, nn.Module):
                    yield from named_parameters(v, f"{name}.{i}.")


def save_state(module: nn.Module, path: str) -> None:
    np.savez(path, **{n: p.numpy() for n, p in named_parameters(module)})


def load_state(module: nn.Module, path: str) -> None:
    data = np.load(path)
    params = dict(named_parameters(module))
    missing = set(params) ^ set(data.files)
    if missing:
        raise KeyError(f"checkpoint / module parameter mismatch: {sorted(missing)[:5]}")
    for n, p in params.items():
        arr = data[n]
        if arr.shape != tuple(p.shape):
            raise ValueError(f"shape mismatch for {n}: {arr.shape} vs {tuple(p.shape)}")
        p.data = ndl.Tensor(arr.astype(np.dtype(p.dtype)), device=p.device, dtype=str(np.dtype(p.dtype)), requires_grad=False)
