"""Insert LoRA / dense MoLoRA / sparse MoLoRA adapters into TinyTransformer projection layers."""
from __future__ import annotations

import numpy as np

import needle.nn as nn
from needle.autograd import Tensor

from molora import DenseMoLoRALinear, LoRALinear, SparseMoLoRALinear

__all__ = ["freeze_backbone", "inject_adapters", "adapter_layers", "adapter_trainable_parameters", "count_parameters",
           "routing_stats", "sparse_execution_stats", "TARGETS"]

TARGETS = {"ffn": [("ffn", "fc1"), ("ffn", "fc2")], "attn_qv": [("attn", "q_proj"), ("attn", "v_proj")]}


def count_parameters(params) -> int:
    return int(sum(int(np.prod(p.shape)) for p in params))


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Mark every current parameter frozen (Phase 1 strategy: requires_grad False + grad None)."""
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    return model


def inject_adapters(model, kind: str, rank: int = 4, alpha: float = 8.0, n_experts: int = 4, top_k: int = 2,
                    router_grad_mode: str = "renormalized", execution_mode: str = "dispatch", where=("ffn",),
                    router_init: str = "xavier") -> list:
    """Replace the targeted ``nn.Linear`` layers of every block with adapter layers sharing the frozen weights.

    kind: "lora" | "dense" | "sparse". Returns the list of inserted adapter layers (in block order).
    The frozen backbone weights are shared by storage (``from_linear``), never copied.
    """
    if kind not in ("lora", "dense", "sparse"):
        raise ValueError(f"unknown adapter kind {kind!r}")
    inserted = []
    for blk in model.blocks:
        for group in where:
            for parent_name, attr in TARGETS[group]:
                parent = getattr(blk, parent_name)
                lin = getattr(parent, attr)
                if not isinstance(lin, nn.Linear):
                    raise TypeError(f"{parent_name}.{attr} is already adapted ({type(lin).__name__})")
                if kind == "lora":
                    layer = LoRALinear.from_linear(lin, rank=rank, alpha=alpha)
                elif kind == "dense":
                    layer = DenseMoLoRALinear.from_linear(lin, n_experts=n_experts, rank=rank, alpha=alpha, router_init=router_init)
                else:
                    layer = SparseMoLoRALinear.from_linear(lin, n_experts=n_experts, rank=rank, alpha=alpha, top_k=top_k,
                                                           router_grad_mode=router_grad_mode, execution_mode=execution_mode,
                                                           router_init=router_init)
                setattr(parent, attr, layer)
                inserted.append(layer)
    return inserted


def adapter_layers(model) -> list:
    out = []
    for blk in model.blocks:
        for group in TARGETS.values():
            for parent_name, attr in group:
                layer = getattr(getattr(blk, parent_name), attr)
                if isinstance(layer, (LoRALinear, DenseMoLoRALinear)):
                    out.append(layer)
    return out


def adapter_trainable_parameters(model) -> list[Tensor]:
    """Trainable adapter (+ router) parameters of all adapter layers, duplicate-free, backbone excluded."""
    params = []
    for layer in adapter_layers(model):
        params += layer.trainable_parameters()
    assert len({id(p) for p in params}) == len(params)
    assert all(p.requires_grad for p in params)
    return params


def routing_stats(model, x_tokens: np.ndarray) -> list[dict]:
    """Per adapter layer: mean dense routing probabilities, utilisation and entropy over the given tokens (MoE only)."""
    stats = []
    layers = adapter_layers(model)
    if not any(isinstance(l, DenseMoLoRALinear) for l in layers):
        return stats
    # capture layer inputs by running the model once with hooks replaced by wrappers
    captured = {}
    for i, layer in enumerate(layers):
        if isinstance(layer, DenseMoLoRALinear):
            orig = layer.forward

            def wrapped(x, _layer=layer, _i=i, _orig=orig):
                captured[_i] = x
                return _orig(x)
            layer.forward = wrapped
    model(x_tokens)
    for i, layer in enumerate(layers):
        if isinstance(layer, DenseMoLoRALinear):
            del layer.forward  # restore class method
            p = layer.router_probabilities(captured[i]).reshape(-1, layer.n_experts).astype(np.float64)
            ent = float((-(p * np.log(np.clip(p, 1e-300, None))).sum(-1)).mean())
            if isinstance(layer, SparseMoLoRALinear):
                d = layer.routing_diagnostics(captured[i])
                util = d["utilization"]
            else:
                util = [1.0] * layer.n_experts
            stats.append({"layer": i, "mean_prob": p.mean(0).tolist(), "argmax_utilization": (np.bincount(p.argmax(1), minlength=layer.n_experts) / len(p)).tolist(),
                          "utilization": util, "entropy_nats": ent, "max_entropy_nats": float(np.log(layer.n_experts))})
    return stats


def sparse_execution_stats(model) -> dict:
    """Aggregate last_stats of all sparse layers after a forward pass."""
    agg = {"expert_rows_executed": 0, "expert_rows_dense": 0, "expert_matmuls_executed": 0, "experts_skipped": 0, "layers": 0}
    for layer in adapter_layers(model):
        if isinstance(layer, SparseMoLoRALinear) and layer.last_stats:
            st = layer.last_stats
            agg["expert_rows_executed"] += sum(st["expert_rows_executed"])
            agg["expert_rows_dense"] += st["n_experts"] * st["n_tokens"]
            agg["expert_matmuls_executed"] += st["expert_matmuls_executed"]
            agg["experts_skipped"] += st["experts_skipped"]
            agg["layers"] += 1
    agg["expert_flop_fraction_vs_dense"] = agg["expert_rows_executed"] / agg["expert_rows_dense"] if agg["expert_rows_dense"] else None
    return agg
