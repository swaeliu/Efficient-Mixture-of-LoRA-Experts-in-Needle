"""Phase 4C — a small Needle-native decoder-only transformer and adapter injection utilities."""
from .tiny_transformer import (Block, CausalSelfAttention, Embedding, FeedForward, LayerNorm, TinyTransformer,
                               load_state, named_parameters, save_state)
from .transformer_adapters import (adapter_layers, adapter_trainable_parameters, count_parameters, freeze_backbone,
                                   inject_adapters, routing_stats, sparse_execution_stats)

__all__ = ["Block", "CausalSelfAttention", "Embedding", "FeedForward", "LayerNorm", "TinyTransformer", "named_parameters",
           "save_state", "load_state", "adapter_layers", "adapter_trainable_parameters", "count_parameters",
           "freeze_backbone", "inject_adapters", "routing_stats", "sparse_execution_stats"]
