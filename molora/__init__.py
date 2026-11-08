"""molora: Efficient Mixture-of-LoRA Experts in Needle.

Phase 1 provides a native LoRA linear layer (:class:`molora.LoRALinear`) built
entirely from Needle tensor operations and autograd.
"""
from .needle_compat import import_needle, needle_info

ndl = import_needle()  # makes `import needle` work before anything else is imported

from .lora import (  # noqa: E402
    LoRALinear,
    count_parameters,
    freeze_module,
    frozen_parameters,
    trainable_parameters,
)

__version__ = "0.1.0"
__all__ = [
    "LoRALinear",
    "count_parameters",
    "freeze_module",
    "frozen_parameters",
    "trainable_parameters",
    "needle_info",
    "ndl",
]
