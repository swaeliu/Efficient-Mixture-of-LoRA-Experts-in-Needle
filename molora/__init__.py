"""molora: Efficient Mixture-of-LoRA Experts in Needle.

Phase 1 provides a native LoRA linear layer (:class:`molora.LoRALinear`);
Phase 2 adds a dense mixture of LoRA experts (:class:`molora.DenseMoLoRALinear`)
with a softmax router (:class:`molora.DenseSoftmaxRouter`). Everything is built
from Needle tensor operations and autograd.
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
from .router import DenseSoftmaxRouter, softmax  # noqa: E402
from .dense_molora import DenseMoLoRALinear, LoRAExpert  # noqa: E402

__version__ = "0.1.0"
__all__ = [
    "LoRALinear",
    "LoRAExpert",
    "DenseMoLoRALinear",
    "DenseSoftmaxRouter",
    "softmax",
    "count_parameters",
    "freeze_module",
    "frozen_parameters",
    "trainable_parameters",
    "needle_info",
    "ndl",
]
