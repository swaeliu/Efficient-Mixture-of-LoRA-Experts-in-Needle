#!/usr/bin/env python
"""Quick end-to-end smoke test: build a LoRA layer, run forward/backward, take one Adam step."""
import numpy as np

import molora
import needle as ndl
import needle.nn as nn
from molora import LoRALinear


def main() -> None:
    print("needle:", molora.needle_info())
    np.random.seed(0)
    base = nn.Linear(16, 8)
    layer = LoRALinear.from_linear(base, rank=4, alpha=8.0)
    print(layer)
    print("parameter summary:", layer.parameter_summary())

    x = ndl.Tensor(np.random.randn(5, 16).astype(np.float32), dtype="float32", requires_grad=False)
    y0 = layer(x).numpy()
    print("max |lora(x) - base(x)| at init:", float(np.abs(y0 - base(x).numpy()).max()))

    W0 = layer.weight.numpy().copy()
    opt = ndl.optim.Adam(layer.adapter_parameters(), lr=1e-2)
    loss = (layer(x) ** 2).sum() / y0.size
    loss.backward()
    opt.step()
    print("loss:", float(loss.numpy()))
    print("W0 unchanged after step:", np.array_equal(W0, layer.weight.numpy()))
    print("B moved after step:", not np.array_equal(layer.lora_B.numpy(), 0.0))
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
