# Efficient Mixture-of-LoRA Experts in Needle

Research codebase investigating whether **sparse routing among multiple
specialised LoRA adapters** improves the trade-off between model quality,
compute and memory compared with conventional single-adapter LoRA. Everything
model-side is implemented in [Needle](https://dlsyscourse.org/), the
from-scratch deep-learning framework built in CMU 10-714, using its native
tensors, autograd, `nn.Module`s and optimizers. PyTorch is not used for any model
computation; NumPy is used only for data generation and reference checks.

## Research question

Given a frozen backbone, can a mixture of low-rank adapters with (sparse) top-k
routing match or beat a single LoRA adapter of comparable trainable-parameter
budget, and at what computational cost? The planned comparison is

1. single-adapter LoRA (**implemented, Phase 1**),
2. dense Mixture-of-LoRA experts (**implemented, Phase 2**),
3. sparse Mixture-of-LoRA experts with top-k routing (Phase 3),
4. optionally, quantised frozen backbone weights in the spirit of QLoRA (Phase 4).

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Environment, Needle integration, framework verification | **complete** ([docs/needle_environment.md](docs/needle_environment.md)) |
| 1 | Native `LoRALinear`, forward/backward/freezing tests, synthetic low-rank benchmark | **complete** ([docs/phase1_validation.md](docs/phase1_validation.md)) |
| 2 | Dense Mixture-of-LoRA (`DenseMoLoRALinear`, softmax router, heterogeneous routing benchmark, CPU overhead benchmark) | **complete** ([docs/phase2_validation.md](docs/phase2_validation.md)) |
| 3 | Sparse top-k routing, load balancing, efficiency measurements | planned |
| 4 | Quantised frozen backbone (QLoRA-style), optional | planned |

Everything under "planned" does not exist in this repository yet.

## What is implemented

`molora/lora.py` provides `LoRALinear`, a frozen Needle linear layer with a
trainable low-rank update

    y = x W0 + b + (alpha / r) · x A B        W0: (in, out)  A: (in, r)  B: (r, out)

* `B` is zero-initialised, so the layer reproduces the base layer exactly at
  construction; `A` uses Needle's Kaiming-uniform initialiser.
* `W0` and `b` are frozen `Parameter`s (`requires_grad=False`); only `A`, `B`
  are trained. `adapter_parameters()` returns exactly `[A, B]` for the optimizer.
  See the module docstring for why `requires_grad=False` alone is not enough in
  Needle.
* `LoRALinear.from_linear(linear, rank, alpha)` wraps an existing
  `needle.nn.Linear` without copying its weights.
* Parameter accounting: `num_base_parameters()`, `num_adapter_parameters()`
  (= `r (in + out)`, frozen bias excluded), `num_total_parameters()`,
  `trainable_ratio()`, `parameter_summary()`.
* Helpers `freeze_module`, `trainable_parameters`, `frozen_parameters`,
  `count_parameters` for assembling larger frozen-backbone models.
* Arbitrary leading batch dimensions are supported (`(..., in) -> (..., out)`).

`molora/dense_molora.py` and `molora/router.py` (Phase 2) provide
`DenseMoLoRALinear`, a dense mixture of LoRA experts on one shared frozen
backbone:

    y = x W0 + b + Σ_i g_i(x) · (alpha / r) · x A_i B_i,     g(x) = softmax((x W_g + b_g) / T)

* `DenseSoftmaxRouter`: linear projection + temperature softmax over the expert
  axis, built from Needle's `logsumexp` so it is correct for `(batch, seq, in)`
  inputs. `molora.softmax` is the Needle-native softmax it uses.
* `LoRAExpert` holds one `(A_i, B_i)` pair (A Kaiming, B zeros); the backbone
  product `x W0` is computed once, not per expert.
* Gates are read per expert with a constant one-hot matmul because Needle has
  no indexing op; all experts run on every input (dense routing).
* `adapter_parameters()`, `router_parameters()`, `trainable_parameters()`
  (duplicate-free, backbone excluded), `from_linear(linear, **kwargs)`,
  `forward_with_routing(x) -> (y, gates)`, `router_probabilities(x)`,
  `expert_delta_weights()`, full parameter accounting
  (`E r (in + out) + in E + E` trainable; 8 708 for the 4 × rank-8, 128 → 128 default).
* `b_init_std` / `router_init` are experimental initialisation variants; the
  default is exact zero adapter contribution at initialisation.

```python
import numpy as np, needle as ndl, needle.nn as nn
from molora import DenseMoLoRALinear

base = nn.Linear(128, 128, bias=False)
moe = DenseMoLoRALinear.from_linear(base, n_experts=4, rank=8, alpha=16.0)
opt = ndl.optim.Adam(moe.trainable_parameters(), lr=3e-3)   # never pass moe.parameters()
x = ndl.Tensor(np.random.randn(32, 128).astype(np.float32), dtype="float32", requires_grad=False)
y, gates = moe.forward_with_routing(x)                        # gates: (32, 4), rows sum to 1
```

## Setup

Requirements: Python ≥ 3.10 (3.13 tested), a C compiler is **not** needed, no
GPU is needed. The Needle implementation is the user's completed 10-714 hw2
solution and is *linked*, not copied, into the environment (course solution
files are not redistributed in this repository).

```bash
git clone https://github.com/swaeliu/Efficient-Mixture-of-LoRA-Experts-in-Needle
cd Efficient-Mixture-of-LoRA-Experts-in-Needle
bash scripts/setup_env.sh /path/to/needle/python   # directory that CONTAINS needle/
source .venv/bin/activate
```

`setup_env.sh` creates `.venv`, installs this package in editable mode with the
dev extras (`numpy`, `pyyaml`, `pytest`, `matplotlib`) and writes a
`needle_local.pth` into the venv pointing at the Needle directory. If the
argument is omitted, `../hw2/python` relative to the repository is tried.
Without the `.pth` file you can instead `export NEEDLE_PYTHON_DIR=/path/to/python`;
`molora` falls back to it. Details and the exact Needle version used are in
[docs/needle_environment.md](docs/needle_environment.md).

Manual equivalent:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/link_needle.py /path/to/needle/python
```

## Running the tests

```bash
source .venv/bin/activate
python -m pytest            # 176 tests (79 Phase 1 + 97 Phase 2), < 1 s on CPU
python experiments/smoke_test.py
```

| File | Covers |
|---|---|
| `tests/test_environment.py` | Needle import, CPU-only device, tensor/matmul vs NumPy, analytic gradients, `nn.Linear`, SGD/Adam steps, parameter identification |
| `tests/test_lora_forward.py` | shapes, zero update at init, equivalence to the wrapped `Linear`, `W0 + s·AB` reference, ranks, bias, invalid configs |
| `tests/test_lora_backward.py` | finite-difference checks (float64, tight; float32, loose) for `A`, `B`, `x`, analytic gradient comparison, stacked layers |
| `tests/test_lora_training.py` | frozen `W0`/bias unchanged across optimizer steps, optimizer receives only adapters, parameter counts, determinism |
| `tests/test_router.py` | Needle softmax vs NumPy (2–4-D), stability, FD gradient, normalisation over the expert axis, temperature, invalid args |
| `tests/test_dense_molora_forward.py` | 2-D/3-D shapes, gate normalisation, initial equivalence to the backbone, `from_linear`, independent NumPy oracle, one-expert = `LoRALinear`, accounting, invalid configs |
| `tests/test_dense_molora_backward.py` | finite differences for every `A_i`, `B_i`, router weight/bias and input; zero-init gradient structure; identical experts ⇒ zero router gradient |
| `tests/test_dense_molora_training.py` | backbone bit-identical over Adam/SGD steps, experts and router move, optimizer list excludes backbone, stacked model, determinism |

## Running the synthetic experiment

```bash
python experiments/train_synthetic.py --config configs/lora_baseline.yaml
# overrides: --seeds 1 2 3  --max-steps 500  --lr 0.01  --output-dir /some/dir  --no-plot
```

The benchmark builds a frozen `W0`, a hidden rank-8 correction `ΔW* = A*B*`,
and trains only the LoRA adapter on `y = x (W0 + ΔW*)` with held-out
validation, for three seeds, against a frozen-backbone-only baseline. Success
criteria are fixed in the config *before* the run. Output structure:

```
outputs/lora_synthetic/<YYYYmmdd-HHMMSS>/
├── config.yaml      # resolved configuration (incl. overrides)
├── results.json     # per-seed loss histories, final metrics, parameter counts,
│                    # runtime, seeds, software environment, criteria verdicts
├── summary.md       # table of measured values + PASS/FAIL per criterion
└── loss_curve.png   # train / validation MSE vs step, one panel per seed
```

`outputs/` is git-ignored. The measured results of the confirmatory run are
recorded in [docs/phase1_validation.md](docs/phase1_validation.md) and the
figure is kept at `docs/figures/phase1_loss_curve.png`. The exit code is 0 only
if every success criterion passed.

## Running the Phase 2 routing benchmark

```bash
python experiments/train_dense_molora.py --pilot                 # LR sweep, validation data only
python experiments/train_dense_molora.py                         # confirmatory run, seeds 42 43 44, test data
python experiments/train_dense_molora.py --b-init-std 1e-3 --tag binit      # init ablation
python experiments/train_dense_molora.py --router-init zeros --tag routerzeros
python experiments/benchmark_dense_molora.py                     # CPU forward / train-step timings
```

The task has four input regimes (one-hot coded in the last four input
features), each with its own planted rank-8 correction on a shared frozen
`W0`. Four methods share the same splits, `W0`, loss, batch size and step
budget: frozen backbone, single LoRA r=8, parameter-matched single LoRA r=34
and dense MoLoRA 4 × r=8. Output structure:

```
outputs/dense_molora_synthetic/<timestamp>/
├── config.yaml, results.json, summary.md
├── routing_matrix_seed42.csv      # regime x expert mean routing probability
├── loss_curves.png                # test MSE vs step for every method, one panel per seed
├── routing_heatmap.png            # regime x expert probabilities per seed
└── routing_dynamics.png           # router grad norm, routing entropy, expert divergence vs step
outputs/benchmark_dense_molora/<timestamp>/benchmark.{json,md}
```

Measured results and figures are in [docs/phase2_validation.md](docs/phase2_validation.md).

## Repository layout

```
molora/            needle_compat.py (locates Needle), lora.py (LoRALinear),
                   router.py (DenseSoftmaxRouter, softmax), dense_molora.py (LoRAExpert, DenseMoLoRALinear)
tests/             pytest suite (Phases 0-2)
experiments/       smoke_test.py, train_synthetic.py, train_dense_molora.py, benchmark_dense_molora.py
configs/           lora_baseline.yaml, dense_molora.yaml
scripts/           setup_env.sh, link_needle.py
docs/              needle_environment.md, phase1_validation.md, phase2_validation.md, figures/
```

## Planned next steps

Phase 3 adds sparse top-k routing on top of `DenseMoLoRALinear`: a
differentiable top-k gate mask (a new project-side Needle `TensorOp`, since
Needle has no top-k/indexing op), skipping unselected experts' computation,
load-balance diagnostics and an optional auxiliary balance loss as a separate
ablation, and a dense-vs-sparse cost comparison using
`experiments/benchmark_dense_molora.py` as the reference. Dense routing is
measured here to cost about 3.5–5× a single LoRA per step on CPU; no efficiency
claim is made for it. Phase 4 (quantised frozen backbone) remains optional.
