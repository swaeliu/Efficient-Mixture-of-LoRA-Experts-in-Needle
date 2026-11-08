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
2. dense Mixture-of-LoRA experts (Phase 2),
3. sparse Mixture-of-LoRA experts with top-k routing (Phase 3),
4. optionally, quantised frozen backbone weights in the spirit of QLoRA (Phase 4).

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Environment, Needle integration, framework verification | **complete** ([docs/needle_environment.md](docs/needle_environment.md)) |
| 1 | Native `LoRALinear`, forward/backward/freezing tests, synthetic low-rank benchmark | **complete** ([docs/phase1_validation.md](docs/phase1_validation.md)) |
| 2 | Dense Mixture-of-LoRA (several adapters per layer, softmax router) | planned |
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
python -m pytest            # 79 tests, < 1 s on CPU
python experiments/smoke_test.py
```

| File | Covers |
|---|---|
| `tests/test_environment.py` | Needle import, CPU-only device, tensor/matmul vs NumPy, analytic gradients, `nn.Linear`, SGD/Adam steps, parameter identification |
| `tests/test_lora_forward.py` | shapes, zero update at init, equivalence to the wrapped `Linear`, `W0 + s·AB` reference, ranks, bias, invalid configs |
| `tests/test_lora_backward.py` | finite-difference checks (float64, tight; float32, loose) for `A`, `B`, `x`, analytic gradient comparison, stacked layers |
| `tests/test_lora_training.py` | frozen `W0`/bias unchanged across optimizer steps, optimizer receives only adapters, parameter counts, determinism |

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

## Repository layout

```
molora/            package: needle_compat.py (locates Needle), lora.py (LoRALinear)
tests/             pytest suite (Phase 0 + Phase 1)
experiments/       smoke_test.py, train_synthetic.py
configs/           lora_baseline.yaml
scripts/           setup_env.sh, link_needle.py
docs/              needle_environment.md, phase1_validation.md, figures/
```

## Planned next steps

Phase 2 starts with a `MoLoRALinear` that holds `n_experts` `(A_i, B_i)` pairs
on one frozen `W0` and a linear router producing softmax gates, i.e. dense
mixing `y = x W0 + Σ_i g_i(x) · s · x A_i B_i`, together with the same forward,
gradient and freezing test battery used here. Sparse top-k routing (Phase 3)
requires a top-k / scatter operation that Needle does not provide and will be
added as a documented, tested extension.
