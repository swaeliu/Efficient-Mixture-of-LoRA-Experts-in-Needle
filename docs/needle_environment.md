# Needle environment (Phase 0)

Date: 2026-11-08. Machine: Linux 6.12 (x86_64, AMD EPYC 9334, 128 logical
CPUs), system Python 3.13.14. No GPU is required or used; an NVIDIA driver is
present on the host but no CUDA toolkit (`nvcc`) is installed and the selected
Needle build has no CUDA backend.

## 1. Candidate implementations found

The Needle source is **not** directly under `~/projects/dlsys/`; each homework
carries its own copy under `hwN/python/needle/`. The coursework directories are
not git repositories, so there is no commit hash; provenance is recorded by
path and content below.

| Location | Autograd | Ops | `nn` modules | `optim` | `init` | Backend | Verdict |
|---|---|---|---|---|---|---|---|
| `~/projects/dlsys/hw1/python/needle` | complete | mathematic ops complete | — (none) | — | basic only | NumPy | too limited (no modules / optimizers) |
| **`~/projects/dlsys/hw2/python/needle`** | **complete** | **mathematic + logarithmic + tuple ops complete** | **Linear, Flatten, ReLU, Sequential, SoftmaxLoss, BatchNorm1d, LayerNorm1d, Dropout, Residual** | **SGD (momentum, wd, clip), Adam** | **xavier/kaiming uniform+normal** | NumPy (CPU) | **selected** |
| `~/projects/dlsys/hw3/python/needle` | stubs (`NotImplementedError` in `compute_gradient_of_variables`, topo sort) | all `compute`/`gradient` bodies are stubs | all stubs | stubs | stubs | `backend_ndarray` NDArray layer (CPU C++ / CUDA / NumPy) implemented in Python; the only compiled `.so` present is a macOS `cpython-312-darwin` binary | not usable without re-implementing hw2 on top of it |

`hw3` is the only tree with a device abstraction (`backend_selection.py`,
`NEEDLE_BACKEND=nd|np`, `cuda()` device) but every layer above the NDArray
class is unimplemented in this copy, and its compiled CPU backend was built for
macOS/Python 3.12. It is therefore not a working framework on this Linux host.

**Selected: `~/projects/dlsys/hw2/python/needle`** — the most complete,
verified implementation.

## 2. Verification of the selected implementation

Run on 2026-11-08 with the project venv (NumPy 2.5.4, pytest 9.2.0), the course
grader module `mugrade` stubbed and `numdifftools` installed into a scratch
directory (neither is needed by this project):

| Suite | Result |
|---|---|
| `hw2/tests/hw2/test_nn_and_optim.py` (hw2 Needle) | **87 passed, 1 failed** |
| `hw1/tests/hw1/test_autograd_hw.py` run against the hw2 Needle | **29 passed** |

The single failure is `test_optim_adam_z_memory_check_1`, which asserts that the
global tensor counter is within ±1000 of 1132 after training; it measured 0
(all graph tensors had been garbage-collected under Python 3.13). It is a
bookkeeping check on allocation behaviour, not a numerical test, and the
implementation allocates *fewer* tensors than the reference. All forward,
backward, initialiser, module and optimizer tests pass.

Project-level verification lives in `tests/test_environment.py` (15 tests):
import, CPU-only device list, tensor creation, matmul vs NumPy (2-D and
batched), analytic gradient of a scalar objective (`6x + 2`), matmul gradients,
`nn.Linear` instantiation, SGD and Adam steps, parameter identification and
freezing, float64 support, and a check that `torch` is never imported.

## 3. Relevant API facts (hw2 Needle)

* `Tensor(array, device=None, dtype=None, requires_grad=True)`. Passing a Tensor
  shares its storage; passing a NumPy array **keeps its dtype** (so float64
  arrays must be cast explicitly). Device is always `needle.cpu()`.
* `nn.Linear(in, out, bias)` stores `weight` as `(in_features, out_features)`
  and computes `X @ weight + bias` — row-vector convention, matching
  `y = x W` in the LoRA formula directly. Bias init is Kaiming-uniform with
  fan-in `out_features`, reshaped to `(out,)`.
* `nn.Parameter` is a `Tensor` subclass; `Module.parameters()` returns every
  `Parameter` found recursively in `__dict__` (lists, tuples, dicts, child
  modules). There is no `named_parameters`, no buffers, no hooks.
* Optimizers (`optim.SGD`, `optim.Adam`) take an explicit parameter list, skip
  parameters whose `.grad is None`, and update by rebinding `p.data` to a new
  tensor (never in place). They read `p.grad` unconditionally; a parameter that
  never went through `backward()` has no `.grad` attribute at all.
* Autograd (`compute_gradient_of_variables`) writes `.grad` on **every** node it
  visits, including leaves with `requires_grad=False`. `requires_grad=False`
  only causes `make_from_op` to return a *detached* result when **all** inputs
  of an op are constants. Consequence: a frozen weight receives a gradient as
  soon as the layer input requires a gradient (any non-first layer).
* `ops.matmul` broadcasts a 2-D right operand against N-D left operands and its
  gradient sums over the extra leading dims; `broadcast_to` / `reshape` /
  `summation` gradients are implemented. Supported ops: add, mul, pow, div
  (tensor and scalar variants), negate, transpose (two axes), reshape,
  broadcast_to, summation, matmul, log, exp, relu, logsoftmax, logsumexp,
  make_tuple / tuple_get_item.
* float32 and float64 both work end to end (`init.*` accept a `dtype`).
* Random initialisation uses NumPy's **global** RNG (`numpy.random.rand/randn`);
  seed with `np.random.seed(seed)`.

## 4. Can LoRA be built without modifying Needle?

Yes. `LoRALinear` needs only `matmul`, `reshape`, `broadcast_to`, elementwise
add and scalar multiply, all present with gradients. Selective freezing is
achieved without touching Needle by (a) marking base parameters
`requires_grad=False`, (b) initialising their `.grad = None` so Needle's
optimizers can tolerate them, and (c) handing the optimizer only
`adapter_parameters()`. **No modification of the coursework was made.**

## 5. Missing functionality (for later phases, not needed now)

* No `top-k`, `argmax`, `gather`/`scatter`, boolean masking or indexing ops —
  required for sparse routing (Phase 3). These must be added as project-side
  ops (own `TensorOp` subclasses with `compute`/`gradient`) in `molora`.
* No `softmax` module (only `logsoftmax`/`logsumexp` ops) — a router softmax is
  `exp(logsoftmax(z))`, available today.
* No `stack`/`split`/`concatenate` ops — a dense mixture can be expressed as a
  sum over experts without them.
* No CUDA device and no `tanh`/`sigmoid`; GPU results cannot be produced from
  this build. The hw3 NDArray backend could be revisited if GPU numbers become
  necessary, but it would require completing hw3's autograd/ops/nn layers.
* `Module` has no `named_parameters`; parameter identity is tracked by object
  identity in this project.

## 6. Integration mechanism

Needle is linked, not vendored (it is course solution code and is not
redistributed here):

1. `scripts/link_needle.py [dir]` writes `needle_local.pth` into the active
   environment's `site-packages` pointing at the directory containing
   `needle/` (default `~/projects/dlsys/hw2/python`) and verifies the import in
   a fresh interpreter. `--remove` undoes it.
2. Fallback: `NEEDLE_PYTHON_DIR` environment variable, honoured by
   `molora/needle_compat.py`, which also tries `<repo>/../hw2/python`.
3. `scripts/setup_env.sh` performs the whole setup (venv, `pip install -e
   ".[dev]"`, link, import check).

Installed into `.venv` (Python 3.13.14): numpy 2.5.4, pyyaml 6.0.3, pytest
9.2.0, matplotlib 3.11.2, molora 0.1.0 (editable).
