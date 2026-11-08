# Phase 3 design — sparse top-k Mixture-of-LoRA in Needle

Date: 2026-11-08. This document records the operator inventory of the hw2
Needle framework relevant to sparse routing, the project-local operators added
on top of it (without modifying the coursework), their gradient semantics, and
the dispatch implementation. Measured results are in
[phase3_validation.md](phase3_validation.md).

## 1. Needle (hw2) operator inventory

| Capability | Status in `~/projects/dlsys/hw2/python/needle` | Consequence |
|---|---|---|
| `TensorOp` base (`compute(*ndarrays)`, `gradient(out_grad, node)`), `Tensor.make_from_op`, `detach()`, `make_const` | present | project-local ops can be plain subclasses |
| elementwise add/mul/div/pow (+ scalar), negate, log, exp, relu | present, with gradients | used for gate masking / renormalisation |
| `reshape`, `broadcast_to`, `summation(axes)`, `transpose` (2 axes) | present, with gradients | used to flatten tokens and broadcast gates |
| `matmul` with N-D left operand / 2-D right operand | present | one-hot gate extraction, expert products |
| `logsumexp(axes)` | present, arbitrary axes | stable softmax over the expert axis (`logsoftmax` is axis-1 only) |
| tensor indexing / slicing (`__getitem__`, `__setitem__`) | **absent** on `Tensor` (only `TensorTuple.__getitem__`) | gates cannot be sliced per expert |
| gather / scatter / scatter-add | **absent** | needed for token dispatch |
| top-k / argsort / argmax | **absent** | needed for expert selection |
| stack / concatenate / split | **absent** | experts cannot be batched into one tensor |
| masked select / boolean indexing | **absent** | |
| in-place ops | absent by design (optimizers rebind `.data`) | frozen storage sharing stays safe |
| devices | CPU/NumPy only | all new ops are NumPy kernels |

The Phase 2 dense layer already worked around the missing indexing by reading
gate column *i* as `gates @ e_i` with a constant one-hot `(E, 1)` selector.

**Minimal operator set for correct sparse routing:** (a) a top-k *selection*
(discrete; no gradient), (b) masking + renormalisation of the selected
probabilities (existing ops), (c) for genuine sparse execution, a row gather and
a row scatter-add with correct adjoint gradients.

## 2. Top-k operator (`molora/ops/topk.py`)

```
indices = topk_indices(g, k)       # NumPy, (..., k), descending, ties -> lower index (stable argsort of -g)
mask    = {0,1}^(..., E)           # exactly k ones per row, constant Tensor
g_sel   = g * mask                 # Needle EWiseMul with a constant
g_tilde = g_sel / broadcast(sum(g_sel, -1))      # Needle ops   (k < E)
          g_sel                                   # k == E: softmax already sums to one
          mask                                    # k == 1: g_i / g_i == 1 identically -> constant
```

`sparse_renormalized_gates(dense_probs, k, straight_through=False)` returns a
`TopKSelection(weights, dense_probs, indices, mask)`.

**Gradient semantics.** The selection `S_k(x)` is treated as fixed for one
forward/backward evaluation. For `1 < k < E` autograd gives the *exact*
derivative of `g̃_i = g_i / Σ_{j∈S} g_j` with respect to `g` on the open set
where the selection does not change (zero on unselected entries); at a
selection boundary the function is non-differentiable and the computed value is
one-sided. For `k = 1`, `g̃` is identically 1 on the selected expert, so its
derivative with respect to the router is exactly 0 — the weights are returned
as a constant and the router receives no task gradient at all. This is a
property of renormalised top-1 routing, not an implementation gap, and it is
the reason the straight-through surrogate exists:

```
weights_STE = g_sel + (g_tilde - g_sel).detach()
```

has the forward value `g̃` but back-propagates as if the weight were the dense
probability `g_i` on the selected entries. It is **not** an exact derivative of
the forward function; it is the standard straight-through estimator and is
reported as such wherever it is used (`router_grad_mode="straight_through"`).

Deterministic tie-breaking: `np.argsort(-g, kind="stable")` picks the lower
expert index first. Uniform probabilities (zero router) therefore route every
token to experts `0..k-1`.

## 3. Masked reference router (`molora/sparse_router.py`)

`SparseTopKRouter(input_dim, n_experts, top_k, temperature, router_grad_mode)`
owns a `DenseSoftmaxRouter` (`self.dense`; the only parameters) and returns
`route(x) -> TopKSelection` / `forward(x) -> weights`. It computes all `E`
probabilities, masks, renormalises, and returns the metadata. This is the
correctness baseline used by the reference execution mode.

## 4. `SparseMoLoRALinear` (`molora/sparse_molora.py`)

Subclass of `DenseMoLoRALinear`: the frozen shared backbone, the `LoRAExpert`
list, `from_linear`, parameter accounting (`E r (in+out) + in E + E`
trainable), `load_expert` and the Phase 1 freezing strategy are inherited. The
dense router is replaced by a `SparseTopKRouter` holding the same `Parameter`
objects (no duplicates in `parameters()`).

Leading dimensions are flattened to a token axis `(N, in)` with Needle
`reshape` (differentiable) and restored at the end.

**`execution_mode="reference"`** — every expert runs on every token, output
`Σ_i broadcast(g̃ @ e_i) * expert_i(x)`; identical to the dense layer with
`g̃` in place of `g`. Trusted baseline; expert FLOPs = dense.

**`execution_mode="dispatch"`** — for each expert `i`:

```
tok   = flatnonzero(mask[:, i])                 # tokens that selected i (NumPy metadata)
if tok is empty: skip (no gather, no matmul)
x_i   = gather_rows(x_flat, tok)                # (M_i, in)
w_i   = gather_rows(g_tilde, tok) @ e_i         # (M_i, 1) selected weights (graph-bearing)
dy_i  = (x_i @ A_i) @ B_i * (alpha / r)         # the only expert matmuls executed
c_i   = broadcast(w_i) * dy_i
out   = x_flat @ W0 (+ b) + scatter_add_rows([c_1, ..., c_E], [tok_1, ..., tok_E], N)
```

A single multi-input scatter-add accumulates all experts, so no dense
`(N, E)`-by-`out` buffer and no per-expert `(N, out)` temporaries are created;
expert activations total `k·N` rows instead of `E·N`. The dense `(N, E)` gate
matrix itself is unavoidable because the softmax is over all experts.

`last_stats` (plain numbers) records per forward: tokens per expert, rows each
expert executed, expert matmuls executed, experts skipped, adapter FLOPs
executed vs dense and the theoretical fraction `k / E`. Because every token
selects exactly `k` experts, `Σ_i M_i = kN`, so the executed adapter-FLOP
fraction is exactly `k / E` regardless of load balance; only the *count of
skipped experts* depends on balance. The accounting covers expert matmuls only;
backbone, router, gather/scatter and the output add are excluded.

`profile=True` records wall time per stage (backbone / router / expert /
dispatch) for one forward in `last_timing`.

## 5. Dispatch operators (`molora/ops/dispatch.py`)

```
GatherRows(idx):        compute  a[idx]                          (N, D) -> (M, D)
                        gradient scatter_add_rows([g], [idx], N)  (repeats accumulate)
ScatterAddRows(idx_list, n_rows, assume_unique):
                        compute  out = zeros(n_rows, D); for each input: out[idx_j] += y_j
                        gradient tuple(gather_rows(g, idx_j))       one per input
```

The two ops are adjoint (tested: `⟨gather(x), y⟩ = ⟨x, scatter(y)⟩`). Indices
are `np.intp` arrays stored on the op; they are never Tensors. Repeated indices
inside one input are accumulated with `np.add.at`; `assume_unique=True`
(used by dispatch, where a token selects an expert at most once) switches to
buffered fancy-index addition, which is exact under that promise and roughly
10× faster than `np.add.at`. Repeats *across* inputs are always exact because
inputs are applied sequentially.

Limitation: both kernels are NumPy on Needle's CPU backend; a CUDA backend
would need its own gather/scatter kernels. Data movement is a copy (gather) and
a read-modify-write (scatter) of `k·N·D` floats.

## 6. Auxiliary balance loss (`load_balance_loss`)

Switch-Transformer style `E Σ_i f_i P_i` (`f_i` constant selection fraction,
`P_i` mean dense probability). Implemented and tested but **not used** in any
Phase 3 experiment; available for future ablations.

## 7. What is deliberately not done

* No modification of the coursework Needle.
* No custom compiled kernels, no batched expert matmul (no `stack` op).
* No capacity factor / token dropping; every token is routed to exactly `k` experts.
* No GPU path.
