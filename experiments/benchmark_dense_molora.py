#!/usr/bin/env python
"""Phase 2 CPU overhead benchmark (Task 2.11).

Measures forward-pass and full training-step (forward + backward + Adam step on
the trainable parameters) wall time for

    frozen backbone | single LoRA r=8 | parameter-matched LoRA r=34 | dense MoLoRA 4 x r=8

with warm-up iterations and repeated measurements (median and inter-quartile
range), examples / second, parameter counts, Python-level peak allocation of a
training step (tracemalloc; NumPy routes its allocations through it) and the
process peak RSS. Dense routing is expected to cost MORE than single LoRA; this
script only measures, it does not optimise.

    python experiments/benchmark_dense_molora.py [--batch-sizes 32 256] [--repeats 100] [--warmup 10]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import resource
import statistics
import sys
import time
import tracemalloc

import numpy as np

import molora
import needle as ndl
import needle.nn as nn
from molora import DenseMoLoRALinear, LoRALinear

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(name, in_dim, out_dim, seed=0):
    np.random.seed(seed)
    base = nn.Linear(in_dim, out_dim, bias=False)
    if name == "frozen":
        return base
    if name == "single_lora":
        return LoRALinear.from_linear(base, rank=8, alpha=16.0)
    if name == "matched_lora":
        return LoRALinear.from_linear(base, rank=34, alpha=16.0)
    if name == "dense_molora":
        return DenseMoLoRALinear.from_linear(base, n_experts=4, rank=8, alpha=16.0)
    raise ValueError(name)


def trainable(model):
    return [] if isinstance(model, nn.Linear) else model.trainable_parameters()


def timeit(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    q = statistics.quantiles(ts, n=4)
    return {"median_s": statistics.median(ts), "p25_s": q[0], "p75_s": q[2], "min_s": ts[0], "max_s": ts[-1], "n": repeats}


def bench_model(name, in_dim, out_dim, batch, warmup, repeats, dtype="float32"):
    model = build(name, in_dim, out_dim)
    params = trainable(model)
    rng = np.random.default_rng(0)
    x = ndl.Tensor(rng.standard_normal((batch, in_dim)).astype(dtype), dtype=dtype, requires_grad=False)
    y = ndl.Tensor(rng.standard_normal((batch, out_dim)).astype(dtype), dtype=dtype, requires_grad=False)
    # make sure adapters are non-zero so every code path (incl. router gradient) is live
    if isinstance(model, LoRALinear):
        model.load_adapter(B=rng.standard_normal((model.rank, out_dim)).astype(dtype))
    if isinstance(model, DenseMoLoRALinear):
        for i in range(model.n_experts):
            model.load_expert(i, B=rng.standard_normal((model.rank, out_dim)).astype(dtype))

    fwd = timeit(lambda: model(x), warmup, repeats)

    def step():
        if params:
            opt.reset_grad()
        loss = ((model(x) - y) ** 2).sum() / (batch * out_dim)
        if params:
            loss.backward()
            opt.step()
        return loss

    opt = ndl.optim.Adam(params, lr=1e-3) if params else None
    trn = timeit(step, warmup, repeats)

    # Python-level peak allocation of one training step (NumPy buffers included)
    step()
    tracemalloc.start()
    step()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_train = int(sum(int(np.prod(p.shape)) for p in params))
    n_total = int(sum(int(np.prod(p.shape)) for p in model.parameters()))
    return {
        "model": name, "batch_size": batch, "forward": fwd, "train_step": trn,
        "forward_examples_per_s": batch / fwd["median_s"], "train_examples_per_s": batch / trn["median_s"],
        "trainable_parameters": n_train, "total_parameters": n_total,
        "trainable_parameter_bytes_f32": 4 * n_train, "total_parameter_bytes_f32": 4 * n_total,
        "train_step_peak_python_alloc_bytes": int(peak),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dim", type=int, default=128)
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 256])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "benchmark_dense_molora"))
    args = ap.parse_args(argv)

    run_dir = os.path.join(args.output_dir, dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    rows = []
    for batch in args.batch_sizes:
        for name in ("frozen", "single_lora", "matched_lora", "dense_molora"):
            r = bench_model(name, args.in_dim, args.out_dim, batch, args.warmup, args.repeats)
            rows.append(r)
            print(f"batch {batch:4d} {name:13s} fwd {1e3 * r['forward']['median_s']:7.3f} ms  "
                  f"step {1e3 * r['train_step']['median_s']:7.3f} ms  train ex/s {r['train_examples_per_s']:9.0f}  "
                  f"params {r['trainable_parameters']:5d}  step peak alloc {r['train_step_peak_python_alloc_bytes'] / 1e6:6.2f} MB")
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "numpy": np.__version__, "needle": molora.needle_info(), "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
           "blas_threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
           "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with open(os.path.join(run_dir, "benchmark.json"), "w") as f:
        json.dump({"args": vars(args), "environment": env, "rows": rows}, f, indent=1)
    L = ["# CPU benchmark", "", f"timestamp {env['timestamp']}; numpy {env['numpy']}; {env['cpu_count']} CPUs; "
         f"process peak RSS {env['process_peak_rss_mb']:.0f} MB (whole process, all models)", "",
         "| batch | model | trainable params | total params | forward median (ms) | fwd IQR (ms) | train step median (ms) | step IQR (ms) | train ex/s | step peak Python alloc (MB) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f_, t_ = r["forward"], r["train_step"]
        L.append(f"| {r['batch_size']} | {r['model']} | {r['trainable_parameters']} | {r['total_parameters']} | {1e3 * f_['median_s']:.3f} | "
                 f"{1e3 * f_['p25_s']:.3f}–{1e3 * f_['p75_s']:.3f} | {1e3 * t_['median_s']:.3f} | {1e3 * t_['p25_s']:.3f}–{1e3 * t_['p75_s']:.3f} | "
                 f"{r['train_examples_per_s']:.0f} | {r['train_step_peak_python_alloc_bytes'] / 1e6:.2f} |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(run_dir, "benchmark.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"[benchmark] wrote {run_dir}")


if __name__ == "__main__":
    main()
