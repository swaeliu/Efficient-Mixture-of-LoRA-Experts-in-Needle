#!/usr/bin/env python
"""Phase 3 CPU runtime / memory benchmark: sparse top-k dispatch vs dense MoLoRA (Task 3B.6).

Measures, per model and input shape, the median (and inter-quartile range) wall
time of the forward pass, the backward pass alone, and a full optimizer step
(forward + backward + Adam on trainable parameters), tokens/s, per-step peak
Python/NumPy allocation (tracemalloc), parameter counts, actual expert matmuls
executed, and a forward-time breakdown (backbone / router / expert / dispatch)
from the layer's own profiler. BLAS threads are pinned (default 1) BEFORE NumPy
is imported so timings are not dominated by thread scheduling.

    python experiments/benchmark_sparse_molora.py [--threads 1] [--repeats 50] [--warmup 5]
"""
from __future__ import annotations

import argparse
import os
import sys


def _early_threads():
    n = "1"
    if "--threads" in sys.argv:
        n = sys.argv[sys.argv.index("--threads") + 1]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = n
    return int(n)


THREADS = _early_threads()

import datetime as dt  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402

import numpy as np  # noqa: E402

import molora  # noqa: E402
import needle as ndl  # noqa: E402
import needle.nn as nn  # noqa: E402
from molora import DenseMoLoRALinear, LoRALinear, SparseMoLoRALinear  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = OUT = 128

MODELS = {
    "frozen": dict(kind="frozen"),
    "single_lora_r8": dict(kind="lora", rank=8),
    "matched_lora_r34": dict(kind="lora", rank=34),
    "dense_E4": dict(kind="dense", E=4, rank=8),
    "sparse_E4_k1": dict(kind="sparse", E=4, k=1, rank=8, mode="dispatch"),
    "sparse_E4_k2": dict(kind="sparse", E=4, k=2, rank=8, mode="dispatch"),
    "sparse_E4_k4": dict(kind="sparse", E=4, k=4, rank=8, mode="dispatch"),
    "sparse_E4_k2_reference": dict(kind="sparse", E=4, k=2, rank=8, mode="reference"),
}
SCALING = {
    **{f"sparse_E8_k{k}": dict(kind="sparse", E=8, k=k, rank=8, mode="dispatch") for k in (1, 2, 4, 8)},
    "dense_E8": dict(kind="dense", E=8, rank=8),
    **{f"sparse_E4_k2_r{r}": dict(kind="sparse", E=4, k=2, rank=r, mode="dispatch") for r in (4, 16)},
    **{f"dense_E4_r{r}": dict(kind="dense", E=4, rank=r) for r in (4, 16)},
}
SHAPES = [(32,), (128,), (512,), (8, 16), (8, 64), (32, 64)]


def build(spec, seed=0):
    np.random.seed(seed)
    base = nn.Linear(IN, OUT, bias=False)
    rng = np.random.default_rng(seed)
    if spec["kind"] == "frozen":
        return base
    if spec["kind"] == "lora":
        m = LoRALinear.from_linear(base, rank=spec["rank"], alpha=16.0)
        m.load_adapter(B=rng.standard_normal((spec["rank"], OUT)) * 0.1)
        return m
    if spec["kind"] == "dense":
        m = DenseMoLoRALinear.from_linear(base, n_experts=spec["E"], rank=spec["rank"], alpha=16.0)
    else:
        m = SparseMoLoRALinear.from_linear(base, n_experts=spec["E"], rank=spec["rank"], alpha=16.0, top_k=spec["k"],
                                           execution_mode=spec["mode"])
    for i in range(spec["E"]):
        m.load_expert(i, B=rng.standard_normal((spec["rank"], OUT)) * 0.1)  # non-zero so every path is live
    m.load_router(weight=rng.standard_normal((IN, spec["E"])) * 0.3)  # realistic, non-uniform routing
    return m


def trainable(model):
    return [] if isinstance(model, nn.Linear) else model.trainable_parameters()


def _stats(ts):
    ts = sorted(ts)
    q = statistics.quantiles(ts, n=4)
    return {"median_s": statistics.median(ts), "p25_s": q[0], "p75_s": q[2], "min_s": ts[0], "n": len(ts)}


def timeit(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return _stats(ts)


def bench(name, spec, shape, warmup, repeats):
    model = build(spec)
    params = trainable(model)
    rng = np.random.default_rng(1)
    x = ndl.Tensor(rng.standard_normal(shape + (IN,)).astype(np.float32), dtype="float32", requires_grad=False)
    y = ndl.Tensor(rng.standard_normal(shape + (OUT,)).astype(np.float32), dtype="float32", requires_grad=False)
    n_tokens = int(np.prod(shape))
    opt = ndl.optim.Adam(params, lr=1e-3) if params else None

    def loss_fn():
        return ((model(x) - y) ** 2).sum() / (n_tokens * OUT)

    fwd = timeit(lambda: model(x), warmup, repeats)

    bwd = None
    if params:
        def backward_only():
            loss = loss_fn()
            t0 = time.perf_counter()
            loss.backward()
            return time.perf_counter() - t0
        for _ in range(warmup):
            backward_only()
        bwd = _stats([backward_only() for _ in range(repeats)])

    def step():
        if params:
            opt.reset_grad()
        loss = loss_fn()
        if params:
            loss.backward()
            opt.step()

    stp = timeit(step, warmup, repeats)
    step()
    tracemalloc.start()
    step()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    row = {"model": name, "spec": spec, "shape": list(shape), "n_tokens": n_tokens,
           "forward": fwd, "backward": bwd, "train_step": stp,
           "forward_tokens_per_s": n_tokens / fwd["median_s"], "train_tokens_per_s": n_tokens / stp["median_s"],
           "trainable_parameters": int(sum(int(np.prod(p.shape)) for p in params)),
           "total_parameters": int(sum(int(np.prod(p.shape)) for p in model.parameters())),
           "train_step_peak_python_alloc_bytes": int(peak)}
    if isinstance(model, SparseMoLoRALinear):
        row["expert_stats"] = {k: model.last_stats[k] for k in ("tokens_per_expert", "expert_matmuls_executed", "experts_skipped",
                                                              "expert_flop_fraction_vs_dense", "theoretical_flop_fraction")}
        model.profile = True
        prof = []
        for _ in range(repeats):
            model(x)
            prof.append(model.last_timing)
        model.profile = False
        row["forward_breakdown_median_s"] = {k: statistics.median(p[k] for p in prof) for k in prof[0]}
    elif isinstance(model, DenseMoLoRALinear):
        row["expert_stats"] = {"expert_matmuls_executed": 2 * model.n_experts, "experts_skipped": 0,
                               "expert_flop_fraction_vs_dense": 1.0, "theoretical_flop_fraction": 1.0}
    return row


def fmt_ms(d):
    return "—" if d is None else f"{1e3 * d['median_s']:.3f} ({1e3 * d['p25_s']:.3f}–{1e3 * d['p75_s']:.3f})"


def plot(rows, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAL = {"grid": "#e4e3df", "text": "#0b0b0b", "muted": "#52514e", "surface": "#fcfcfb"}
    colors = {"(32,)": "#2a78d6", "(128,)": "#eb6834", "(512,)": "#1baf7a", "(8, 16)": "#eda100", "(8, 64)": "#e87ba4", "(32, 64)": "#6250d6"}
    by = {(r["model"], str(tuple(r["shape"]))): r for r in rows}
    shapes = [str(tuple(s)) for s in SHAPES]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=PAL["surface"])
    for ax, (key, ylabel) in zip(axes, [("forward", "forward latency (ms)"), ("train_step", "train-step latency (ms)"),
                                        ("mem", "step peak Python alloc (MB)")]):
        ax.set_facecolor(PAL["surface"])
        ax.grid(True, color=PAL["grid"], lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for shp in shapes:
            ks, vals = [1, 2, 4], []
            for k in ks:
                r = by[(f"sparse_E4_k{k}", shp)]
                vals.append(1e3 * r[key]["median_s"] if key != "mem" else r["train_step_peak_python_alloc_bytes"] / 1e6)
            d = by[("dense_E4", shp)]
            dense = 1e3 * d[key]["median_s"] if key != "mem" else d["train_step_peak_python_alloc_bytes"] / 1e6
            ax.plot(ks, vals, color=colors[shp], lw=1.8, marker="o", ms=4, label=f"sparse, {int(np.prod(eval(shp)))} tokens {shp}")
            ax.scatter([4.4], [dense], color=colors[shp], marker="s", s=30, zorder=3)
        ax.set_xticks([1, 2, 4, 4.4], ["k=1", "k=2", "k=4", "dense"], fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, color=PAL["muted"], fontsize=9)
        ax.tick_params(colors=PAL["muted"], labelsize=8)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.suptitle("Sparse top-k dispatch vs dense MoLoRA (E=4, r=8, 128x128, CPU, 1 BLAS thread); squares = dense",
                 fontsize=10, color=PAL["text"])
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "latency_memory_vs_k.png"), dpi=130, bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "benchmark_sparse_molora"))
    args = ap.parse_args(argv)
    run_dir = os.path.join(args.output_dir, dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    for shape in SHAPES:
        for name, spec in MODELS.items():
            r = bench(name, spec, shape, args.warmup, args.repeats)
            rows.append(r)
            print(f"{str(shape):10s} {name:24s} fwd {1e3 * r['forward']['median_s']:7.3f} ms  step {1e3 * r['train_step']['median_s']:7.3f} ms  "
                  f"tok/s {r['train_tokens_per_s']:9.0f}  peak {r['train_step_peak_python_alloc_bytes'] / 1e6:6.2f} MB", flush=True)
    scaling_rows = []
    if not args.no_scaling:
        for shape in [(512,), (32, 64)]:
            for name, spec in SCALING.items():
                r = bench(name, spec, shape, args.warmup, args.repeats)
                scaling_rows.append(r)
                print(f"[scaling] {str(shape):10s} {name:20s} fwd {1e3 * r['forward']['median_s']:7.3f} ms  step {1e3 * r['train_step']['median_s']:7.3f} ms  "
                      f"peak {r['train_step_peak_python_alloc_bytes'] / 1e6:6.2f} MB", flush=True)

    env = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
           "cpu_count": os.cpu_count(), "blas_threads": THREADS, "numpy": np.__version__, "needle": molora.needle_info(),
           "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
           "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with open(os.path.join(run_dir, "benchmark.json"), "w") as f:
        json.dump({"args": vars(args), "environment": env, "rows": rows, "scaling_rows": scaling_rows}, f, indent=1)

    L = ["# Phase 3 CPU benchmark", "", f"{env['timestamp']}; numpy {env['numpy']}; BLAS threads {THREADS}; {env['processor']}; "
         f"process peak RSS {env['process_peak_rss_mb']:.0f} MB (whole process)", "",
         "| shape | model | trainable params | forward ms (IQR) | backward ms (IQR) | step ms (IQR) | train tok/s | step peak alloc MB | expert matmuls | skipped | FLOP frac |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows + scaling_rows:
        es = r.get("expert_stats", {})
        L.append(f"| {tuple(r['shape'])} | {r['model']} | {r['trainable_parameters']} | {fmt_ms(r['forward'])} | {fmt_ms(r['backward'])} | "
                 f"{fmt_ms(r['train_step'])} | {r['train_tokens_per_s']:.0f} | {r['train_step_peak_python_alloc_bytes'] / 1e6:.2f} | "
                 f"{es.get('expert_matmuls_executed', '—')} | {es.get('experts_skipped', '—')} | {es.get('expert_flop_fraction_vs_dense', '—')} |")
    L += ["", "## Forward-time breakdown (sparse models, medians, ms)", "",
          "| shape | model | backbone | router | expert | dispatch (gather+scatter) | total |", "|---|---|---|---|---|---|---|"]
    for r in rows + scaling_rows:
        b = r.get("forward_breakdown_median_s")
        if b:
            L.append(f"| {tuple(r['shape'])} | {r['model']} | {1e3 * b['backbone_s']:.3f} | {1e3 * b['router_s']:.3f} | {1e3 * b['expert_s']:.3f} | "
                     f"{1e3 * b['dispatch_s']:.3f} | {1e3 * b['forward_total_s']:.3f} |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(run_dir, "benchmark.md"), "w") as f:
        f.write(text)
    plot(rows, run_dir)
    print(text)
    print(f"[benchmark] wrote {run_dir}")


if __name__ == "__main__":
    main()
