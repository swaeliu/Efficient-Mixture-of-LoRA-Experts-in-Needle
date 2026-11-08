#!/usr/bin/env python
"""Phase 4C — systems benchmark of adapter variants inside the tiny transformer (CPU, pinned BLAS threads).

Measures per method and (batch, seq_len): inference forward latency, training-step
latency (forward + backward + Adam on trainable params), tokens/s, per-step peak
Python/NumPy allocation, trainable parameters, and a forward-time attribution
(adapter layers vs the rest of the backbone) obtained by timing the adapter
layers' forward calls in place; for sparse layers the layer profiler further
splits router / expert / dispatch time.

    python experiments/benchmark_transformer.py [--threads 1] [--repeats 20] [--warmup 3]
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> models/
import needle as ndl  # noqa: E402
from models import TinyTransformer, adapter_layers, adapter_trainable_parameters, count_parameters, freeze_backbone, inject_adapters  # noqa: E402
from molora import SparseMoLoRALinear  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V, D, H, L, DFF = 16, 64, 4, 2, 128
METHODS = {
    "frozen": None,
    "lora_r4": dict(kind="lora", rank=4),
    "lora_r18": dict(kind="lora", rank=18),
    "dense_E4": dict(kind="dense", rank=4, n_experts=4),
    "sparse_E4_k1": dict(kind="sparse", rank=4, n_experts=4, top_k=1, router_grad_mode="straight_through"),
    "sparse_E4_k2": dict(kind="sparse", rank=4, n_experts=4, top_k=2),
    "sparse_E4_k4": dict(kind="sparse", rank=4, n_experts=4, top_k=4),
    "dense_E8": dict(kind="dense", rank=4, n_experts=8),
    "sparse_E8_k2": dict(kind="sparse", rank=4, n_experts=8, top_k=2),
}
SHAPES = [(8, 16), (8, 32), (32, 16), (32, 32)]


def build(spec, seq_len):
    np.random.seed(0)
    m = TinyTransformer(V, D, H, L, DFF, 32)
    freeze_backbone(m)
    if spec is not None:
        np.random.seed(1)
        layers = inject_adapters(m, **{k: v for k, v in spec.items()}, alpha=8.0)
        rng = np.random.default_rng(2)
        for layer in layers:  # non-zero adapters and realistic routers so every path is live
            if hasattr(layer, "experts"):
                for i in range(layer.n_experts):
                    layer.load_expert(i, B=rng.standard_normal((layer.rank, layer.out_features)).astype(np.float32) * 0.05)
                layer.load_router(weight=rng.standard_normal((layer.in_features, layer.n_experts)).astype(np.float32) * 0.3)
            else:
                layer.load_adapter(B=rng.standard_normal((layer.rank, layer.out_features)).astype(np.float32) * 0.05)
    return m


def stats(ts):
    ts = sorted(ts)
    q = statistics.quantiles(ts, n=4) if len(ts) >= 4 else [ts[0], ts[len(ts) // 2], ts[-1]]
    return {"median_s": statistics.median(ts), "p25_s": q[0], "p75_s": q[2], "n": len(ts)}


def timeit(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return stats(ts)


def bench(name, spec, B, T, warmup, repeats):
    model = build(spec, T)
    params = adapter_trainable_parameters(model) if spec is not None else []
    rng = np.random.default_rng(3)
    x = rng.integers(0, V, (B, T))
    y = rng.integers(0, V, (B, T))
    n_tok = B * T
    opt = ndl.optim.Adam(params, lr=1e-3) if params else None
    fwd = timeit(lambda: model(x), warmup, repeats)

    def step():
        if params:
            opt.reset_grad()
        loss = model.loss(model(x), y)
        if params:
            loss.backward()
            opt.step()

    stp = timeit(step, warmup, repeats)
    step()
    tracemalloc.start()
    step()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # forward attribution: time adapter layers' forward in place
    layers = adapter_layers(model)
    acc = {"adapter_s": 0.0}
    sub = {"router_s": 0.0, "expert_s": 0.0, "dispatch_s": 0.0}
    for layer in layers:
        orig = layer.forward

        def wrapped(inp, _orig=orig, _layer=layer):
            t0 = time.perf_counter()
            out = _orig(inp)
            acc["adapter_s"] += time.perf_counter() - t0
            if isinstance(_layer, SparseMoLoRALinear) and _layer.last_timing:
                for k in sub:
                    sub[k] += _layer.last_timing[k]
            return out
        layer.forward = wrapped
        if isinstance(layer, SparseMoLoRALinear):
            layer.profile = True
    totals = []
    for _ in range(repeats):
        acc["adapter_s"] = 0.0
        for k in sub:
            sub[k] = 0.0
        t0 = time.perf_counter()
        model(x)
        totals.append((time.perf_counter() - t0, acc["adapter_s"], dict(sub)))
    for layer in layers:
        del layer.forward
        if isinstance(layer, SparseMoLoRALinear):
            layer.profile = False
    med = sorted(totals, key=lambda t: t[0])[len(totals) // 2]
    attribution = {"forward_total_s": med[0], "adapter_s": med[1], "backbone_s": med[0] - med[1], **{k: med[2][k] for k in sub}}

    return {"method": name, "batch": B, "seq_len": T, "tokens": n_tok, "forward": fwd, "train_step": stp,
            "inference_tokens_per_s": n_tok / fwd["median_s"], "train_tokens_per_s": n_tok / stp["median_s"],
            "trainable_parameters": count_parameters(params), "total_parameters": count_parameters(model.parameters()),
            "train_step_peak_python_alloc_bytes": int(peak), "forward_attribution_s": attribution,
            "adapter_share_of_forward": attribution["adapter_s"] / attribution["forward_total_s"] if attribution["forward_total_s"] else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "benchmark_transformer"))
    args = ap.parse_args(argv)
    run_dir = os.path.join(args.output_dir, dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    rows = []
    for B, T in SHAPES:
        for name, spec in METHODS.items():
            r = bench(name, spec, B, T, args.warmup, args.repeats)
            rows.append(r)
            a = r["forward_attribution_s"]
            print(f"B={B:2d} T={T:2d} {name:13s} fwd {1e3 * r['forward']['median_s']:7.2f} ms  step {1e3 * r['train_step']['median_s']:7.2f} ms  "
                  f"train tok/s {r['train_tokens_per_s']:7.0f}  peak {r['train_step_peak_python_alloc_bytes'] / 1e6:6.2f} MB  adapter share {100 * (r['adapter_share_of_forward'] or 0):4.0f}%", flush=True)
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(), "blas_threads": THREADS,
           "numpy": np.__version__, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
           "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with open(os.path.join(run_dir, "benchmark.json"), "w") as f:
        json.dump({"args": vars(args), "environment": env, "rows": rows}, f, indent=1)
    L = ["# Phase 4C transformer benchmark", "", f"{env['timestamp']}; BLAS threads {THREADS}; numpy {env['numpy']}; process peak RSS {env['process_peak_rss_mb']:.0f} MB", "",
         "| B | T | method | trainable params | forward ms (IQR) | step ms (IQR) | inference tok/s | train tok/s | step peak alloc MB | adapter share of forward | adapter ms (router / expert / dispatch) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f_, s_, a = r["forward"], r["train_step"], r["forward_attribution_s"]
        sub = f"{1e3 * a['adapter_s']:.2f} ({1e3 * a['router_s']:.2f} / {1e3 * a['expert_s']:.2f} / {1e3 * a['dispatch_s']:.2f})" if r["method"].startswith("sparse") else f"{1e3 * a['adapter_s']:.2f}"
        L.append(f"| {r['batch']} | {r['seq_len']} | {r['method']} | {r['trainable_parameters']} | {1e3 * f_['median_s']:.2f} ({1e3 * f_['p25_s']:.2f}–{1e3 * f_['p75_s']:.2f}) | "
                 f"{1e3 * s_['median_s']:.2f} ({1e3 * s_['p25_s']:.2f}–{1e3 * s_['p75_s']:.2f}) | {r['inference_tokens_per_s']:.0f} | {r['train_tokens_per_s']:.0f} | "
                 f"{r['train_step_peak_python_alloc_bytes'] / 1e6:.2f} | {100 * (r['adapter_share_of_forward'] or 0):.0f}% | {sub} |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(run_dir, "benchmark.md"), "w") as f:
        f.write(text)
    print(text)
    print("[benchmark] wrote", run_dir)


if __name__ == "__main__":
    main()
