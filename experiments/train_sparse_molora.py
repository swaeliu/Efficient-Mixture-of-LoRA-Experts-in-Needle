#!/usr/bin/env python
"""Phase 3 sparse-vs-dense routing experiment on the Phase 2 heterogeneous task.

Data generation, splits, planted corrections and frozen W0 come from
``experiments/train_dense_molora.make_problem`` unchanged. Methods:

    frozen, single_lora (r=8), matched_lora (r=34), dense_molora (4 x r=8),
    sparse_top1 / sparse_top2 / sparse_top4 (4 x r=8, top-k, dispatch mode)

Modes::

    python experiments/train_sparse_molora.py --pilot            # LR (and top-1 router-grad mode) sweep, VALIDATION only
    python experiments/train_sparse_molora.py                    # confirmatory run on TEST data, seeds from config
    python experiments/train_sparse_molora.py --sweep experts    # E=8, k in {1,2,4,8}
    python experiments/train_sparse_molora.py --sweep rank       # top-2 with ranks {4,8,16}
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import time
import zlib

import numpy as np
import yaml

import needle as ndl
import needle.nn as nn
from molora import DenseMoLoRALinear, LoRALinear, SparseMoLoRALinear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_dense_molora import PAL, _style, environment_info, make_problem, mse_by_regime, mse_needle, predict  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = {"frozen": "A frozen", "single_lora": "B LoRA r=8", "matched_lora": "D LoRA r=34", "dense_molora": "C dense 4x8",
         "sparse_top1": "S1 sparse top-1", "sparse_top2": "S2 sparse top-2", "sparse_top4": "S4 sparse top-4"}
COLOR = {"frozen": "#52514e", "single_lora": "#2a78d6", "matched_lora": "#eb6834", "dense_molora": "#1baf7a",
         "sparse_top1": "#eda100", "sparse_top2": "#e87ba4", "sparse_top4": "#6250d6"}


# --------------------------------------------------------------------------- #
def build_model(name: str, spec: dict, problem: dict, cfg: dict, seed: int):
    m, b = cfg["model"], cfg["baselines"]
    base = problem["base"]
    np.random.seed(seed * 1000 + zlib.crc32(name.encode()) % 997)  # stable across processes (hash() is not)
    if name == "frozen":
        return base
    if name == "single_lora":
        return LoRALinear.from_linear(base, rank=b["single_lora_rank"], alpha=m["alpha"])
    if name == "matched_lora":
        return LoRALinear.from_linear(base, rank=b["matched_lora_rank"], alpha=m["alpha"])
    if name == "dense_molora":
        return DenseMoLoRALinear.from_linear(base, n_experts=m["n_experts"], rank=m["rank"], alpha=m["alpha"],
                                             router_temperature=m["router_temperature"], router_init=m["router_init"],
                                             b_init_std=m["b_init_std"])
    if name.startswith("sparse"):
        return SparseMoLoRALinear.from_linear(
            base, n_experts=spec.get("n_experts", m["n_experts"]), rank=spec.get("rank", m["rank"]), alpha=m["alpha"],
            top_k=spec["top_k"], router_temperature=m["router_temperature"], router_init=m["router_init"],
            b_init_std=m["b_init_std"], execution_mode=spec.get("execution_mode", m["execution_mode"]),
            router_grad_mode=spec.get("router_grad_mode", "renormalized"))
    raise ValueError(name)


def trainable(model):
    return [] if isinstance(model, nn.Linear) else model.trainable_parameters()


def _entropy(p):
    return float((-(p * np.log(np.clip(p, 1e-300, None))).sum(-1)).mean())


def _router_grad_norm(model):
    g = [p.grad.numpy() for p in model.router_parameters() if p.grad is not None]
    return float(np.sqrt(sum(float((a.astype(np.float64) ** 2).sum()) for a in g))) if g else 0.0


def routing_by_regime(model, split, R, dtype):
    """Regime x expert matrices (dense probs and sparse/selection weights), entropy, utilisation."""
    x = ndl.Tensor(split["x"], dtype=dtype, requires_grad=False)
    if isinstance(model, SparseMoLoRALinear):
        d = model.routing_diagnostics(x)
        p, w, mask = d["dense_probs"], d["sparse_weights"], d["mask"]
    else:
        p = model.router_probabilities(x).astype(np.float64)
        w, mask = p, np.ones_like(p)  # dense: every expert "selected"
    z = split["z"]
    E = p.shape[1]
    out = {
        "regime_expert_mean_dense_prob": [p[z == r].mean(0).tolist() for r in range(R)],
        "regime_expert_mean_weight": [w[z == r].mean(0).tolist() for r in range(R)],
        "regime_expert_selection_rate": [mask[z == r].mean(0).tolist() for r in range(R)],
        "regime_entropy_nats": [_entropy(p[z == r]) for r in range(R)],
        "overall_entropy_nats": _entropy(p), "max_entropy_nats": float(np.log(E)),
        "overall_utilization": mask.mean(0).tolist(),  # fraction of tokens selecting each expert
        "overall_argmax_utilization": (np.bincount(p.argmax(-1), minlength=E) / len(p)).tolist(),
        "tokens_per_expert": mask.sum(0).astype(int).tolist(),
    }
    return out


def expert_divergence(model):
    D = model.expert_delta_weights()
    norms = [float(np.linalg.norm(d)) for d in D]
    rel = [float(np.linalg.norm(D[i] - D[j]) / (0.5 * (norms[i] + norms[j]) or 1.0))
           for i in range(len(D)) for j in range(i + 1, len(D))]
    return {"expert_delta_norms": norms, "pairwise_relative_distance_mean": float(np.mean(rel)) if rel else 0.0}


# --------------------------------------------------------------------------- #
def train_one(name, spec, problem, cfg, seed, lr, *, eval_split):
    tr, dtype, R = cfg["training"], cfg["dtype"], cfg["data"]["n_regimes"]
    model = build_model(name, spec, problem, cfg, seed)
    params = trainable(model)
    is_moe = isinstance(model, DenseMoLoRALinear)
    is_sparse = isinstance(model, SparseMoLoRALinear)
    W0 = problem["base"].weight.numpy().copy()
    x_tr, y_tr, ev = problem["train"]["x"], problem["train"]["y"], problem[eval_split]
    n, bs = len(x_tr), tr["batch_size"]
    rng = np.random.default_rng(seed + 12345)
    track_x = ndl.Tensor(problem["val"]["x"][:256], dtype=dtype, requires_grad=False)

    hist = {"train_loss": [], "eval_mse": [(0, mse_by_regime(predict(model, ev["x"], dtype), ev, R)[0])],
            "router_grad_norm": [], "routing_entropy": []}
    exec_acc = {"expert_rows_executed": 0, "expert_rows_dense": 0, "expert_matmuls_executed": 0,
                "expert_matmuls_dense": 0, "experts_skipped": 0, "forwards": 0}
    t0 = time.perf_counter()
    if params:
        opt = ndl.optim.Adam(params, lr=lr)
        perm, ptr = rng.permutation(n), 0
        for step in range(1, tr["max_steps"] + 1):
            if ptr + bs > n:
                perm, ptr = rng.permutation(n), 0
            idx = perm[ptr: ptr + bs]
            ptr += bs
            xb = ndl.Tensor(x_tr[idx], dtype=dtype, requires_grad=False)
            yb = ndl.Tensor(y_tr[idx], dtype=dtype, requires_grad=False)
            opt.reset_grad()
            loss = mse_needle(model(xb), yb)
            loss.backward()
            if is_sparse:
                st = model.last_stats
                exec_acc["expert_rows_executed"] += sum(st["expert_rows_executed"])
                exec_acc["expert_rows_dense"] += st["n_experts"] * st["n_tokens"]
                exec_acc["expert_matmuls_executed"] += st["expert_matmuls_executed"]
                exec_acc["expert_matmuls_dense"] += 2 * st["n_experts"]
                exec_acc["experts_skipped"] += st["experts_skipped"]
                exec_acc["forwards"] += 1
            if is_moe and (step == 1 or step % cfg["diagnostics"]["track_every"] == 0):
                hist["router_grad_norm"].append((step, _router_grad_norm(model)))
            opt.step()
            hist["train_loss"].append(float(loss.numpy()))
            if step % tr["eval_every"] == 0 or step == tr["max_steps"]:
                hist["eval_mse"].append((step, mse_by_regime(predict(model, ev["x"], dtype), ev, R)[0]))
            if is_moe and step % cfg["diagnostics"]["track_every"] == 0:
                hist["routing_entropy"].append((step, _entropy(model.router_probabilities(track_x).astype(np.float64))))
    elapsed = time.perf_counter() - t0

    out = {"method": name, "spec": spec, "seed": seed, "learning_rate": lr if params else None,
           "trainable_parameters": int(sum(int(np.prod(p.shape)) for p in params)),
           "total_parameters": int(sum(int(np.prod(p.shape)) for p in model.parameters())),
           "frozen_backbone_unchanged": bool(np.array_equal(W0, problem["base"].weight.numpy())),
           "elapsed_seconds": elapsed, "history": hist}
    overall, per = mse_by_regime(predict(model, problem["train"]["x"][:2048], dtype), {k: v[:2048] for k, v in problem["train"].items()}, R)
    out["train_mse"] = overall
    for split in ("val", "test"):
        if split == "test" and eval_split != "test":
            continue
        overall, per = mse_by_regime(predict(model, problem[split]["x"], dtype), problem[split], R)
        out[f"{split}_mse"], out[f"{split}_mse_by_regime"] = overall, per
    if is_moe:
        out["routing_val"] = routing_by_regime(model, problem["val"], R, dtype)
        if eval_split == "test":
            out["routing_test"] = routing_by_regime(model, problem["test"], R, dtype)
        out["expert_divergence"] = expert_divergence(model)
        out["parameter_summary"] = model.parameter_summary()
    if is_sparse:
        f = exec_acc["forwards"] or 1
        out["execution"] = {**exec_acc,
                            "expert_flop_fraction_vs_dense": exec_acc["expert_rows_executed"] / max(exec_acc["expert_rows_dense"], 1),
                            "mean_experts_skipped_per_forward": exec_acc["experts_skipped"] / f,
                            "theoretical_flop_fraction": model.top_k / model.n_experts}
    out["all_finite"] = bool(np.all(np.isfinite(hist["train_loss"])) and np.all(np.isfinite([v for _, v in hist["eval_mse"]])))
    return out


# --------------------------------------------------------------------------- #
def run_dir_for(cfg, tag):
    root = cfg["experiment"]["output_dir"]
    root = root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)
    d = os.path.join(root, cfg["experiment"]["name"] + (f"_{tag}" if tag else ""), dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def pilot(cfg, run_dir):
    seed = cfg["pilot"]["seed"]
    problem = make_problem(cfg, seed)
    table = {}
    for name, spec in cfg["methods"].items():
        if not name.startswith("sparse"):
            continue
        modes = cfg["pilot"]["top1_router_grad_modes"] if spec["top_k"] == 1 else [spec.get("router_grad_mode", "renormalized")]
        for mode in modes:
            for lr in cfg["pilot"]["learning_rates"]:
                s = dict(spec, router_grad_mode=mode)
                r = train_one(name, s, problem, cfg, seed, lr, eval_split="val")
                key = f"{mode}|{lr}"
                table.setdefault(name, {})[key] = {"val_mse": r["val_mse"], "elapsed_seconds": r["elapsed_seconds"],
                                                   "all_finite": r["all_finite"],
                                                   "routing_entropy": r["routing_val"]["overall_entropy_nats"],
                                                   "utilization": r["routing_val"]["overall_utilization"]}
                print(f"[pilot] {name:12s} {mode:17s} lr={lr:<6} val MSE {r['val_mse']:.4e}  entropy {r['routing_val']['overall_entropy_nats']:.3f}  "
                      f"util {np.round(r['routing_val']['overall_utilization'], 2).tolist()}  ({r['elapsed_seconds']:.1f}s)")
    best = {n: min(d.items(), key=lambda kv: kv[1]["val_mse"])[0] for n, d in table.items()}
    print("[pilot] best (router_grad_mode|lr) by validation MSE:", best)
    with open(os.path.join(run_dir, "pilot_results.json"), "w") as f:
        json.dump({"seed": seed, "results": table, "best": best, "environment": environment_info()}, f, indent=1)


def confirmatory(cfg, run_dir, no_plot=False):
    lrs, sc, R = cfg["training"]["learning_rate"], cfg["success_criteria"], cfg["data"]["n_regimes"]
    results = {}
    for seed in cfg["experiment"]["seeds"]:
        problem = make_problem(cfg, seed)
        results[seed] = {}
        for name, spec in cfg["methods"].items():
            r = train_one(name, spec, problem, cfg, seed, lrs.get(name), eval_split="test")
            results[seed][name] = r
            ex = r.get("execution")
            extra = f"  flop frac {ex['expert_flop_fraction_vs_dense']:.3f} skipped/fwd {ex['mean_experts_skipped_per_forward']:.2f}" if ex else ""
            print(f"[seed {seed}] {name:12s} test {r['test_mse']:.4e} val {r['val_mse']:.4e} train {r['train_mse']:.4e} "
                  f"params {r['trainable_parameters']:5d} {r['elapsed_seconds']:.1f}s W0 {r['frozen_backbone_unchanged']}{extra}")
    s0 = cfg["experiment"]["seeds"][0]
    repeat = None
    if cfg["experiment"].get("repeat_first_seed"):
        repeat = train_one("sparse_top2", cfg["methods"]["sparse_top2"], make_problem(cfg, s0), cfg, s0, lrs["sparse_top2"], eval_split="test")
        print(f"[repeat seed {s0}] sparse_top2 test MSE {repeat['test_mse']:.6e} vs {results[s0]['sparse_top2']['test_mse']:.6e}")

    def ratio(a, b):
        return {str(s): results[s][a]["test_mse"] / results[s][b]["test_mse"] for s in results}

    r_t2_dense, r_t2_matched, r_t4_dense = ratio("sparse_top2", "dense_molora"), ratio("sparse_top2", "matched_lora"), ratio("sparse_top4", "dense_molora")
    checks = {
        "top2_within_2x_of_dense_every_seed": all(v <= sc["max_test_mse_ratio_top2_vs_dense"] for v in r_t2_dense.values()),
        "top2_below_0.1x_matched_lora_every_seed": all(v <= sc["max_test_mse_ratio_top2_vs_matched_lora"] for v in r_t2_matched.values()),
        "top4_within_1.5x_of_dense_every_seed": all(v <= sc["max_test_mse_ratio_top4_vs_dense"] for v in r_t4_dense.values()),
        "flop_fraction_equals_k_over_E": all(abs(results[s][n]["execution"]["expert_flop_fraction_vs_dense"] - results[s][n]["execution"]["theoretical_flop_fraction"]) < 1e-12
                                             for s in results for n in results[s] if n.startswith("sparse")),
        "frozen_backbone_unchanged": all(r["frozen_backbone_unchanged"] for s in results for r in results[s].values()),
        "all_values_finite": all(r["all_finite"] for s in results for r in results[s].values()),
    }
    if repeat is not None:
        checks["reproducible"] = bool(np.isclose(repeat["test_mse"], results[s0]["sparse_top2"]["test_mse"], rtol=sc["reproducibility_rtol"], atol=0))
    checks["all_passed"] = all(checks.values())

    methods = list(cfg["methods"])
    def agg(n, key="test_mse"):
        v = np.array([results[s][n][key] for s in results])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()), "max": float(v.max())}
    summary = {"aggregate_test_mse": {n: agg(n) for n in methods}, "aggregate_val_mse": {n: agg(n, "val_mse") for n in methods},
               "aggregate_train_mse": {n: agg(n, "train_mse") for n in methods},
               "ratios": {"top2_vs_dense": r_t2_dense, "top2_vs_matched_lora": r_t2_matched, "top4_vs_dense": r_t4_dense},
               "trainable_parameters": {n: results[s0][n]["trainable_parameters"] for n in methods}, "criteria": checks}
    env = environment_info()
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"config": cfg, "environment": env, "results": {str(s): v for s, v in results.items()}, "repeat_run": repeat, "summary": summary}, f, indent=1)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if not no_plot:
        plot_curves(results, methods, os.path.join(run_dir, "loss_curves.png"))
        plot_accuracy_vs_k(results, os.path.join(run_dir, "accuracy_vs_k.png"))
        plot_routing(results, os.path.join(run_dir, "routing_selection_heatmap.png"))
    text = write_summary(os.path.join(run_dir, "summary.md"), cfg, results, repeat, summary, env, methods)
    print(text)
    return checks["all_passed"]


def sweep(cfg, run_dir, kind):
    """Scaling ablations on the same task: expert count (E=8, k in 1,2,4,8) or rank (top-2, r in 4,8,16)."""
    lrs = cfg["training"]["learning_rate"]
    if kind == "experts":
        variants = {f"sparse_E8_top{k}": dict(n_experts=8, top_k=k, router_grad_mode="straight_through" if k == 1 else "renormalized",
                                             lr=lrs["sparse_top1"] if k == 1 else lrs["sparse_top2"]) for k in (1, 2, 4, 8)}
    else:
        variants = {f"sparse_top2_r{r}": dict(rank=r, top_k=2, router_grad_mode="renormalized", lr=lrs["sparse_top2"]) for r in (4, 8, 16)}
    results = {}
    for seed in cfg["experiment"]["seeds"]:
        problem = make_problem(cfg, seed)
        results[seed] = {}
        for name, spec in variants.items():
            spec = dict(spec)
            lr = spec.pop("lr")
            r = train_one("sparse_" + name, spec, problem, cfg, seed, lr, eval_split="test")
            results[seed][name] = r
            print(f"[sweep {kind}] seed {seed} {name:16s} test {r['test_mse']:.4e} params {r['trainable_parameters']:5d} "
                  f"entropy {r['routing_test']['overall_entropy_nats']:.3f} util {np.round(r['routing_test']['overall_utilization'], 2).tolist()} {r['elapsed_seconds']:.1f}s")
    agg = {n: {"mean_test_mse": float(np.mean([results[s][n]["test_mse"] for s in results])),
               "std_test_mse": float(np.std([results[s][n]["test_mse"] for s in results], ddof=1)),
               "trainable_parameters": results[cfg["experiment"]["seeds"][0]][n]["trainable_parameters"],
               "mean_elapsed_seconds": float(np.mean([results[s][n]["elapsed_seconds"] for s in results]))} for n in variants}
    with open(os.path.join(run_dir, f"sweep_{kind}.json"), "w") as f:
        json.dump({"variants": variants, "results": {str(s): v for s, v in results.items()}, "aggregate": agg, "environment": environment_info()}, f, indent=1)
    print(json.dumps(agg, indent=1))


# --------------------------------------------------------------------------- #
def plot_curves(results, methods, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = sorted(results)
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.4 * len(seeds), 3.6), sharey=True, squeeze=False, facecolor=PAL["surface"])
    for ax, seed in zip(axes[0], seeds):
        _style(ax)
        for n in methods:
            if n == "frozen":
                ax.axhline(results[seed][n]["test_mse"], color=COLOR[n], ls="--", lw=1.2, label=LABEL[n])
                continue
            s, v = zip(*results[seed][n]["history"]["eval_mse"])
            ax.plot(s, v, color=COLOR[n], lw=1.8, label=LABEL[n], ls=":" if n == "sparse_top4" else "-")
        ax.set_yscale("log")
        ax.set_title(f"seed {seed}", fontsize=10, loc="left", color=PAL["text"])
        ax.set_xlabel("optimizer step", color=PAL["muted"])
    axes[0][0].set_ylabel("test MSE (log)", color=PAL["muted"])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Sparse top-k vs dense MoLoRA on the 4-regime task", y=1.14, fontsize=11, color=PAL["text"])
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)


def plot_accuracy_vs_k(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = sorted(results)
    ks = {"sparse_top1": 1, "sparse_top2": 2, "sparse_top4": 4}
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor=PAL["surface"])
    _style(ax)
    for n, k in ks.items():
        vals = [results[s][n]["test_mse"] for s in seeds]
        ax.scatter([k] * len(vals), vals, color=COLOR[n], s=28, zorder=3, label=LABEL[n])
    ax.plot(list(ks.values()), [np.mean([results[s][n]["test_mse"] for s in seeds]) for n in ks], color=PAL["muted"], lw=1.2, zorder=2)
    for n in ("dense_molora", "matched_lora", "single_lora"):
        ax.axhline(np.mean([results[s][n]["test_mse"] for s in seeds]), color=COLOR[n], ls="--", lw=1.2, label=LABEL[n] + " (mean)")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("active experts k (of 4)", color=PAL["muted"])
    ax.set_ylabel("held-out test MSE (log)", color=PAL["muted"])
    ax.legend(frameon=False, fontsize=7, loc="center right")
    ax.set_title("Test MSE vs k (points = seeds)", fontsize=10, loc="left", color=PAL["text"])
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)


def plot_routing(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    seeds = sorted(results)
    names = ["sparse_top1", "sparse_top2", "sparse_top4"]
    fig, axes = plt.subplots(len(names), len(seeds), figsize=(3.9 * len(seeds), 3.1 * len(names)), squeeze=False, facecolor=PAL["surface"])
    fig.subplots_adjust(wspace=0.55, hspace=0.5)
    for row, n in enumerate(names):
        for col, seed in enumerate(seeds):
            ax = axes[row][col]
            mat = np.array(results[seed][n]["routing_test"]["regime_expert_selection_rate"])
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#f0efec" if mat[i, j] > 0.55 else PAL["text"])
            ax.set_xticks(range(mat.shape[1]), [f"e{j + 1}" for j in range(mat.shape[1])], fontsize=8)
            ax.set_yticks(range(mat.shape[0]), [f"regime {i + 1}" for i in range(mat.shape[0])], fontsize=8)
            ax.set_title(f"{LABEL[n]}, seed {seed}", fontsize=9, loc="left", color=PAL["text"])
            ax.tick_params(colors=PAL["muted"], length=0)
            for s in ax.spines.values():
                s.set_visible(False)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cb.set_label("selection rate (fraction of regime tokens routed to expert)", color=PAL["muted"], fontsize=8)
    cb.ax.tick_params(labelsize=8, colors=PAL["muted"])
    fig.suptitle("Sparse routing: regime x expert selection rate (test split)", fontsize=11, color=PAL["text"])
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)


def write_summary(path, cfg, results, repeat, summary, env, methods):
    L = [f"# {cfg['experiment']['name']} — run summary", "", f"timestamp: {env['timestamp']}  ", f"git: {env['git_branch']} @ {env['git_commit'][:12]}", "",
         "## Held-out test MSE per seed", "", "| seed | " + " | ".join(LABEL[n] for n in methods) + " |", "|---|" + "---|" * len(methods)]
    for s in sorted(results):
        L.append(f"| {s} | " + " | ".join(f"{results[s][n]['test_mse']:.4e}" for n in methods) + " |")
    L += ["", "## Aggregate (mean ± std over seeds)", "", "| method | trainable params | train MSE | val MSE | test MSE | std | mean time (s) |", "|---|---|---|---|---|---|---|"]
    for n in methods:
        a, v, t = summary["aggregate_test_mse"][n], summary["aggregate_val_mse"][n], summary["aggregate_train_mse"][n]
        L.append(f"| {LABEL[n]} | {summary['trainable_parameters'][n]} | {t['mean']:.4e} | {v['mean']:.4e} | {a['mean']:.4e} | {a['std']:.1e} | "
                 f"{np.mean([results[s][n]['elapsed_seconds'] for s in results]):.1f} |")
    L += ["", "## Per-regime test MSE (seed-averaged)", "", "| method | " + " | ".join(f"regime {r + 1}" for r in range(cfg['data']['n_regimes'])) + " |", "|---|" + "---|" * cfg["data"]["n_regimes"]]
    for n in methods:
        per = np.mean([results[s][n]["test_mse_by_regime"] for s in results], axis=0)
        L.append(f"| {LABEL[n]} | " + " | ".join(f"{v:.3e}" for v in per) + " |")
    L += ["", "## Sparse execution accounting (summed over all training forwards)", "",
          "| method | seed | adapter FLOP fraction vs dense | theoretical k/E | expert matmuls executed / dense | mean experts skipped per forward |", "|---|---|---|---|---|---|"]
    for s in sorted(results):
        for n in methods:
            ex = results[s][n].get("execution")
            if ex:
                L.append(f"| {LABEL[n]} | {s} | {ex['expert_flop_fraction_vs_dense']:.4f} | {ex['theoretical_flop_fraction']:.2f} | "
                         f"{ex['expert_matmuls_executed']} / {ex['expert_matmuls_dense']} | {ex['mean_experts_skipped_per_forward']:.3f} |")
    L += ["", "## Routing on the test split", ""]
    for n in methods:
        if n not in ("dense_molora", "sparse_top1", "sparse_top2", "sparse_top4"):
            continue
        for s in sorted(results):
            rt = results[s][n]["routing_test"]
            L.append(f"{LABEL[n]} seed {s}: dense-prob entropy per regime {['%.3f' % e for e in rt['regime_entropy_nats']]} (uniform {rt['max_entropy_nats']:.3f}); "
                     f"utilization (fraction of tokens selecting each expert) {['%.2f' % u for u in rt['overall_utilization']]}; "
                     f"mean pairwise expert distance {results[s][n]['expert_divergence']['pairwise_relative_distance_mean']:.3f}")
            L.append("")
            L.append("| selection rate | " + " | ".join(f"expert {j + 1}" for j in range(len(rt["overall_utilization"]))) + " |")
            L.append("|---|" + "---|" * len(rt["overall_utilization"]))
            for r, row in enumerate(rt["regime_expert_selection_rate"]):
                L.append(f"| regime {r + 1} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
            L.append("")
    if repeat is not None:
        s0 = cfg["experiment"]["seeds"][0]
        L += [f"repeat seed {s0} sparse_top2 test MSE: {repeat['test_mse']:.6e} (first run {results[s0]['sparse_top2']['test_mse']:.6e})", ""]
    L += ["## Pre-registered criteria", ""] + [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in summary["criteria"].items()]
    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "sparse_molora.yaml"))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sweep", choices=["experts", "rank"])
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    if args.seeds:
        cfg["experiment"]["seeds"] = args.seeds
    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps
    tag = args.tag or ("pilot" if args.pilot else f"sweep_{args.sweep}" if args.sweep else "")
    run_dir = run_dir_for(cfg, tag)
    print(f"[molora] writing to {run_dir}")
    if args.pilot:
        pilot(cfg, run_dir)
        return 0
    if args.sweep:
        for n in ("sparse_top1", "sparse_top2"):
            if cfg["training"]["learning_rate"].get(n) is None:
                raise SystemExit("run --pilot first and fill training.learning_rate for the sparse methods")
        sweep(cfg, run_dir, args.sweep)
        return 0
    if any(cfg["training"]["learning_rate"].get(n) is None for n in cfg["methods"] if n.startswith("sparse")):
        raise SystemExit("sparse learning rates are unset: run --pilot and record them in the config first")
    return 0 if confirmatory(cfg, run_dir, no_plot=args.no_plot) else 1


if __name__ == "__main__":
    raise SystemExit(main())
