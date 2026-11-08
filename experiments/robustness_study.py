#!/usr/bin/env python
"""Phase 4A — 30-seed robustness study of sparse routing (frozen protocol in configs/phase4_robustness.yaml).

Subcommands::

    python experiments/robustness_study.py pilot-balance        # validation-only balance-coefficient pilot (seeds 2000-2004)
    python experiments/robustness_study.py study                # 30 seeds x 5 methods (resume-safe, appends runs.jsonl)
    python experiments/robustness_study.py ablation             # balance-loss variants on the same 30 seeds
    python experiments/robustness_study.py analyze RUN_DIR      # statistics, failure diagnostics, plots, summary.md

Randomness per study seed s: data + frozen W0 from make_problem(cfg, s) (Phase 2/3
convention), expert/router initialisation from derive_seed(s, "init") (identical
for every MoE method -> paired), minibatch order from derive_seed(s, "shuffle").
Each (seed, method) result is written as one JSON line as soon as it finishes.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import yaml

import needle as ndl
import needle.nn as nn
from molora import DenseMoLoRALinear, LoRALinear, SparseMoLoRALinear
from molora.protocol import binomial_ci, derive_seed, is_failure, paired_bootstrap_ci, summary_stats
from molora.sparse_molora import load_balance_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_dense_molora import PAL, _style, environment_info, make_problem, mse_by_regime, mse_needle, predict  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = {"single_lora": "LoRA r=8", "matched_lora": "LoRA r=34", "dense_molora": "dense 4x8", "sparse_top2": "sparse top-2",
         "sparse_top1_ste": "sparse top-1 (STE)", "sparse_top2_balance": "sparse top-2 + balance",
         "sparse_top1_ste_balance": "sparse top-1 (STE) + balance"}
COLOR = {"single_lora": "#2a78d6", "matched_lora": "#eb6834", "dense_molora": "#1baf7a", "sparse_top2": "#e87ba4",
         "sparse_top1_ste": "#eda100", "sparse_top2_balance": "#6250d6", "sparse_top1_ste_balance": "#e34948"}


# --------------------------------------------------------------------------- #
def build_model(name, spec, problem, cfg, seed):
    """Build a method's model on the shared frozen base. All MoE methods draw the SAME init."""
    m = cfg["model"]
    base = problem["base"]
    np.random.seed(derive_seed(seed, "init"))
    if spec["kind"] == "lora":
        return LoRALinear.from_linear(base, rank=spec["rank"], alpha=m["alpha"])
    if spec["kind"] == "dense":
        return DenseMoLoRALinear.from_linear(base, n_experts=m["n_experts"], rank=m["rank"], alpha=m["alpha"],
                                             router_temperature=m["router_temperature"], router_init=m["router_init"],
                                             b_init_std=m["b_init_std"])
    if spec["kind"] == "sparse":
        return SparseMoLoRALinear.from_linear(base, n_experts=m["n_experts"], rank=m["rank"], alpha=m["alpha"],
                                              top_k=spec["top_k"], router_temperature=m["router_temperature"],
                                              router_init=m["router_init"], b_init_std=m["b_init_std"],
                                              execution_mode=m["execution_mode"], router_grad_mode=spec["router_grad_mode"])
    raise ValueError(spec)


def _entropy(p):
    return float((-(p * np.log(np.clip(p, 1e-300, None))).sum(-1)).mean())


def routing_summary(model, split, R, dtype):
    x = ndl.Tensor(split["x"], dtype=dtype, requires_grad=False)
    if isinstance(model, SparseMoLoRALinear):
        d = model.routing_diagnostics(x)
        p, mask = d["dense_probs"], d["mask"]
    else:
        p = model.router_probabilities(x).astype(np.float64)
        mask = np.ones_like(p)
    z = split["z"]
    E = p.shape[1]
    sel_rate = np.stack([mask[z == r].mean(0) for r in range(R)])
    mean_prob = np.stack([p[z == r].mean(0) for r in range(R)])
    return {
        "regime_expert_mean_prob": mean_prob.tolist(),
        "regime_expert_selection_rate": sel_rate.tolist(),
        "regime_argmax_expert": mean_prob.argmax(1).tolist(),  # dominant expert per regime
        "n_distinct_dominant_experts": int(len(set(mean_prob.argmax(1).tolist()))),
        "utilization": mask.mean(0).tolist(),
        "argmax_utilization": (np.bincount(p.argmax(-1), minlength=E) / len(p)).tolist(),
        "entropy_nats": _entropy(p),
        "regime_entropy_nats": [_entropy(p[z == r]) for r in range(R)],
        "max_entropy_nats": float(np.log(E)),
    }


def grad_norms(model):
    rn = [float(np.sqrt(sum(float((p.grad.numpy().astype(np.float64) ** 2).sum()) for p in model.router_parameters()
                            if p.grad is not None)))]
    en = [float(np.sqrt(sum(float((p.grad.numpy().astype(np.float64) ** 2).sum()) for p in (e.lora_A, e.lora_B)
                            if p.grad is not None))) for e in model.experts]
    return rn[0], en


def train_one(name, spec, problem, cfg, seed, lr, balance_coef=0.0, extra_eval=None):
    """Train one method on one problem. ``extra_eval(model, problem) -> dict`` is merged into the record."""
    tr, dtype, R = cfg["training"], cfg["dtype"], cfg["data"]["n_regimes"]
    model = build_model(name, spec, problem, cfg, seed)
    params = model.trainable_parameters()
    is_moe = isinstance(model, DenseMoLoRALinear)
    is_sparse = isinstance(model, SparseMoLoRALinear)
    use_balance = bool(spec.get("balance")) and is_sparse
    W0 = problem["base"].weight.numpy().copy()
    x_tr, y_tr = problem["train"]["x"], problem["train"]["y"]
    n, bs = len(x_tr), tr["batch_size"]
    rng = np.random.default_rng(derive_seed(seed, "shuffle"))
    val, test = problem["val"], problem["test"]

    rec = {"method": name, "seed": seed, "spec": spec, "learning_rate": lr, "balance_coef": balance_coef if use_balance else 0.0,
           "init_seed": derive_seed(seed, "init"), "shuffle_seed": derive_seed(seed, "shuffle"),
           "trainable_parameters": int(sum(int(np.prod(p.shape)) for p in params))}
    if is_moe:
        rec["initial_routing_val"] = routing_summary(model, val, R, dtype)
    hist = {"train_loss": [], "val_mse": [(0, mse_by_regime(predict(model, val["x"], dtype), val, R)[0])],
            "router_grad_norm": [], "expert_grad_norms": [], "routing_entropy": []}
    opt = ndl.optim.Adam(params, lr=lr)
    perm, ptr = rng.permutation(n), 0
    t0 = time.perf_counter()
    for step in range(1, tr["max_steps"] + 1):
        if ptr + bs > n:
            perm, ptr = rng.permutation(n), 0
        idx = perm[ptr: ptr + bs]
        ptr += bs
        xb = ndl.Tensor(x_tr[idx], dtype=dtype, requires_grad=False)
        yb = ndl.Tensor(y_tr[idx], dtype=dtype, requires_grad=False)
        opt.reset_grad()
        if use_balance:
            out, sel = model.forward_with_selection(xb)
            loss = mse_needle(out, yb) + load_balance_loss(sel) * balance_coef
        else:
            loss = mse_needle(model(xb), yb)
        loss.backward()
        if is_moe and (step == 1 or step % tr["track_every"] == 0):
            r_n, e_n = grad_norms(model)
            hist["router_grad_norm"].append((step, r_n))
            hist["expert_grad_norms"].append((step, e_n))
        opt.step()
        if step % 10 == 0 or step == 1:
            hist["train_loss"].append((step, float(loss.numpy())))
        if step % tr["eval_every"] == 0:
            hist["val_mse"].append((step, mse_by_regime(predict(model, val["x"], dtype), val, R)[0]))
            if is_moe:
                hist["routing_entropy"].append((step, _entropy(model.router_probabilities(
                    ndl.Tensor(val["x"][:256], dtype=dtype, requires_grad=False)).astype(np.float64))))
    rec["elapsed_seconds"] = time.perf_counter() - t0
    rec["history"] = hist
    rec["frozen_backbone_unchanged"] = bool(np.array_equal(W0, problem["base"].weight.numpy()))
    rec["val_mse"], rec["val_mse_by_regime"] = mse_by_regime(predict(model, val["x"], dtype), val, R)
    rec["test_mse"], rec["test_mse_by_regime"] = mse_by_regime(predict(model, test["x"], dtype), test, R)
    rec["failed"] = is_failure(rec["val_mse"], cfg["failure"]["val_mse_threshold"])
    rec["all_finite"] = bool(np.isfinite(rec["val_mse"]) and np.isfinite(rec["test_mse"])
                             and all(np.isfinite(v) for _, v in hist["train_loss"]))
    if is_moe:
        rec["final_routing_val"] = routing_summary(model, val, R, dtype)
        rec["final_routing_test"] = routing_summary(model, test, R, dtype)
        D = model.expert_delta_weights()
        rec["expert_delta_norms"] = [float(np.linalg.norm(d)) for d in D]
    if extra_eval is not None:
        rec.update(extra_eval(model, problem))
    return rec


# --------------------------------------------------------------------------- #
def run_dir_for(cfg, tag):
    root = cfg["experiment"]["output_dir"]
    root = root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)
    d = os.path.join(root, cfg["experiment"]["name"] + (f"_{tag}" if tag else ""), dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def load_runs(path):
    runs = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
    return runs


def run_grid(cfg, methods, seeds, run_dir, balance_coef=0.0, label="study"):
    path = os.path.join(run_dir, "runs.jsonl")
    done = {(r["method"], r["seed"]) for r in load_runs(path)}
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    t_all = time.perf_counter()
    for seed in seeds:
        problem = None
        for name, spec in methods.items():
            if (name, seed) in done:
                continue
            if problem is None:
                problem = make_problem(cfg, seed)
            lr = cfg["training"]["learning_rate"][name.replace("_balance", "")]
            rec = train_one(name, spec, problem, cfg, seed, lr, balance_coef=balance_coef)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            flag = " FAIL" if rec["failed"] else ""
            print(f"[{label}] seed {seed} {name:24s} val {rec['val_mse']:.3e} test {rec['test_mse']:.3e} "
                  f"{rec['elapsed_seconds']:.1f}s W0 {rec['frozen_backbone_unchanged']}{flag}", flush=True)
    print(f"[{label}] done in {time.perf_counter() - t_all:.0f}s -> {path}")


def pilot_balance(cfg, run_dir):
    spec = cfg["balance_ablation"]["methods"]["sparse_top2_balance"]
    table = {}
    for coef in cfg["balance_ablation"]["pilot_coefficients"]:
        vals, utils = [], []
        for seed in cfg["balance_ablation"]["pilot_seeds"]:
            problem = make_problem(cfg, seed)
            rec = train_one("sparse_top2_balance", spec, problem, cfg, seed, cfg["training"]["learning_rate"]["sparse_top2"], balance_coef=coef)
            vals.append(rec["val_mse"])
            utils.append(rec["final_routing_val"]["utilization"])
            print(f"[pilot-balance] coef {coef} seed {seed} val {rec['val_mse']:.3e} util {np.round(rec['final_routing_val']['utilization'], 2).tolist()}", flush=True)
        table[str(coef)] = {"val_mse": vals, "mean_val_mse": float(np.mean(vals)), "median_val_mse": float(np.median(vals)),
                            "failures": int(sum(is_failure(v, cfg["failure"]["val_mse_threshold"]) for v in vals)),
                            "mean_utilization": np.mean(utils, axis=0).tolist()}
    # also the unbalanced reference on the same pilot seeds
    vals = []
    for seed in cfg["balance_ablation"]["pilot_seeds"]:
        rec = train_one("sparse_top2", cfg["methods"]["sparse_top2"], make_problem(cfg, seed), cfg, seed, cfg["training"]["learning_rate"]["sparse_top2"])
        vals.append(rec["val_mse"])
    table["0 (no balance)"] = {"val_mse": vals, "mean_val_mse": float(np.mean(vals)), "median_val_mse": float(np.median(vals)),
                               "failures": int(sum(is_failure(v, cfg["failure"]["val_mse_threshold"]) for v in vals))}
    best = min((c for c in table if c != "0 (no balance)"), key=lambda c: (table[c]["failures"], table[c]["median_val_mse"]))
    out = {"table": table, "chosen_coefficient": float(best), "rule": "fewest validation failures, then lowest median validation MSE, among non-zero coefficients"}
    with open(os.path.join(run_dir, "pilot_balance.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "val_mse"} for k, v in table.items()}, indent=1))
    print("[pilot-balance] chosen coefficient:", best)


# --------------------------------------------------------------------------- #
def analyze(cfg, run_dir, ablation_dir=None):
    runs = load_runs(os.path.join(run_dir, "runs.jsonl"))
    if ablation_dir:
        runs += load_runs(os.path.join(ablation_dir, "runs.jsonl"))
    by = {}
    for r in runs:
        by.setdefault(r["method"], {})[r["seed"]] = r
    seeds = sorted(set.intersection(*[set(d) for d in by.values()]))
    thr = cfg["failure"]["val_mse_threshold"]
    an = cfg["analysis"]
    stats = {}
    for m, d in by.items():
        test = [d[s]["test_mse"] for s in seeds]
        val = [d[s]["val_mse"] for s in seeds]
        fails = int(sum(d[s]["failed"] for s in seeds))
        lo, hi = binomial_ci(fails, len(seeds), an["alpha"])
        st = {"test": summary_stats(test), "val": summary_stats(val), "failures": fails, "n": len(seeds),
              "failure_rate": fails / len(seeds), "failure_ci95": [lo, hi],
              "failed_seeds": [s for s in seeds if d[s]["failed"]],
              "frozen_backbone_unchanged_all": all(d[s]["frozen_backbone_unchanged"] for s in seeds),
              "all_finite": all(d[s]["all_finite"] for s in seeds),
              "mean_elapsed_s": float(np.mean([d[s]["elapsed_seconds"] for s in seeds]))}
        if m != "dense_molora" and "dense_molora" in by:
            dense = [by["dense_molora"][s]["test_mse"] for s in seeds]
            st["vs_dense"] = {
                "per_seed_diff": [t - u for t, u in zip(test, dense)],
                "mean_diff_bootstrap": paired_bootstrap_ci(test, dense, an["n_bootstrap"], an["bootstrap_seed"], an["alpha"]),
                "median_diff_bootstrap": paired_bootstrap_ci(test, dense, an["n_bootstrap"], an["bootstrap_seed"], an["alpha"], statistic=np.median),
                "log10_ratio_mean_bootstrap": paired_bootstrap_ci(np.log10(test), np.log10(dense), an["n_bootstrap"], an["bootstrap_seed"], an["alpha"]),
                "fraction_better_than_dense": float(np.mean([t < u for t, u in zip(test, dense)])),
            }
        if "final_routing_test" in d[seeds[0]]:
            st["routing"] = {
                "entropy_mean": float(np.mean([d[s]["final_routing_test"]["entropy_nats"] for s in seeds])),
                "distinct_dominant_experts_hist": {str(k): int(v) for k, v in zip(*np.unique([d[s]["final_routing_test"]["n_distinct_dominant_experts"] for s in seeds], return_counts=True))},
                "utilization_min_mean": float(np.mean([min(d[s]["final_routing_test"]["utilization"]) for s in seeds])),
                "utilization_max_mean": float(np.mean([max(d[s]["final_routing_test"]["utilization"]) for s in seeds])),
            }
        stats[m] = st

    # ---- failure diagnostics -------------------------------------------- #
    diag = []
    for m, d in by.items():
        for s in seeds:
            r = d[s]
            if not r["failed"] or "final_routing_test" not in r:
                continue
            fr, ir = r["final_routing_test"], r["initial_routing_val"]
            diag.append({
                "method": m, "seed": s, "init_seed": r["init_seed"], "val_mse": r["val_mse"], "test_mse": r["test_mse"],
                "test_mse_by_regime": r["test_mse_by_regime"],
                "initial_regime_argmax_expert": ir["regime_argmax_expert"], "initial_regime_expert_mean_prob": ir["regime_expert_mean_prob"],
                "final_regime_argmax_expert": fr["regime_argmax_expert"], "final_regime_expert_mean_prob": fr["regime_expert_mean_prob"],
                "final_regime_expert_selection_rate": fr["regime_expert_selection_rate"],
                "final_utilization": fr["utilization"], "n_distinct_dominant_experts": fr["n_distinct_dominant_experts"],
                "final_entropy": fr["entropy_nats"], "expert_delta_norms": r["expert_delta_norms"],
                "val_mse_trajectory": r["history"]["val_mse"], "router_grad_norm": r["history"]["router_grad_norm"],
                "expert_grad_norms": r["history"]["expert_grad_norms"], "train_loss_tail": r["history"]["train_loss"][-20:],
            })
    out = {"seeds": seeds, "threshold": thr, "stats": stats, "failure_diagnostics": diag, "environment": environment_info(),
           "run_dir": run_dir, "ablation_dir": ablation_dir}
    with open(os.path.join(run_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    plots(by, seeds, thr, run_dir)
    text = write_summary(cfg, by, seeds, stats, diag, run_dir)
    print(text)
    return out


def plots(by, seeds, thr, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [m for m in LABEL if m in by]
    # 1. distribution
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=PAL["surface"])
    _style(ax)
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        v = np.array([by[m][s]["test_mse"] for s in seeds])
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=COLOR[m], alpha=0.8, zorder=3)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=PAL["text"], lw=1.4, zorder=4)
    ax.axhline(thr, color=PAL["muted"], ls="--", lw=1, label=f"failure threshold (val) {thr}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(methods)), [LABEL[m] for m in methods], fontsize=8, rotation=15)
    ax.set_ylabel("held-out test MSE (log)", color=PAL["muted"])
    ax.set_title(f"Test MSE over {len(seeds)} seeds (bar = median)", fontsize=10, loc="left", color=PAL["text"])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "test_mse_distribution.png"), dpi=130, bbox_inches="tight", facecolor=PAL["surface"]); plt.close(fig)
    # 2. paired scatter
    if "dense_molora" in by:
        fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor=PAL["surface"])
        _style(ax)
        d = np.array([by["dense_molora"][s]["test_mse"] for s in seeds])
        for m in ("sparse_top2", "sparse_top1_ste", "sparse_top2_balance", "sparse_top1_ste_balance"):
            if m in by:
                v = np.array([by[m][s]["test_mse"] for s in seeds])
                ax.scatter(d, v, s=18, color=COLOR[m], label=LABEL[m], alpha=0.85, zorder=3)
        lo, hi = 1e-14, 1
        ax.plot([lo, hi], [lo, hi], color=PAL["muted"], lw=1, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("dense MoLoRA test MSE", color=PAL["muted"]); ax.set_ylabel("sparse test MSE (same seed)", color=PAL["muted"])
        ax.set_title("Paired by seed (below diagonal = sparse better)", fontsize=9, loc="left", color=PAL["text"])
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(run_dir, "paired_sparse_vs_dense.png"), dpi=130, bbox_inches="tight", facecolor=PAL["surface"]); plt.close(fig)
    # 3. validation-MSE histogram with threshold (failure definition)
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=PAL["surface"])
    _style(ax)
    bins = np.logspace(-14, 0, 57)
    for m in methods:
        v = np.clip([by[m][s]["val_mse"] for s in seeds], 1e-14, None)
        ax.hist(v, bins=bins, histtype="step", lw=1.6, color=COLOR[m], label=LABEL[m])
    ax.axvline(thr, color=PAL["muted"], ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("validation MSE after 2000 steps (log)", color=PAL["muted"]); ax.set_ylabel("runs", color=PAL["muted"])
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("Convergence-failure histogram (dashed = pre-registered threshold)", fontsize=9, loc="left", color=PAL["text"])
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "failure_histogram.png"), dpi=130, bbox_inches="tight", facecolor=PAL["surface"]); plt.close(fig)
    # 4. entropy vs val loss + 5. utilisation summary
    moe = [m for m in methods if "final_routing_test" in by[m][seeds[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=PAL["surface"])
    _style(axes[0]); _style(axes[1])
    for m in moe:
        e = [by[m][s]["final_routing_test"]["entropy_nats"] for s in seeds]
        v = [by[m][s]["val_mse"] for s in seeds]
        axes[0].scatter(e, v, s=16, color=COLOR[m], label=LABEL[m], alpha=0.85)
    axes[0].set_yscale("log"); axes[0].axhline(thr, color=PAL["muted"], ls="--", lw=1)
    axes[0].set_xlabel("router entropy on test (nats; uniform = 1.386)", color=PAL["muted"]); axes[0].set_ylabel("validation MSE (log)", color=PAL["muted"])
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Router entropy vs validation loss", fontsize=9, loc="left", color=PAL["text"])
    for i, m in enumerate(moe):
        u = np.sort(np.array([by[m][s]["final_routing_test"]["utilization"] for s in seeds]), axis=1)  # per-seed sorted utilisation
        for j in range(u.shape[1]):
            axes[1].scatter(np.full(len(seeds), i) + (j - 1.5) * 0.12, u[:, j], s=10, color=COLOR[m], alpha=0.6)
    axes[1].set_xticks(range(len(moe)), [LABEL[m] for m in moe], fontsize=7, rotation=15)
    axes[1].set_ylabel("per-expert utilisation (sorted within seed)", color=PAL["muted"], fontsize=8)
    axes[1].set_title("Expert utilisation per seed (fraction of tokens selecting expert)", fontsize=9, loc="left", color=PAL["text"])
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "entropy_and_utilization.png"), dpi=130, bbox_inches="tight", facecolor=PAL["surface"]); plt.close(fig)


def write_summary(cfg, by, seeds, stats, diag, run_dir):
    methods = [m for m in LABEL if m in by]
    L = [f"# Phase 4A robustness study — {len(seeds)} seeds ({seeds[0]}–{seeds[-1]})", "",
         f"failure criterion (pre-registered): validation MSE > {cfg['failure']['val_mse_threshold']} after {cfg['training']['max_steps']} steps", "",
         "| method | trainable params | mean test MSE | median | std | IQR (q1–q3) | min | max | failures (val) | failure rate [95% CI] | W0 frozen | mean time (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        st = stats[m]; t = st["test"]
        L.append(f"| {LABEL[m]} | {by[m][seeds[0]]['trainable_parameters']} | {t['mean']:.3e} | {t['median']:.3e} | {t['std']:.2e} | {t['q1']:.2e}–{t['q3']:.2e} | {t['min']:.2e} | {t['max']:.2e} | "
                 f"{st['failures']}/{st['n']} | {st['failure_rate']:.3f} [{st['failure_ci95'][0]:.3f}, {st['failure_ci95'][1]:.3f}] | {st['frozen_backbone_unchanged_all']} | {st['mean_elapsed_s']:.1f} |")
    L += ["", "## Paired comparison with dense MoLoRA (same seed = same data, W0, init, minibatches)", "",
          "| method | mean Δ test MSE (method − dense) [95% paired bootstrap CI] | median Δ [CI] | mean log10 ratio [CI] | fraction of seeds better than dense |", "|---|---|---|---|---|"]
    for m in methods:
        if "vs_dense" not in stats[m]:
            continue
        v = stats[m]["vs_dense"]; a, b, c = v["mean_diff_bootstrap"], v["median_diff_bootstrap"], v["log10_ratio_mean_bootstrap"]
        L.append(f"| {LABEL[m]} | {a['estimate']:+.3e} [{a['ci_low']:+.3e}, {a['ci_high']:+.3e}] | {b['estimate']:+.3e} [{b['ci_low']:+.3e}, {b['ci_high']:+.3e}] | "
                 f"{c['estimate']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {v['fraction_better_than_dense']:.3f} |")
    L += ["", "## Routing (test split, final)", "", "| method | mean entropy (nats) | #distinct dominant experts (hist over seeds) | mean min / max utilisation |", "|---|---|---|---|"]
    for m in methods:
        if "routing" in stats[m]:
            r = stats[m]["routing"]
            L.append(f"| {LABEL[m]} | {r['entropy_mean']:.3f} | {r['distinct_dominant_experts_hist']} | {r['utilization_min_mean']:.2f} / {r['utilization_max_mean']:.2f} |")
    L += ["", f"## Failed runs ({len(diag)})", ""]
    if not diag:
        L.append("No run exceeded the validation threshold.")
    for d in diag:
        L.append(f"- **{LABEL[d['method']]} seed {d['seed']}** (init seed {d['init_seed']}): val {d['val_mse']:.3e}, test {d['test_mse']:.3e}; per-regime test "
                 f"{['%.2e' % v for v in d['test_mse_by_regime']]}; dominant expert per regime initial {d['initial_regime_argmax_expert']} → final {d['final_regime_argmax_expert']} "
                 f"({d['n_distinct_dominant_experts']} distinct); utilisation {['%.2f' % u for u in d['final_utilization']]}; entropy {d['final_entropy']:.3f}; "
                 f"expert norms {['%.2f' % v for v in d['expert_delta_norms']]}; val trajectory {[(s, '%.2e' % v) for s, v in d['val_mse_trajectory'][::5]]}")
    text = "\n".join(L) + "\n"
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot-balance", "study", "ablation", "analyze"])
    ap.add_argument("run_dir", nargs="?", help="for analyze: the study run directory (or for study/ablation: resume into this dir)")
    ap.add_argument("--ablation-dir", help="for analyze: include balance-ablation runs from this directory")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "phase4_robustness.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+")
    args = ap.parse_args(argv)
    cfg = copy.deepcopy(yaml.safe_load(open(args.config)))
    seeds = args.seeds or cfg["experiment"]["seeds"]
    if args.command == "pilot-balance":
        d = run_dir_for(cfg, "pilot_balance"); print("[molora] writing to", d); pilot_balance(cfg, d); return 0
    if args.command == "study":
        d = args.run_dir or run_dir_for(cfg, ""); print("[molora] writing to", d)
        run_grid(cfg, cfg["methods"], seeds, d, label="study"); return 0
    if args.command == "ablation":
        coef = cfg["balance_ablation"]["coefficient"]
        if coef is None:
            raise SystemExit("set balance_ablation.coefficient from the pilot first")
        d = args.run_dir or run_dir_for(cfg, "balance_ablation"); print("[molora] writing to", d)
        run_grid(cfg, cfg["balance_ablation"]["methods"], seeds, d, balance_coef=float(coef), label="ablation"); return 0
    if args.command == "analyze":
        analyze(cfg, args.run_dir, args.ablation_dir); return 0


if __name__ == "__main__":
    raise SystemExit(main())
