#!/usr/bin/env python
"""Phase 2 heterogeneous synthetic routing benchmark (Tasks 2.7 - 2.10).

Four input regimes z in {0..3}; the last `indicator_dims` input features hold a
one-hot regime indicator and the rest are N(0, 1). Targets are

    y = x (W0 + dW_z),   dW_z = c_z A_z* B_z*   (rank `regime_rank`, ||dW_z|| = 0.5 ||W0||)

Methods (all share W0, data splits, loss, batch size and step budget):
    frozen        A: frozen backbone, nothing trained
    single_lora   B: LoRALinear rank 8
    matched_lora  D: LoRALinear with rank chosen to match dense MoLoRA's trainable count
    dense_molora  C: DenseMoLoRALinear, 4 experts x rank 8, softmax router

Modes::

    python experiments/train_dense_molora.py --pilot          # LR sweep on VALIDATION data, seed 42
    python experiments/train_dense_molora.py                  # confirmatory run (seeds from config, TEST data)
    python experiments/train_dense_molora.py --b-init-std 1e-3 --tag binit   # ablation variant

All model computation runs in Needle; NumPy generates data and reference statistics.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import yaml

import molora
import needle as ndl
import needle.nn as nn
from molora import DenseMoLoRALinear, LoRALinear

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ("frozen", "single_lora", "matched_lora", "dense_molora")
TRAINED = ("single_lora", "matched_lora", "dense_molora")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_problem(cfg: dict, seed: int) -> dict:
    d = cfg["data"]
    dtype = np.dtype(cfg["dtype"])
    in_dim, out_dim, R, k = d["input_dim"], d["output_dim"], d["n_regimes"], d["indicator_dims"]
    assert k >= R, "indicator_dims must be >= n_regimes for a one-hot regime code"

    np.random.seed(seed)  # Needle initialisers draw from the global RNG
    base = nn.Linear(in_dim, out_dim, bias=d["bias"], dtype=cfg["dtype"])
    W0 = base.weight.numpy().astype(np.float64)

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(R):
        A = rng.standard_normal((in_dim, d["regime_rank"]))
        B = rng.standard_normal((d["regime_rank"], out_dim))
        dW = A @ B
        dW *= d["delta_rel_norm"] * np.linalg.norm(W0) / np.linalg.norm(dW)
        deltas.append(dW)

    def sample(n):
        assert n % R == 0
        z = rng.permutation(np.repeat(np.arange(R), n // R))
        x = rng.standard_normal((n, in_dim))
        x[:, in_dim - k :] = 0.0
        x[np.arange(n), in_dim - k + z] = 1.0  # one-hot regime indicator in the reserved features
        y = np.einsum("ni,nio->no", x, np.stack([W0 + deltas[zi] for zi in z]))
        if d["bias"]:
            y = y + base.bias.numpy().astype(np.float64)
        if d["noise_std"] > 0:
            y = y + d["noise_std"] * rng.standard_normal(y.shape)
        return {"x": x.astype(dtype), "y": y.astype(dtype), "z": z}

    return {
        "base": base,
        "deltas": [dw.astype(dtype) for dw in deltas],
        "train": sample(d["n_train"]),
        "val": sample(d["n_val"]),
        "test": sample(d["n_test"]),
        "rng": rng,
    }


def build_model(name: str, problem: dict, cfg: dict, seed: int, *, b_init_std=None, router_init=None):
    """Models share the frozen base; adapter init is seeded per (method, seed)."""
    np.random.seed(seed * 1000 + METHODS.index(name))
    m, b = cfg["model"], cfg["baselines"]
    base = problem["base"]
    if name == "frozen":
        return base
    if name == "single_lora":
        return LoRALinear.from_linear(base, rank=b["single_lora_rank"], alpha=m["alpha"])
    if name == "matched_lora":
        return LoRALinear.from_linear(base, rank=b["matched_lora_rank"], alpha=m["alpha"])
    if name == "dense_molora":
        return DenseMoLoRALinear.from_linear(
            base, n_experts=m["n_experts"], rank=m["rank"], alpha=m["alpha"],
            router_temperature=m["router_temperature"],
            router_init=router_init or m["router_init"],
            b_init_std=m["b_init_std"] if b_init_std is None else b_init_std,
        )
    raise ValueError(name)


def trainable(model):
    if isinstance(model, nn.Linear):
        return []
    return model.trainable_parameters()


# --------------------------------------------------------------------------- #
# Evaluation and diagnostics (NumPy on detached outputs)
# --------------------------------------------------------------------------- #
def predict(model, x_np, dtype):
    return model(ndl.Tensor(x_np, dtype=dtype, requires_grad=False)).numpy()


def mse_by_regime(pred, split, R):
    err = ((pred - split["y"]) ** 2).mean(axis=1)
    return float(err.mean()), [float(err[split["z"] == r].mean()) for r in range(R)]


def routing_diagnostics(model: DenseMoLoRALinear, split: dict, R: int, dtype: str) -> dict:
    p = model.router_probabilities(ndl.Tensor(split["x"], dtype=dtype, requires_grad=False)).astype(np.float64)
    ent = -(p * np.log(np.clip(p, 1e-300, None))).sum(-1)
    top = p.argmax(-1)
    E = p.shape[1]
    mat = np.stack([p[split["z"] == r].mean(0) for r in range(R)])  # regime x expert mean probability
    return {
        "regime_expert_mean_probability": mat.tolist(),
        "regime_entropy_nats": [float(ent[split["z"] == r].mean()) for r in range(R)],
        "overall_entropy_nats": float(ent.mean()),
        "max_entropy_nats": float(np.log(E)),
        "regime_argmax_utilization": [np.bincount(top[split["z"] == r], minlength=E).astype(float).__truediv__((split["z"] == r).sum()).tolist() for r in range(R)],
        "overall_argmax_utilization": (np.bincount(top, minlength=E) / len(top)).tolist(),
        "overall_mean_probability": p.mean(0).tolist(),
    }


def expert_divergence(model: DenseMoLoRALinear) -> dict:
    D = model.expert_delta_weights()
    norms = [float(np.linalg.norm(d)) for d in D]
    rel = []
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            denom = 0.5 * (norms[i] + norms[j])
            rel.append(float(np.linalg.norm(D[i] - D[j]) / denom) if denom > 0 else 0.0)
    return {"expert_delta_norms": norms, "pairwise_relative_distance_mean": float(np.mean(rel)) if rel else 0.0,
            "pairwise_relative_distance_min": float(np.min(rel)) if rel else 0.0}


def router_grad_norm(model: DenseMoLoRALinear) -> float:
    g = [p.grad.numpy() for p in model.router_parameters() if p.grad is not None]
    return float(np.sqrt(sum(float((a.astype(np.float64) ** 2).sum()) for a in g))) if g else 0.0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def mse_needle(pred, target):
    return ((pred - target) ** 2).sum() / int(np.prod(pred.shape))


def train(model, problem: dict, cfg: dict, lr: float, rng: np.random.Generator, *, eval_split="val") -> dict:
    tr, dg, dtype = cfg["training"], cfg["diagnostics"], cfg["dtype"]
    R = cfg["data"]["n_regimes"]
    params = trainable(model)
    is_moe = isinstance(model, DenseMoLoRALinear)
    x_tr, y_tr = problem["train"]["x"], problem["train"]["y"]
    ev = problem[eval_split]
    n, bs = len(x_tr), tr["batch_size"]
    track_x = problem["val"]["x"][: min(256, len(problem["val"]["x"]))]

    t0 = time.perf_counter()
    hist = {"train_loss": [], "eval_mse": [], "router_grad_norm": [], "routing_entropy": [], "expert_divergence": []}
    base_pred = predict(problem["base"], ev["x"], dtype)
    hist["initial_adapter_output_max_abs"] = float(np.abs(predict(model, ev["x"], dtype) - base_pred).max()) if params else 0.0
    hist["eval_mse"].append((0, mse_by_regime(predict(model, ev["x"], dtype), ev, R)[0]))
    if is_moe:
        p0 = model.router_probabilities(ndl.Tensor(track_x, dtype=dtype, requires_grad=False)).astype(np.float64)
        hist["routing_entropy"].append((0, float((-(p0 * np.log(np.clip(p0, 1e-300, None))).sum(-1)).mean())))
        hist["expert_divergence"].append((0, expert_divergence(model)["pairwise_relative_distance_mean"]))

    if params:
        opt = ndl.optim.Adam(params, lr=lr) if tr["optimizer"].lower() == "adam" else ndl.optim.SGD(params, lr=lr)
        perm, ptr = rng.permutation(n), 0
        for step in range(1, tr["max_steps"] + 1):
            if ptr + bs > n:
                perm, ptr = rng.permutation(n), 0
            idx = perm[ptr : ptr + bs]
            ptr += bs
            xb = ndl.Tensor(x_tr[idx], dtype=dtype, requires_grad=False)
            yb = ndl.Tensor(y_tr[idx], dtype=dtype, requires_grad=False)
            opt.reset_grad()
            loss = mse_needle(model(xb), yb)
            loss.backward()
            if is_moe and (step == 1 or step % dg["track_every"] == 0):
                hist["router_grad_norm"].append((step, router_grad_norm(model)))
            opt.step()
            hist["train_loss"].append(float(loss.numpy()))
            if step % tr["eval_every"] == 0 or step == tr["max_steps"]:
                hist["eval_mse"].append((step, mse_by_regime(predict(model, ev["x"], dtype), ev, R)[0]))
            if is_moe and step % dg["track_every"] == 0:
                p = model.router_probabilities(ndl.Tensor(track_x, dtype=dtype, requires_grad=False)).astype(np.float64)
                hist["routing_entropy"].append((step, float((-(p * np.log(np.clip(p, 1e-300, None))).sum(-1)).mean())))
                hist["expert_divergence"].append((step, expert_divergence(model)["pairwise_relative_distance_mean"]))
    hist["elapsed_seconds"] = time.perf_counter() - t0
    return hist


def run_method(name, problem, cfg, seed, lr, *, eval_split, b_init_std=None, router_init=None) -> dict:
    model = build_model(name, problem, cfg, seed, b_init_std=b_init_std, router_init=router_init)
    W0 = problem["base"].weight.numpy().copy()
    rng = np.random.default_rng(seed + 12345)  # minibatch order, identical across methods
    hist = train(model, problem, cfg, lr, rng, eval_split=eval_split)
    dtype, R = cfg["dtype"], cfg["data"]["n_regimes"]
    out = {"method": name, "seed": seed, "learning_rate": lr if name != "frozen" else None,
           "frozen_backbone_unchanged": bool(np.array_equal(W0, problem["base"].weight.numpy())),
           "trainable_parameters": int(sum(int(np.prod(p.shape)) for p in trainable(model))),
           "total_parameters": int(sum(int(np.prod(p.shape)) for p in model.parameters())),
           "history": hist, "elapsed_seconds": hist["elapsed_seconds"]}
    for split in ("val", "test"):
        if split == "test" and eval_split != "test":
            continue  # pilot never touches test data
        overall, per_regime = mse_by_regime(predict(model, problem[split]["x"], dtype), problem[split], R)
        out[f"{split}_mse"] = overall
        out[f"{split}_mse_by_regime"] = per_regime
    if isinstance(model, DenseMoLoRALinear):
        out["routing_val"] = routing_diagnostics(model, problem["val"], R, dtype)
        if eval_split == "test":
            out["routing_test"] = routing_diagnostics(model, problem["test"], R, dtype)
        out["expert_divergence"] = expert_divergence(model)
        out["parameter_summary"] = model.parameter_summary()
    elif isinstance(model, LoRALinear):
        out["parameter_summary"] = model.parameter_summary()
    out["all_finite"] = bool(np.all(np.isfinite(hist["train_loss"])) and np.all(np.isfinite([v for _, v in hist["eval_mse"]])))
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
PAL = {"frozen": "#52514e", "single_lora": "#2a78d6", "matched_lora": "#eb6834", "dense_molora": "#1baf7a",
       "grid": "#e4e3df", "text": "#0b0b0b", "muted": "#52514e", "surface": "#fcfcfb"}
LABEL = {"frozen": "A frozen backbone", "single_lora": "B single LoRA r=8", "matched_lora": "D matched LoRA r=34",
         "dense_molora": "C dense MoLoRA 4x r=8"}


def _style(ax):
    ax.set_facecolor(PAL["surface"])
    ax.grid(True, color=PAL["grid"], lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["grid"])
    ax.tick_params(colors=PAL["muted"], labelsize=8)


def plot_curves(results: dict, path: str, split_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted(results)
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 3.4), sharey=True, squeeze=False, facecolor=PAL["surface"])
    for ax, seed in zip(axes[0], seeds):
        _style(ax)
        for name in TRAINED:
            s, v = zip(*results[seed][name]["history"]["eval_mse"])
            ax.plot(s, v, color=PAL[name], lw=2, label=LABEL[name])
        ax.axhline(results[seed]["frozen"][f"{split_label}_mse"], color=PAL["frozen"], ls="--", lw=1.2, label=LABEL["frozen"])
        ax.set_yscale("log")
        ax.set_title(f"seed {seed}", fontsize=10, loc="left", color=PAL["text"])
        ax.set_xlabel("optimizer step", color=PAL["muted"])
    axes[0][0].set_ylabel(f"{split_label} MSE (log)", color=PAL["muted"])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Heterogeneous 4-regime task: dense MoLoRA vs single-adapter baselines", y=1.11, fontsize=11, color=PAL["text"])
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)


def plot_routing_heatmap(results: dict, path: str, key="routing_test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    seeds = sorted(results)
    fig, axes = plt.subplots(1, len(seeds), figsize=(3.9 * len(seeds), 3.4), squeeze=False, facecolor=PAL["surface"])
    fig.subplots_adjust(wspace=0.55)
    for ax, seed in zip(axes[0], seeds):
        mat = np.array(results[seed]["dense_molora"][key]["regime_expert_mean_probability"])
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="#f0efec" if mat[i, j] > 0.55 else PAL["text"])
        ax.set_xticks(range(mat.shape[1]), [f"expert {j + 1}" for j in range(mat.shape[1])], fontsize=8)
        ax.set_yticks(range(mat.shape[0]), [f"regime {i + 1}" for i in range(mat.shape[0])], fontsize=8)
        ax.set_title(f"seed {seed}", fontsize=10, loc="left", color=PAL["text"])
        ax.tick_params(colors=PAL["muted"], length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
    cb.set_label("mean routing probability", color=PAL["muted"], fontsize=8)
    cb.ax.tick_params(labelsize=8, colors=PAL["muted"])
    fig.suptitle("Router: mean probability per regime (rows) and expert (columns)", fontsize=11, color=PAL["text"])
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)


def plot_dynamics(results: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted(results)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4), facecolor=PAL["surface"])
    series = [("router_grad_norm", "router gradient L2 norm", True), ("routing_entropy", "routing entropy (nats)", False),
              ("expert_divergence", "mean pairwise relative expert distance", False)]
    shades = ["#2a78d6", "#eb6834", "#1baf7a"]
    for ax, (key, ylabel, logy) in zip(axes, series):
        _style(ax)
        for c, seed in zip(shades, seeds):
            pts = results[seed]["dense_molora"]["history"][key]
            if not pts:
                continue
            s, v = zip(*pts)
            ax.plot(s, v, color=c, lw=2, marker="o", ms=3, label=f"seed {seed}")
        if logy:
            ax.set_yscale("symlog", linthresh=1e-6)
        if key == "routing_entropy":
            ax.axhline(np.log(results[seeds[0]]["dense_molora"]["parameter_summary"]["n_experts"]), color=PAL["frozen"],
                       ls="--", lw=1, label="uniform (log E)")
        ax.set_xlabel("optimizer step", color=PAL["muted"])
        ax.set_ylabel(ylabel, color=PAL["muted"])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Dense MoLoRA routing dynamics (zero-B init)", fontsize=11, color=PAL["text"])
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- #
def environment_info() -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return "unknown"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "numpy": np.__version__, "molora": molora.__version__, "needle": molora.needle_info(),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"), "git_commit": git("rev-parse", "HEAD"),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds")}


def make_run_dir(cfg, tag):
    root = cfg["experiment"]["output_dir"]
    root = root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)
    name = cfg["experiment"]["name"] + (f"_{tag}" if tag else "")
    d = os.path.join(root, name, dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def pilot(cfg, run_dir):
    seed = cfg["pilot"]["seed"]
    problem = make_problem(cfg, seed)
    table = {}
    for name in TRAINED:
        table[name] = {}
        for lr in cfg["pilot"]["learning_rates"]:
            r = run_method(name, problem, cfg, seed, lr, eval_split="val")
            table[name][str(lr)] = {"val_mse": r["val_mse"], "elapsed_seconds": r["elapsed_seconds"], "all_finite": r["all_finite"]}
            print(f"[pilot] {name:13s} lr={lr:<7} val MSE {r['val_mse']:.4e}  ({r['elapsed_seconds']:.1f}s)")
    best = {name: min(d.items(), key=lambda kv: kv[1]["val_mse"])[0] for name, d in table.items()}
    out = {"seed": seed, "candidates": cfg["pilot"]["learning_rates"], "results": table, "best_lr_by_val_mse": best,
           "environment": environment_info()}
    with open(os.path.join(run_dir, "pilot_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[pilot] best lr by validation MSE:", best)
    return out


def confirmatory(cfg, run_dir, *, b_init_std=None, router_init=None, no_plot=False):
    sc, lrs = cfg["success_criteria"], cfg["training"]["learning_rate"]
    results = {}
    for seed in cfg["experiment"]["seeds"]:
        problem = make_problem(cfg, seed)
        results[seed] = {}
        for name in METHODS:
            r = run_method(name, problem, cfg, seed, lrs.get(name), eval_split="test", b_init_std=b_init_std, router_init=router_init)
            results[seed][name] = r
            print(f"[seed {seed}] {name:13s} test MSE {r['test_mse']:.4e}  val MSE {r['val_mse']:.4e}  "
                  f"params {r['trainable_parameters']:5d}  {r['elapsed_seconds']:.1f}s  W0 frozen {r['frozen_backbone_unchanged']}")
    repeat = None
    if cfg["experiment"].get("repeat_first_seed"):
        s0 = cfg["experiment"]["seeds"][0]
        problem = make_problem(cfg, s0)
        repeat = run_method("dense_molora", problem, cfg, s0, lrs["dense_molora"], eval_split="test", b_init_std=b_init_std, router_init=router_init)
        print(f"[repeat seed {s0}] dense_molora test MSE {repeat['test_mse']:.6e} vs {results[s0]['dense_molora']['test_mse']:.6e}")

    ratios_matched = {s: results[s]["dense_molora"]["test_mse"] / results[s]["matched_lora"]["test_mse"] for s in results}
    ratios_frozen = {s: results[s]["dense_molora"]["test_mse"] / results[s]["frozen"]["test_mse"] for s in results}
    checks = {
        "molora_beats_matched_lora_on_every_seed": all(v <= sc["max_test_mse_ratio_vs_matched_lora"] for v in ratios_matched.values()),
        "molora_reduces_frozen_mse_90pct_on_every_seed": all(v <= sc["max_test_mse_ratio_vs_frozen"] for v in ratios_frozen.values()),
        "frozen_backbone_unchanged": all(r["frozen_backbone_unchanged"] for s in results for r in results[s].values()),
        "all_values_finite": all(r["all_finite"] for s in results for r in results[s].values()),
    }
    if repeat is not None:
        checks["reproducible"] = bool(np.isclose(repeat["test_mse"], results[cfg["experiment"]["seeds"][0]]["dense_molora"]["test_mse"],
                                                 rtol=sc["reproducibility_rtol"], atol=0))
    checks["all_passed"] = all(checks.values())

    def agg(name, key="test_mse"):
        v = np.array([results[s][name][key] for s in results])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()), "max": float(v.max())}

    summary = {"aggregate_test_mse": {n: agg(n) for n in METHODS}, "aggregate_val_mse": {n: agg(n, "val_mse") for n in METHODS},
               "test_mse_ratio_molora_vs_matched_lora": {str(k): v for k, v in ratios_matched.items()},
               "test_mse_ratio_molora_vs_frozen": {str(k): v for k, v in ratios_frozen.items()},
               "trainable_parameters": {n: results[cfg["experiment"]["seeds"][0]][n]["trainable_parameters"] for n in METHODS},
               "criteria": checks}
    env = environment_info()
    payload = {"config": cfg, "variant": {"b_init_std": b_init_std, "router_init": router_init}, "environment": env,
               "results": {str(s): v for s, v in results.items()}, "repeat_run": repeat, "summary": summary}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=1)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    s0 = cfg["experiment"]["seeds"][0]
    np.savetxt(os.path.join(run_dir, f"routing_matrix_seed{s0}.csv"),
               np.array(results[s0]["dense_molora"]["routing_test"]["regime_expert_mean_probability"]), delimiter=",", fmt="%.6f",
               header="rows = regimes, columns = experts; mean routing probability on the test split")
    if not no_plot:
        plot_curves(results, os.path.join(run_dir, "loss_curves.png"), "test")
        plot_routing_heatmap(results, os.path.join(run_dir, "routing_heatmap.png"))
        plot_dynamics(results, os.path.join(run_dir, "routing_dynamics.png"))
    text = write_summary(os.path.join(run_dir, "summary.md"), cfg, results, repeat, summary, env)
    print(text)
    return checks["all_passed"]


def write_summary(path, cfg, results, repeat, summary, env):
    L = [f"# {cfg['experiment']['name']} — run summary", "", f"timestamp: {env['timestamp']}  ",
         f"git: {env['git_branch']} @ {env['git_commit'][:12]}", "",
         "## Test MSE per seed", "", "| seed | " + " | ".join(LABEL[n] for n in METHODS) + " | C/D ratio | C/A ratio |",
         "|---|" + "---|" * (len(METHODS) + 2)]
    for s in sorted(results):
        L.append(f"| {s} | " + " | ".join(f"{results[s][n]['test_mse']:.4e}" for n in METHODS)
                 + f" | {summary['test_mse_ratio_molora_vs_matched_lora'][str(s)]:.3e} | {summary['test_mse_ratio_molora_vs_frozen'][str(s)]:.3e} |")
    L += ["", "## Aggregate test MSE (mean ± std over seeds)", "", "| method | trainable params | mean | std | min | max | mean train time (s) |", "|---|---|---|---|---|---|---|"]
    for n in METHODS:
        a = summary["aggregate_test_mse"][n]
        t = np.mean([results[s][n]["elapsed_seconds"] for s in results])
        L.append(f"| {LABEL[n]} | {summary['trainable_parameters'][n]} | {a['mean']:.4e} | {a['std']:.1e} | {a['min']:.4e} | {a['max']:.4e} | {t:.1f} |")
    L += ["", "## Per-regime test MSE (seed-averaged)", "", "| method | " + " | ".join(f"regime {r + 1}" for r in range(cfg['data']['n_regimes'])) + " |",
          "|---|" + "---|" * cfg["data"]["n_regimes"]]
    for n in METHODS:
        per = np.mean([results[s][n]["test_mse_by_regime"] for s in results], axis=0)
        L.append(f"| {LABEL[n]} | " + " | ".join(f"{v:.3e}" for v in per) + " |")
    L += ["", "## Dense MoLoRA routing (test split)", ""]
    for s in sorted(results):
        rt = results[s]["dense_molora"]["routing_test"]
        dv = results[s]["dense_molora"]["expert_divergence"]
        L.append(f"seed {s}: entropy per regime {['%.3f' % e for e in rt['regime_entropy_nats']]} (uniform = {rt['max_entropy_nats']:.3f}); "
                 f"argmax utilization {['%.2f' % u for u in rt['overall_argmax_utilization']]}; "
                 f"mean pairwise expert distance {dv['pairwise_relative_distance_mean']:.3f}")
        L.append("")
        L.append("| | " + " | ".join(f"expert {j + 1}" for j in range(len(rt["overall_mean_probability"]))) + " |")
        L.append("|---|" + "---|" * len(rt["overall_mean_probability"]))
        for r, row in enumerate(rt["regime_expert_mean_probability"]):
            L.append(f"| regime {r + 1} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
        L.append("")
    if repeat is not None:
        s0 = cfg["experiment"]["seeds"][0]
        L += [f"repeat seed {s0} dense MoLoRA test MSE: {repeat['test_mse']:.6e} (first run {results[s0]['dense_molora']['test_mse']:.6e})", ""]
    L += ["## Pre-registered criteria", ""] + [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in summary["criteria"].items()]
    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "dense_molora.yaml"))
    ap.add_argument("--pilot", action="store_true", help="learning-rate sweep on validation data only")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--b-init-std", type=float, help="ablation: non-zero B init for dense MoLoRA experts")
    ap.add_argument("--router-init", choices=["xavier", "zeros"], help="ablation: router weight init")
    ap.add_argument("--tag", default="", help="suffix for the output directory name")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    if args.seeds:
        cfg["experiment"]["seeds"] = args.seeds
    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps
    tag = args.tag or ("pilot" if args.pilot else "")
    run_dir = make_run_dir(cfg, tag)
    print(f"[molora] writing to {run_dir}")
    if args.pilot:
        pilot(cfg, run_dir)
        return 0
    ok = confirmatory(cfg, run_dir, b_init_std=args.b_init_std, router_init=args.router_init, no_plot=args.no_plot)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
