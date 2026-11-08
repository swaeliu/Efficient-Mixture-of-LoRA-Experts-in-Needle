#!/usr/bin/env python
"""Phase 1 synthetic low-rank regression benchmark (Tasks 1.9 - 1.11).

Teacher:   y = x (W0 + dW*),  dW* = c A* B*  with rank(dW*) = target_rank <= lora_rank
Student:   LoRALinear sharing the SAME frozen W0; only A, B are trained.
Baseline:  the frozen backbone alone, y_hat = x W0.

Everything model-side (forward, backward, optimizer) runs in Needle. NumPy is
used only to generate data and compute reference statistics.

Usage::

    python experiments/train_synthetic.py --config configs/lora_baseline.yaml
    python experiments/train_synthetic.py --config ... --seeds 1 2 3 --max-steps 200 --lr 0.01

Outputs go to <output_dir>/<name>/<YYYYmmdd-HHMMSS>/ : config.yaml, results.json,
summary.md, loss_curve.png.
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
from molora import LoRALinear

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_problem(cfg: dict, seed: int):
    """Build W0 (via Needle's Linear init), the low-rank target, and train/val data."""
    m, d = cfg["model"], cfg["data"]
    dtype = np.dtype(cfg["dtype"])
    in_dim, out_dim = m["input_dim"], m["output_dim"]

    np.random.seed(seed)  # Needle's initialisers use numpy's global RNG
    base = nn.Linear(in_dim, out_dim, bias=m["bias"], dtype=cfg["dtype"])
    W0 = base.weight.numpy().astype(np.float64)

    rng = np.random.default_rng(seed)
    r_star = d["target_rank"]
    A_star = rng.standard_normal((in_dim, r_star))
    B_star = rng.standard_normal((r_star, out_dim))
    dW = A_star @ B_star
    dW *= d["delta_rel_norm"] * np.linalg.norm(W0) / np.linalg.norm(dW)  # scale target correction

    def sample(n):
        x = rng.standard_normal((n, in_dim))
        y = x @ (W0 + dW)
        if m["bias"]:
            y = y + base.bias.numpy().astype(np.float64)
        if d["noise_std"] > 0:
            y = y + d["noise_std"] * rng.standard_normal(y.shape)
        return x.astype(dtype), y.astype(dtype)

    x_tr, y_tr = sample(d["n_train"])
    x_va, y_va = sample(d["n_val"])
    return base, dW.astype(dtype), (x_tr, y_tr), (x_va, y_va), rng


def mse_needle(pred, target):
    n = int(np.prod(pred.shape))
    return ((pred - target) ** 2).sum() / n


def evaluate(layer, x_np, y_np, dtype):
    x = ndl.Tensor(x_np, dtype=dtype, requires_grad=False)
    y = ndl.Tensor(y_np, dtype=dtype, requires_grad=False)
    return float(mse_needle(layer(x), y).numpy())


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def run_seed(cfg: dict, seed: int) -> dict:
    t0 = time.perf_counter()
    m, tr, dtype = cfg["model"], cfg["training"], cfg["dtype"]
    base, dW_star, (x_tr, y_tr), (x_va, y_va), rng = make_problem(cfg, seed)

    # ---- frozen-backbone baseline (no adapter, nothing trained) ------------ #
    baseline_train = evaluate(base, x_tr, y_tr, dtype)
    baseline_val = evaluate(base, x_va, y_va, dtype)

    # ---- LoRA student sharing W0 ----------------------------------------- #
    layer = LoRALinear.from_linear(base, rank=m["lora_rank"], alpha=m["lora_alpha"])
    W0_before = layer.weight.numpy().copy()
    b_before = None if layer.bias is None else layer.bias.numpy().copy()

    opt_name = tr["optimizer"].lower()
    if opt_name == "adam":
        opt = ndl.optim.Adam(layer.adapter_parameters(), lr=tr["learning_rate"])
    elif opt_name == "sgd":
        opt = ndl.optim.SGD(layer.adapter_parameters(), lr=tr["learning_rate"])
    else:
        raise ValueError(f"unknown optimizer {tr['optimizer']!r}")

    n_tr, bs = len(x_tr), tr["batch_size"]
    train_hist, val_hist = [], [(0, evaluate(layer, x_va, y_va, dtype))]
    perm, ptr = rng.permutation(n_tr), 0
    for step in range(1, tr["max_steps"] + 1):
        if ptr + bs > n_tr:
            perm, ptr = rng.permutation(n_tr), 0
        idx = perm[ptr : ptr + bs]
        ptr += bs
        xb = ndl.Tensor(x_tr[idx], dtype=dtype, requires_grad=False)
        yb = ndl.Tensor(y_tr[idx], dtype=dtype, requires_grad=False)
        opt.reset_grad()
        loss = mse_needle(layer(xb), yb)
        loss.backward()
        opt.step()
        train_hist.append(float(loss.numpy()))
        if step % tr["eval_every"] == 0 or step == tr["max_steps"]:
            val_hist.append((step, evaluate(layer, x_va, y_va, dtype)))

    elapsed = time.perf_counter() - t0
    final_train = evaluate(layer, x_tr, y_tr, dtype)
    final_val = val_hist[-1][1]

    # how close is the learned correction to the true one?
    dW_learned = layer.delta_weight()
    rel_delta_err = float(np.linalg.norm(dW_learned - dW_star) / np.linalg.norm(dW_star))

    frozen_unchanged = bool(np.array_equal(W0_before, layer.weight.numpy())) and (
        b_before is None or bool(np.array_equal(b_before, layer.bias.numpy()))
    )
    all_finite = bool(np.all(np.isfinite(train_hist)) and np.all(np.isfinite([v for _, v in val_hist])))

    return {
        "seed": seed,
        "baseline_train_mse": baseline_train,
        "baseline_val_mse": baseline_val,
        "initial_train_loss": train_hist[0],
        "final_train_loss_minibatch": train_hist[-1],
        "final_train_mse_full": final_train,
        "final_val_mse": final_val,
        "val_mse_ratio_vs_baseline": final_val / baseline_val,
        "relative_delta_weight_error": rel_delta_err,
        "frozen_params_unchanged": frozen_unchanged,
        "all_values_finite": all_finite,
        "elapsed_seconds": elapsed,
        "steps": tr["max_steps"],
        "parameters": layer.parameter_summary(),
        "train_loss_history": train_hist,
        "val_loss_history": val_hist,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def environment_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        commit = branch = "unknown"
    try:
        import matplotlib

        mpl = matplotlib.__version__
    except Exception:  # pragma: no cover
        mpl = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "matplotlib": mpl,
        "molora": molora.__version__,
        "needle": molora.needle_info(),
        "git_branch": branch,
        "git_commit": commit,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }


def check_criteria(cfg: dict, runs: list[dict], repeat: dict | None) -> dict:
    sc = cfg["success_criteria"]
    checks = {
        "val_mse_ratio_below_threshold": all(r["val_mse_ratio_vs_baseline"] <= sc["max_val_mse_ratio"] for r in runs),
        "train_loss_decreased": (not sc["require_train_loss_decrease"])
        or all(r["final_train_mse_full"] < r["initial_train_loss"] for r in runs),
        "val_loss_improved_over_baseline": all(r["final_val_mse"] < r["baseline_val_mse"] for r in runs),
        "frozen_params_unchanged": all(r["frozen_params_unchanged"] for r in runs),
        "all_values_finite": all(r["all_values_finite"] for r in runs),
    }
    if repeat is not None:
        first = runs[0]
        checks["reproducible"] = bool(
            np.isclose(first["final_val_mse"], repeat["final_val_mse"], rtol=sc["reproducibility_rtol"], atol=0)
        )
    checks["all_passed"] = all(checks.values())
    return checks


PALETTE = {"train": "#2a78d6", "val": "#eb6834", "baseline": "#52514e", "grid": "#e4e3df", "text": "#0b0b0b",
           "muted": "#52514e", "surface": "#fcfcfb"}


def plot_losses(runs: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), sharey=True, squeeze=False, facecolor=PALETTE["surface"])
    for ax, r in zip(axes[0], runs):
        ax.set_facecolor(PALETTE["surface"])
        steps = np.arange(1, len(r["train_loss_history"]) + 1)
        ax.plot(steps, r["train_loss_history"], color=PALETTE["train"], lw=1.2, alpha=0.9, label="LoRA train (minibatch)")
        vs, vl = zip(*r["val_loss_history"])
        ax.plot(vs, vl, color=PALETTE["val"], lw=2, marker="o", ms=3.5, label="LoRA validation")
        ax.axhline(r["baseline_val_mse"], color=PALETTE["baseline"], lw=1.2, ls="--", label="frozen backbone, validation")
        ax.set_yscale("log")
        ax.set_title(f"seed {r['seed']}", fontsize=10, color=PALETTE["text"], loc="left")
        ax.set_xlabel("optimizer step", color=PALETTE["muted"])
        ax.grid(True, which="major", color=PALETTE["grid"], lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(PALETTE["grid"])
        ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    axes[0][0].set_ylabel("mean squared error (log)", color=PALETTE["muted"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("LoRA learns a rank-8 correction on a frozen 128x128 backbone", y=1.10, fontsize=11, color=PALETTE["text"])
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)


def write_summary(path: str, cfg: dict, runs: list[dict], repeat: dict | None, checks: dict, env: dict) -> str:
    lines = [f"# {cfg['experiment']['name']} — run summary", "", f"timestamp: {env['timestamp']}  ",
             f"git: {env['git_branch']} @ {env['git_commit'][:12]}  ", f"needle: {env['needle']['needle_path']}", "",
             "| seed | baseline val MSE | LoRA val MSE | ratio | rel. ΔW error | init train loss | final train MSE | time (s) | W0 frozen |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(
            f"| {r['seed']} | {r['baseline_val_mse']:.4f} | {r['final_val_mse']:.3e} | {r['val_mse_ratio_vs_baseline']:.3e} | "
            f"{r['relative_delta_weight_error']:.3e} | {r['initial_train_loss']:.4f} | {r['final_train_mse_full']:.3e} | "
            f"{r['elapsed_seconds']:.2f} | {r['frozen_params_unchanged']} |"
        )
    if repeat is not None:
        lines += ["", f"repeat of seed {repeat['seed']}: final val MSE {repeat['final_val_mse']:.6e} "
                      f"(first run {runs[0]['final_val_mse']:.6e})"]
    p = runs[0]["parameters"]
    lines += ["", f"parameters: base (frozen) {p['base_parameters']}, adapter (trainable) {p['adapter_parameters']}, "
                  f"total {p['total_parameters']}, trainable ratio {p['trainable_ratio']:.4f}", "",
              "## success criteria", ""]
    lines += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()]
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "lora_baseline.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", help="override experiment.seeds")
    ap.add_argument("--max-steps", type=int, help="override training.max_steps")
    ap.add_argument("--lr", type=float, help="override training.learning_rate")
    ap.add_argument("--output-dir", help="override experiment.output_dir")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if args.seeds:
        cfg["experiment"]["seeds"] = args.seeds
    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir

    out_root = cfg["experiment"]["output_dir"]
    if not os.path.isabs(out_root):
        out_root = os.path.join(REPO_ROOT, out_root)
    run_dir = os.path.join(out_root, cfg["experiment"]["name"], dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    env = environment_info()
    print(f"[molora] writing to {run_dir}")
    print(f"[molora] needle at {env['needle']['needle_path']} (backend {env['needle']['array_api']})")

    runs = []
    for seed in cfg["experiment"]["seeds"]:
        r = run_seed(cfg, seed)
        runs.append(r)
        print(f"[seed {seed}] baseline val MSE {r['baseline_val_mse']:.4f} -> LoRA val MSE {r['final_val_mse']:.3e} "
              f"(ratio {r['val_mse_ratio_vs_baseline']:.3e}), rel dW err {r['relative_delta_weight_error']:.3e}, "
              f"{r['elapsed_seconds']:.2f}s, W0 frozen: {r['frozen_params_unchanged']}")

    repeat = None
    if cfg["experiment"].get("repeat_first_seed", False):
        repeat = run_seed(cfg, cfg["experiment"]["seeds"][0])
        print(f"[repeat seed {repeat['seed']}] final val MSE {repeat['final_val_mse']:.6e} vs {runs[0]['final_val_mse']:.6e}")

    checks = check_criteria(cfg, runs, repeat)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"config": cfg, "environment": env, "runs": runs, "repeat_run": repeat, "criteria": checks}, f, indent=1)
    if not args.no_plot:
        plot_losses(runs, os.path.join(run_dir, "loss_curve.png"))
    print(write_summary(os.path.join(run_dir, "summary.md"), cfg, runs, repeat, checks, env))
    return 0 if checks["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
