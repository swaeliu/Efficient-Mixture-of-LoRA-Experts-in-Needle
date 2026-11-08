#!/usr/bin/env python
"""Phase 4C — pretrain a tiny Needle transformer, freeze it, fine-tune adapters, compare.

    python experiments/train_tiny_transformer.py pretrain           # shared backbone checkpoint (outputs/phase4c_transformer/backbone/)
    python experiments/train_tiny_transformer.py pilot              # validation-only LR pilot for every adapter method
    python experiments/train_tiny_transformer.py study              # confirmatory fine-tuning over seeds (resume-safe)
    python experiments/train_tiny_transformer.py analyze RUN_DIR
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> models/
import needle as ndl
from models import (TinyTransformer, adapter_trainable_parameters, count_parameters, freeze_backbone, inject_adapters,
                    load_state, named_parameters, routing_stats, save_state, sparse_execution_stats)
from molora.protocol import derive_seed, paired_bootstrap_ci, summary_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sequence_task as st  # noqa: E402
from train_dense_molora import PAL, _style, environment_info  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = {"frozen": "frozen backbone", "single_lora": "LoRA r=4", "matched_lora": "LoRA r=18", "dense_molora": "dense MoLoRA 4x4",
         "sparse_top2": "sparse top-2", "sparse_top1_ste": "sparse top-1 (STE)"}
COLOR = {"frozen": "#52514e", "single_lora": "#2a78d6", "matched_lora": "#eb6834", "dense_molora": "#1baf7a",
         "sparse_top2": "#e87ba4", "sparse_top1_ste": "#eda100"}


def backbone_dir(cfg):
    return os.path.join(REPO_ROOT, cfg["experiment"]["output_dir"], cfg["experiment"]["name"], "backbone")


def build_backbone(cfg):
    m = cfg["model"]
    return TinyTransformer(cfg["data"]["vocab_size"], m["d_model"], m["n_heads"], m["n_layers"], m["d_ff"], m["max_seq_len"])


def evaluate(model, split, batch=256):
    """Mean cross-entropy (nats/token) and accuracy over a split, in minibatches."""
    tot_loss, tot_correct, n_tok = 0.0, 0, 0
    for i in range(0, len(split["x"]), batch):
        xb, yb = split["x"][i:i + batch], split["y"][i:i + batch]
        logits = model(xb)
        tot_loss += float(model.loss(logits, yb).numpy()) * yb.size
        tot_correct += int((logits.numpy().argmax(-1) == yb).sum())
        n_tok += yb.size
    return tot_loss / n_tok, tot_correct / n_tok


def evaluate_by_process(model, split, K):
    out = []
    for k in range(K):
        idx = split["z"] == k
        out.append(evaluate(model, {"x": split["x"][idx], "y": split["y"][idx]})[0])
    return out


def train_loop(model, params, data, lr, steps, batch, rng, eval_every, log_prefix=""):
    opt = ndl.optim.Adam(params, lr=lr)
    n = len(data["train"]["x"])
    perm, ptr = rng.permutation(n), 0
    hist = {"train_loss": [], "val": [(0,) + evaluate(model, data["val"])]}
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if ptr + batch > n:
            perm, ptr = rng.permutation(n), 0
        idx = perm[ptr:ptr + batch]
        ptr += batch
        opt.reset_grad()
        loss = model.loss(model(data["train"]["x"][idx]), data["train"]["y"][idx])
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == 1:
            hist["train_loss"].append((step, float(loss.numpy())))
        if step % eval_every == 0 or step == steps:
            hist["val"].append((step,) + evaluate(model, data["val"]))
            if log_prefix:
                print(f"{log_prefix} step {step} train {float(loss.numpy()):.4f} val CE {hist['val'][-1][1]:.4f} acc {hist['val'][-1][2]:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    hist["elapsed_seconds"] = time.perf_counter() - t0
    return hist


def pretrain(cfg):
    d = backbone_dir(cfg)
    os.makedirs(d, exist_ok=True)
    procs = st.make_processes(cfg, cfg["data"]["process_seed"])
    pcfg = copy.deepcopy(cfg)
    pcfg["data"]["n_train"] = cfg["data"]["pretrain_n_train"]
    data = st.make_splits(pcfg, procs, cfg["data"]["pretrain_data_seed"], "pretrain")
    np.random.seed(cfg["pretrain"]["seed"])
    model = build_backbone(cfg)
    rng = np.random.default_rng(derive_seed(cfg["pretrain"]["seed"], "pretrain-shuffle"))
    hist = train_loop(model, model.parameters(), data, cfg["pretrain"]["learning_rate"], cfg["pretrain"]["steps"],
                      cfg["pretrain"]["batch_size"], rng, 100, log_prefix="[pretrain]")
    save_state(model, os.path.join(d, "backbone.npz"))
    adapt = st.make_splits(cfg, procs, cfg["data"]["adapt_data_seed"], "adapt")
    info = {
        "config": cfg, "history": hist, "n_parameters": count_parameters(model.parameters()),
        "pretrain_val": evaluate(model, data["val"]), "pretrain_test": evaluate(model, data["test"]),
        "pretrain_oracle_entropy_rate_test": st.entropy_rate(procs["pretrain"], data["test"]),
        "pretrain_oracle_accuracy_test": st.oracle_accuracy(procs["pretrain"], data["test"]),
        "unigram_entropy_nats": float(np.log(cfg["data"]["vocab_size"])),
        "frozen_on_adapt_val": evaluate(model, adapt["val"]), "frozen_on_adapt_test": evaluate(model, adapt["test"]),
        "adapt_oracle_entropy_rate_test": st.entropy_rate(procs["adapt"], adapt["test"]),
        "adapt_oracle_accuracy_test": st.oracle_accuracy(procs["adapt"], adapt["test"]),
        "adapt_process_identifiability_test": st.process_identifiability(procs, adapt["test"], "adapt"),
        "environment": environment_info(),
    }
    with open(os.path.join(d, "pretrain_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("config", "history", "environment")}, indent=1))


def load_backbone(cfg):
    model = build_backbone(cfg)
    load_state(model, os.path.join(backbone_dir(cfg), "backbone.npz"))
    freeze_backbone(model)
    return model


def finetune_one(cfg, name, spec, data, procs, seed, lr):
    a = cfg["adapters"]
    model = load_backbone(cfg)
    frozen_before = {n: p.numpy().copy() for n, p in named_parameters(model)}
    rec = {"method": name, "seed": seed, "learning_rate": lr, "spec": spec}
    if spec["kind"] == "frozen":
        params, hist = [], {"train_loss": [], "val": [(0,) + evaluate(model, data["val"])], "elapsed_seconds": 0.0}
    else:
        np.random.seed(derive_seed(seed, "adapter-init"))
        inject_adapters(model, spec["kind"], rank=spec.get("rank", a["rank"]), alpha=a["alpha"], n_experts=a["n_experts"],
                        top_k=spec.get("top_k", a["top_k"]), router_grad_mode=spec.get("router_grad_mode", "renormalized"),
                        where=tuple(a["where"]))
        params = adapter_trainable_parameters(model)
        rng = np.random.default_rng(derive_seed(seed, "adapt-shuffle"))
        hist = train_loop(model, params, data, lr, cfg["finetune"]["steps"], cfg["finetune"]["batch_size"], rng, cfg["finetune"]["eval_every"])
    rec["trainable_parameters"] = count_parameters(params)
    rec["total_parameters"] = count_parameters(model.parameters())
    rec["history"] = hist
    rec["elapsed_seconds"] = hist["elapsed_seconds"]
    rec["val_ce"], rec["val_acc"] = evaluate(model, data["val"])
    rec["test_ce"], rec["test_acc"] = evaluate(model, data["test"])
    rec["test_ce_by_process"] = evaluate_by_process(model, data["test"], cfg["data"]["n_processes"])
    after = dict(named_parameters(model))
    rec["frozen_backbone_unchanged"] = all(np.array_equal(after[n].numpy(), arr) for n, arr in frozen_before.items() if n in after)
    rec["all_finite"] = bool(np.isfinite(rec["test_ce"]) and all(np.isfinite(v) for _, v in hist["train_loss"]))
    if spec["kind"] in ("dense", "sparse"):
        rec["routing"] = routing_stats(model, data["test"]["x"][:128])
        # routing by true process (first block fc1 layer): mean prob per process
        if spec["kind"] == "sparse":
            model(data["test"]["x"][:128])
            rec["execution"] = sparse_execution_stats(model)
    return rec


def run_grid(cfg, run_dir, seeds, methods, lrs, label):
    path = os.path.join(run_dir, "runs.jsonl")
    done = {(r["method"], r["seed"]) for r in _load(path)}
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    procs = st.make_processes(cfg, cfg["data"]["process_seed"])
    data = st.make_splits(cfg, procs, cfg["data"]["adapt_data_seed"], "adapt")
    for seed in seeds:
        for name, spec in methods.items():
            if (name, seed) in done:
                continue
            rec = finetune_one(cfg, name, spec, data, procs, seed, lrs.get(name))
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{label}] seed {seed} {name:16s} test CE {rec['test_ce']:.4f} acc {rec['test_acc']:.3f} params {rec['trainable_parameters']:6d} "
                  f"{rec['elapsed_seconds']:.0f}s frozen {rec['frozen_backbone_unchanged']}", flush=True)


def _load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def pilot(cfg, run_dir):
    methods = {k: v for k, v in cfg["methods"].items() if v["kind"] != "frozen"}
    table = {}
    for lr in cfg["pilot"]["learning_rates"]:
        d = os.path.join(run_dir, f"lr_{lr}")
        os.makedirs(d, exist_ok=True)
        run_grid(cfg, d, [cfg["pilot"]["seed"]], methods, {m: lr for m in methods}, f"pilot lr={lr}")
        for r in _load(os.path.join(d, "runs.jsonl")):
            table.setdefault(r["method"], {})[str(lr)] = r["val_ce"]
    choice = {m: min(d, key=d.get) for m, d in table.items()}
    with open(os.path.join(run_dir, "pilot_results.json"), "w") as f:
        json.dump({"table": table, "chosen": choice, "rule": "lowest validation CE"}, f, indent=1)
    print(json.dumps(table, indent=1)); print("chosen:", choice)


def analyze(cfg, run_dir):
    runs = _load(os.path.join(run_dir, "runs.jsonl"))
    info = json.load(open(os.path.join(backbone_dir(cfg), "pretrain_info.json")))
    by = {}
    for r in runs:
        by.setdefault(r["method"], {})[r["seed"]] = r
    methods = [m for m in LABEL if m in by]
    seeds = sorted(set.intersection(*[set(by[m]) for m in methods]))
    an = cfg["analysis"]
    dense = [by["dense_molora"][s]["test_ce"] for s in seeds]
    stats = {}
    for m in methods:
        ce = [by[m][s]["test_ce"] for s in seeds]
        acc = [by[m][s]["test_acc"] for s in seeds]
        stt = {"test_ce": summary_stats(ce), "test_acc": summary_stats(acc), "val_ce": summary_stats([by[m][s]["val_ce"] for s in seeds]),
               "trainable_parameters": by[m][seeds[0]]["trainable_parameters"], "total_parameters": by[m][seeds[0]]["total_parameters"],
               "mean_elapsed_s": float(np.mean([by[m][s]["elapsed_seconds"] for s in seeds])),
               "frozen_backbone_unchanged_all": all(by[m][s]["frozen_backbone_unchanged"] for s in seeds),
               "test_ce_by_process_mean": np.mean([by[m][s]["test_ce_by_process"] for s in seeds], axis=0).tolist()}
        if m != "dense_molora":
            stt["vs_dense"] = paired_bootstrap_ci(ce, dense, an["n_bootstrap"], an["bootstrap_seed"], an["alpha"])
            stt["fraction_better_than_dense"] = float(np.mean([a < b for a, b in zip(ce, dense)]))
        if "routing" in by[m][seeds[0]]:
            stt["routing_entropy_mean_by_layer"] = np.mean([[l["entropy_nats"] for l in by[m][s]["routing"]] for s in seeds], axis=0).tolist()
            stt["utilization_mean_by_layer"] = np.mean([[sorted(l["utilization"]) for l in by[m][s]["routing"]] for s in seeds], axis=0).tolist()
        if "execution" in by[m][seeds[0]]:
            stt["expert_flop_fraction_vs_dense"] = float(np.mean([by[m][s]["execution"]["expert_flop_fraction_vs_dense"] for s in seeds]))
        stats[m] = stt
    out = {"seeds": seeds, "stats": stats, "backbone": {k: v for k, v in info.items() if k not in ("config", "history", "environment")},
           "environment": environment_info()}
    with open(os.path.join(run_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    plot(by, methods, seeds, info, run_dir)
    text = summary(cfg, stats, methods, seeds, info, run_dir)
    print(text)


def plot(by, methods, seeds, info, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=PAL["surface"])
    _style(axes[0]); _style(axes[1])
    for m in methods:
        if m == "frozen":
            continue
        curves = np.array([[v[1] for v in by[m][s]["history"]["val"]] for s in seeds])
        steps = [v[0] for v in by[m][seeds[0]]["history"]["val"]]
        axes[0].plot(steps, curves.mean(0), color=COLOR[m], lw=1.8, label=LABEL[m])
        axes[0].fill_between(steps, curves.min(0), curves.max(0), color=COLOR[m], alpha=0.15, lw=0)
    axes[0].axhline(by["frozen"][seeds[0]]["val_ce"], color=COLOR["frozen"], ls="--", lw=1.2, label="frozen backbone")
    axes[0].axhline(info["adapt_oracle_entropy_rate_test"], color=PAL["muted"], ls=":", lw=1.2, label="oracle entropy rate")
    axes[0].set_xlabel("fine-tuning step", color=PAL["muted"]); axes[0].set_ylabel("validation cross-entropy (nats/token)", color=PAL["muted"])
    axes[0].set_title("Adapter fine-tuning on the shifted processes (mean, min–max over seeds)", fontsize=9, loc="left", color=PAL["text"])
    axes[0].legend(frameon=False, fontsize=7)
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        v = [by[m][s]["test_ce"] for s in seeds]
        axes[1].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, color=COLOR[m], s=18, zorder=3)
        axes[1].hlines(np.median(v), i - 0.3, i + 0.3, color=PAL["text"], lw=1.4)
    axes[1].axhline(info["adapt_oracle_entropy_rate_test"], color=PAL["muted"], ls=":", lw=1.2)
    axes[1].set_xticks(range(len(methods)), [LABEL[m] for m in methods], fontsize=7, rotation=20)
    axes[1].set_ylabel("test cross-entropy (nats/token)", color=PAL["muted"])
    axes[1].set_title("Held-out test CE per seed (bar = median; dotted = oracle)", fontsize=9, loc="left", color=PAL["text"])
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "transformer_finetune.png"), dpi=130, bbox_inches="tight", facecolor=PAL["surface"]); plt.close(fig)


def summary(cfg, stats, methods, seeds, info, run_dir):
    L = [f"# Phase 4C transformer adaptation — {len(seeds)} seeds ({seeds[0]}–{seeds[-1]})", "",
         f"Backbone: {info['n_parameters']} params, pretrain test CE {info['pretrain_test'][0]:.4f} (oracle {info['pretrain_oracle_entropy_rate_test']:.4f}, "
         f"unigram {info['unigram_entropy_nats']:.4f}), acc {info['pretrain_test'][1]:.3f} (oracle {info['pretrain_oracle_accuracy_test']:.3f}).  ",
         f"Shifted adaptation data: frozen backbone test CE {info['frozen_on_adapt_test'][0]:.4f} / acc {info['frozen_on_adapt_test'][1]:.3f}; "
         f"oracle entropy rate {info['adapt_oracle_entropy_rate_test']:.4f} / oracle acc {info['adapt_oracle_accuracy_test']:.3f}; "
         f"Bayes process identifiability from a whole sequence {info['adapt_process_identifiability_test']:.3f}.", "",
         "| method | trainable params | test CE mean ± std | median | min–max | test acc mean | Δ CE vs dense [95% paired CI] | better than dense | frozen OK | time/run (s) | routing entropy by layer | sparse FLOP frac |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        s = stats[m]; t = s["test_ce"]
        vd = s.get("vs_dense"); vdt = f"{vd['estimate']:+.4f} [{vd['ci_low']:+.4f}, {vd['ci_high']:+.4f}]" if vd else "—"
        fb = f"{s['fraction_better_than_dense']:.2f}" if vd else "—"
        ent = "/".join(f"{e:.2f}" for e in s["routing_entropy_mean_by_layer"]) if "routing_entropy_mean_by_layer" in s else "—"
        ff = f"{s['expert_flop_fraction_vs_dense']:.2f}" if "expert_flop_fraction_vs_dense" in s else "—"
        L.append(f"| {LABEL[m]} | {s['trainable_parameters']} | {t['mean']:.4f} ± {t['std']:.4f} | {t['median']:.4f} | {t['min']:.4f}–{t['max']:.4f} | {s['test_acc']['mean']:.3f} | {vdt} | {fb} | "
                 f"{s['frozen_backbone_unchanged_all']} | {s['mean_elapsed_s']:.0f} | {ent} | {ff} |")
    L += ["", "Per-process test CE (seed-averaged): " + "; ".join(f"{LABEL[m]} {['%.3f' % v for v in stats[m]['test_ce_by_process_mean']]}" for m in methods)]
    text = "\n".join(L) + "\n"
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pretrain", "pilot", "study", "analyze"])
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "phase4_transformer.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+")
    args = ap.parse_args(argv)
    cfg = copy.deepcopy(yaml.safe_load(open(args.config)))
    root = os.path.join(REPO_ROOT, cfg["experiment"]["output_dir"], cfg["experiment"]["name"])
    if args.command == "pretrain":
        pretrain(cfg); return 0
    if args.command == "pilot":
        d = os.path.join(root, "pilot_" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")); os.makedirs(d, exist_ok=True)
        print("[molora] writing to", d); pilot(cfg, d); return 0
    if args.command == "study":
        lrs = cfg["finetune"]["learning_rate"]
        if any(lrs.get(m) is None for m, v in cfg["methods"].items() if v["kind"] != "frozen"):
            raise SystemExit("run the pilot and record learning rates first")
        d = args.run_dir or os.path.join(root, "study_" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")); os.makedirs(d, exist_ok=True)
        print("[molora] writing to", d)
        run_grid(cfg, d, args.seeds or cfg["experiment"]["seeds"], cfg["methods"], lrs, "study"); return 0
    if args.command == "analyze":
        analyze(cfg, args.run_dir); return 0


if __name__ == "__main__":
    raise SystemExit(main())
