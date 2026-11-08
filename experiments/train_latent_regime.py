#!/usr/bin/env python
"""Phase 4B — implicit regime discovery on the latent-regime dataset.

    python experiments/train_latent_regime.py pilot              # validation-only LR pilot (seeds 3000-3001, all conditions)
    python experiments/train_latent_regime.py study              # confirmatory: conditions x methods x seeds (resume-safe)
    python experiments/train_latent_regime.py analyze RUN_DIR    # statistics, plots, summary.md

Models receive x (and y for the loss) only; the regime z is used for diagnostics.
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
from molora import DenseMoLoRALinear, SparseMoLoRALinear
from molora.protocol import binomial_ci, is_failure, paired_bootstrap_ci, summary_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import latent_regime_dataset as lrd  # noqa: E402
from robustness_study import build_model, load_runs, routing_summary, train_one  # noqa: E402
from train_dense_molora import PAL, _style, environment_info, mse_by_regime, predict  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = {"frozen": "frozen", "single_lora": "LoRA r=8", "matched_lora": "LoRA r=34", "dense_molora": "dense 4x8",
         "sparse_top2": "sparse top-2", "sparse_top1_ste": "sparse top-1 (STE)"}
COLOR = {"frozen": "#52514e", "single_lora": "#2a78d6", "matched_lora": "#eb6834", "dense_molora": "#1baf7a",
         "sparse_top2": "#e87ba4", "sparse_top1_ste": "#eda100"}


def make_extra_eval(cfg, problem):
    thr = cfg["ambiguity_threshold"]
    masks = {s: lrd.ambiguity_mask(problem, s, thr) for s in ("val", "test")}

    def extra(model, problem):
        out = {}
        for split in ("val", "test"):
            pred = predict(model, problem[split]["x"], cfg["dtype"])
            err = ((pred - problem[split]["y"]) ** 2).mean(axis=1)
            m = masks[split]
            out[f"{split}_mse_ambiguous"] = float(err[m].mean()) if m.any() else None
            out[f"{split}_mse_unambiguous"] = float(err[~m].mean()) if (~m).any() else None
            out[f"{split}_ambiguous_fraction"] = float(m.mean())
        if isinstance(model, DenseMoLoRALinear):
            x = ndl.Tensor(problem["test"]["x"], dtype=cfg["dtype"], requires_grad=False)
            p = model.router_probabilities(x).astype(np.float64)
            top = p.argmax(1)
            z = problem["test"]["z"]
            R, E = len(problem["mus"]), p.shape[1]
            cont = np.array([[np.sum((z == r) & (top == e)) for e in range(E)] for r in range(R)], float)
            out["routing_purity_test"] = float(cont.max(1).sum() / len(z))  # fraction of tokens whose regime's majority expert is theirs
            # agreement between router argmax and Bayes-optimal regime (up to the best regime->expert matching = purity)
            out["router_argmax_matches_bayes_fraction"] = float(np.mean(top == _match(cont)[lrd.bayes_posterior(problem["test"]["x"], problem["mus"], problem["sigma"]).argmax(1)]))
        return out

    return extra


def _match(cont):
    """Greedy regime -> expert assignment from a contingency table (R x E)."""
    cont = cont.copy()
    assign = np.full(cont.shape[0], -1)
    for _ in range(min(cont.shape)):
        r, e = np.unravel_index(cont.argmax(), cont.shape)
        assign[r] = e
        cont[r, :] = -1
        cont[:, e] = -1
    return assign


def run_frozen(problem, cfg, seed, cond):
    R = cfg["data"]["n_regimes"]
    rec = {"method": "frozen", "seed": seed, "condition": cond, "trainable_parameters": 0, "elapsed_seconds": 0.0,
           "frozen_backbone_unchanged": True, "all_finite": True}
    for split in ("val", "test"):
        rec[f"{split}_mse"], rec[f"{split}_mse_by_regime"] = mse_by_regime(predict(problem["base"], problem[split]["x"], cfg["dtype"]), problem[split], R)
    rec.update(make_extra_eval(cfg, problem)(problem["base"], problem))
    rec["failed"] = True  # cannot route; reported for completeness
    return rec


def failure_threshold(cfg, oracle_val):
    return max(cfg["failure"]["val_mse_threshold"], cfg["failure"]["oracle_multiplier"] * oracle_val["soft_bayes_oracle_mse"])


def run_grid(cfg, run_dir, seeds, conditions, methods, lrs, label):
    path = os.path.join(run_dir, "runs.jsonl")
    done = {(r["condition"], r["method"], r["seed"]) for r in load_runs(path)}
    prob_path = os.path.join(run_dir, "problems.jsonl")
    done_prob = {(r["condition"], r["seed"]) for r in load_runs(prob_path)}
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    t_all = time.perf_counter()
    for cond, cspec in conditions.items():
        for seed in seeds:
            problem = lrd.make_latent_problem(cfg, seed, delta=cspec["delta"])
            oracle_val, oracle_test = lrd.oracle_mses(problem, "val"), lrd.oracle_mses(problem, "test")
            thr = failure_threshold(cfg, oracle_val)
            if (cond, seed) not in done_prob:
                with open(prob_path, "a") as f:
                    f.write(json.dumps({"condition": cond, "seed": seed, "delta": cspec["delta"], "identifiability": lrd.identifiability(problem, cfg["ambiguity_threshold"]),
                                        "oracle_val": oracle_val, "oracle_test": oracle_test, "failure_threshold": thr}) + "\n")
            extra = make_extra_eval(cfg, problem)
            for name, spec in methods.items():
                if (cond, name, seed) in done:
                    continue
                if spec["kind"] == "frozen":
                    rec = run_frozen(problem, cfg, seed, cond)
                else:
                    rec = train_one(name, spec, problem, cfg, seed, lrs[name], extra_eval=extra)
                    rec["condition"] = cond
                    rec["failed"] = is_failure(rec["val_mse"], thr)
                rec["failure_threshold"] = thr
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                flag = " FAIL" if rec["failed"] and spec["kind"] not in ("frozen", "lora") else ""
                print(f"[{label}] {cond:6s} seed {seed} {name:16s} val {rec['val_mse']:.3e} test {rec['test_mse']:.3e} "
                      f"amb {rec.get('test_mse_ambiguous') or 0:.2e} unamb {rec.get('test_mse_unambiguous') or 0:.2e} {rec['elapsed_seconds']:.1f}s{flag}", flush=True)
    print(f"[{label}] done in {time.perf_counter() - t_all:.0f}s -> {path}")


def pilot(cfg, run_dir):
    methods = {k: v for k, v in cfg["methods"].items() if v["kind"] != "frozen"}
    table = {}
    for lr in cfg["pilot"]["learning_rates"]:
        d = os.path.join(run_dir, f"lr_{lr}")
        os.makedirs(d, exist_ok=True)
        run_grid(cfg, d, cfg["pilot"]["seeds"], cfg["conditions"], methods, {m: lr for m in methods}, f"pilot lr={lr}")
        for r in load_runs(os.path.join(d, "runs.jsonl")):
            table.setdefault(r["method"], {}).setdefault(str(lr), []).append(r["val_mse"])
    choice = {m: min(d, key=lambda lr: float(np.mean(np.log10(d[lr])))) for m, d in table.items()}  # geometric-mean val MSE over conditions & seeds
    out = {"table": {m: {lr: {"val_mses": v, "geo_mean": float(10 ** np.mean(np.log10(v)))} for lr, v in d.items()} for m, d in table.items()},
           "chosen": choice, "rule": "lowest geometric-mean validation MSE over the 3 conditions x 2 pilot seeds"}
    with open(os.path.join(run_dir, "pilot_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    for m, d in out["table"].items():
        print(m, {lr: f"{v['geo_mean']:.3e}" for lr, v in d.items()}, "->", choice[m])


def analyze(cfg, run_dir):
    runs = load_runs(os.path.join(run_dir, "runs.jsonl"))
    probs = {(p["condition"], p["seed"]): p for p in load_runs(os.path.join(run_dir, "problems.jsonl"))}
    an = cfg["analysis"]
    by = {}
    for r in runs:
        by.setdefault(r["condition"], {}).setdefault(r["method"], {})[r["seed"]] = r
    conds = [c for c in cfg["conditions"] if c in by]
    methods = [m for m in LABEL if any(m in by[c] for c in conds)]
    stats = {}
    for c in conds:
        seeds = sorted(set.intersection(*[set(by[c][m]) for m in methods if m in by[c]]))
        ident = {k: float(np.mean([probs[(c, s)]["identifiability"][k] for s in seeds])) for k in
                 ("bayes_accuracy_test", "nearest_centroid_accuracy_test", "ambiguous_fraction_test", "mean_max_posterior_test")}
        oracle = {k: float(np.mean([probs[(c, s)]["oracle_test"][k] for s in seeds])) for k in probs[(c, seeds[0])]["oracle_test"]}
        stats[c] = {"seeds": seeds, "delta": cfg["conditions"][c]["delta"], "pairwise_distance": cfg["conditions"][c]["delta"] * np.sqrt(2),
                    "identifiability": ident, "oracle_test": oracle, "methods": {}}
        dense = [by[c]["dense_molora"][s]["test_mse"] for s in seeds]
        for m in methods:
            d = by[c][m]
            test = [d[s]["test_mse"] for s in seeds]
            st = {"test": summary_stats(test), "val": summary_stats([d[s]["val_mse"] for s in seeds]),
                  "test_mean_bootstrap": paired_bootstrap_ci(test, [0.0] * len(test), an["n_bootstrap"], an["bootstrap_seed"], an["alpha"]),
                  "test_ambiguous_mean": float(np.mean([d[s]["test_mse_ambiguous"] for s in seeds if d[s]["test_mse_ambiguous"] is not None] or [np.nan])),
                  "test_unambiguous_mean": float(np.mean([d[s]["test_mse_unambiguous"] for s in seeds if d[s]["test_mse_unambiguous"] is not None] or [np.nan])),
                  "per_regime_test_mean": np.mean([d[s]["test_mse_by_regime"] for s in seeds], axis=0).tolist(),
                  "trainable_parameters": d[seeds[0]]["trainable_parameters"], "mean_elapsed_s": float(np.mean([d[s]["elapsed_seconds"] for s in seeds]))}
            if m not in ("frozen", "single_lora", "matched_lora"):
                fails = int(sum(d[s]["failed"] for s in seeds))
                st["failures"], st["failure_ci95"] = fails, list(binomial_ci(fails, len(seeds), an["alpha"]))
                st["routing_purity_mean"] = float(np.mean([d[s]["routing_purity_test"] for s in seeds]))
                st["router_matches_bayes_mean"] = float(np.mean([d[s]["router_argmax_matches_bayes_fraction"] for s in seeds]))
                st["entropy_mean"] = float(np.mean([d[s]["final_routing_test"]["entropy_nats"] for s in seeds]))
                st["utilization_mean_sorted"] = np.mean(np.sort([d[s]["final_routing_test"]["utilization"] for s in seeds], axis=1), axis=0).tolist()
                st["regime_expert_selection_rate_seed0"] = d[seeds[0]]["final_routing_test"]["regime_expert_selection_rate"]
            if m != "dense_molora":
                st["vs_dense"] = {"mean_diff_bootstrap": paired_bootstrap_ci(test, dense, an["n_bootstrap"], an["bootstrap_seed"], an["alpha"]),
                                  "log10_ratio_mean_bootstrap": paired_bootstrap_ci(np.log10(test), np.log10(dense), an["n_bootstrap"], an["bootstrap_seed"], an["alpha"]),
                                  "fraction_better": float(np.mean([t < u for t, u in zip(test, dense)]))}
            stats[c]["methods"][m] = st
    out = {"stats": stats, "environment": environment_info(), "run_dir": run_dir}
    with open(os.path.join(run_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    plots(cfg, by, stats, conds, methods, run_dir)
    text = write_summary(cfg, stats, conds, methods, run_dir)
    print(text)
    return out


def plots(cfg, by, stats, conds, methods, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    # 1. test MSE vs separability
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=PAL["surface"])
    _style(ax)
    xs = [stats[c]["pairwise_distance"] for c in conds]
    for m in methods:
        means = [stats[c]["methods"][m]["test"]["mean"] for c in conds]
        lo = [stats[c]["methods"][m]["test_mean_bootstrap"]["ci_low"] for c in conds]
        hi = [stats[c]["methods"][m]["test_mean_bootstrap"]["ci_high"] for c in conds]
        ax.plot(xs, means, color=COLOR[m], lw=1.8, marker="o", ms=4, label=LABEL[m])
        ax.fill_between(xs, lo, hi, color=COLOR[m], alpha=0.15, lw=0)
    for key, ls, lab in (("hard_bayes_oracle_mse", "--", "hard-Bayes oracle"), ("soft_bayes_oracle_mse", ":", "soft-Bayes oracle")):
        ax.plot(xs, [stats[c]["oracle_test"][key] for c in conds], color=PAL["muted"], ls=ls, lw=1.2, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"{c}\n(Δ={x:.2f})" for c, x in zip(conds, xs)], fontsize=8)
    ax.set_xlabel("regime separability: pairwise Mahalanobis distance between means", color=PAL["muted"], fontsize=9)
    ax.set_ylabel("held-out test MSE (mean, 95% CI; log)", color=PAL["muted"], fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("Test MSE vs regime overlap (10 seeds per condition)", fontsize=10, loc="left", color=PAL["text"])
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "test_mse_vs_separability.png"), dpi=130, bbox_inches="tight", facecolor=PAL["surface"]); plt.close(fig)
    # 2. ambiguous vs unambiguous
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.6), sharey=True, squeeze=False, facecolor=PAL["surface"])
    for ax, c in zip(axes[0], conds):
        _style(ax)
        ms = [m for m in methods if m != "frozen"]
        xi = np.arange(len(ms))
        amb = [stats[c]["methods"][m]["test_ambiguous_mean"] for m in ms]
        un = [stats[c]["methods"][m]["test_unambiguous_mean"] for m in ms]
        ax.bar(xi - 0.2, un, 0.38, color=[COLOR[m] for m in ms], alpha=0.95, label="unambiguous (max posterior ≥ 0.9)")
        ax.bar(xi + 0.2, amb, 0.38, color=[COLOR[m] for m in ms], alpha=0.45, hatch="//", label="ambiguous (< 0.9)")
        ax.set_yscale("log")
        ax.set_xticks(xi, [LABEL[m] for m in ms], fontsize=7, rotation=25)
        frac = stats[c]["identifiability"]["ambiguous_fraction_test"]
        ax.set_title(f"{c}: {100 * frac:.0f}% of test points ambiguous", fontsize=9, loc="left", color=PAL["text"])
    axes[0][0].set_ylabel("test MSE (log)", color=PAL["muted"])
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "ambiguous_vs_unambiguous.png"), dpi=130, bbox_inches="tight", facecolor=PAL["surface"]); plt.close(fig)
    # 3. routing selection-rate heatmaps, seed 0 of each condition, MoE methods
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    moe = [m for m in ("dense_molora", "sparse_top2", "sparse_top1_ste") if m in methods]
    fig, axes = plt.subplots(len(moe), len(conds), figsize=(3.6 * len(conds), 3.0 * len(moe)), squeeze=False, facecolor=PAL["surface"])
    fig.subplots_adjust(wspace=0.5, hspace=0.55)
    for i, m in enumerate(moe):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            key = "regime_expert_mean_prob" if m == "dense_molora" else "regime_expert_selection_rate"
            s0 = stats[c]["seeds"][0]
            mat = np.array(by[c][m][s0]["final_routing_test"][key])
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for a in range(mat.shape[0]):
                for b in range(mat.shape[1]):
                    ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=7, color="#f0efec" if mat[a, b] > 0.55 else PAL["text"])
            ax.set_xticks(range(mat.shape[1]), [f"e{k + 1}" for k in range(mat.shape[1])], fontsize=7)
            ax.set_yticks(range(mat.shape[0]), [f"z={k + 1}" for k in range(mat.shape[0])], fontsize=7)
            ax.set_title(f"{LABEL[m]}, {c} (seed {s0})", fontsize=8, loc="left", color=PAL["text"])
            ax.tick_params(length=0, colors=PAL["muted"])
            for sp in ax.spines.values():
                sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cb.set_label("mean probability (dense) / selection rate (sparse)", fontsize=8, color=PAL["muted"])
    fig.suptitle("Routing by TRUE latent regime (rows) and expert (columns); regime never seen by the model", fontsize=10, color=PAL["text"])
    fig.savefig(os.path.join(run_dir, "routing_by_latent_regime.png"), dpi=130, bbox_inches="tight", facecolor=PAL["surface"]); plt.close(fig)


def write_summary(cfg, stats, conds, methods, run_dir):
    L = ["# Phase 4B latent-regime study — summary", ""]
    for c in conds:
        st = stats[c]; idn, orc = st["identifiability"], st["oracle_test"]
        L += [f"## Condition `{c}` — delta {st['delta']}, pairwise mean distance {st['pairwise_distance']:.2f} σ; seeds {st['seeds'][0]}–{st['seeds'][-1]} (n={len(st['seeds'])})", "",
              f"Identifiability (test, averaged over seeds): Bayes-optimal accuracy {idn['bayes_accuracy_test']:.3f}, nearest-centroid accuracy {idn['nearest_centroid_accuracy_test']:.3f}, "
              f"mean max posterior {idn['mean_max_posterior_test']:.3f}, ambiguous fraction (max posterior < {cfg['ambiguity_threshold']}) {idn['ambiguous_fraction_test']:.3f}.  ",
              f"Oracles (test, NOT fair baselines): true-regime {orc['true_regime_oracle_mse']:.2e}, hard-Bayes {orc['hard_bayes_oracle_mse']:.3e}, soft-Bayes {orc['soft_bayes_oracle_mse']:.3e}, frozen {orc['frozen_backbone_mse']:.3e}.", "",
              "| method | params | test MSE mean [95% CI] | median | min–max | Δ vs dense (paired) [CI] | better than dense | failures | ambiguous / unambiguous test MSE | routing purity | router=Bayes | entropy | utilisation (sorted) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in methods:
            s = st["methods"][m]; t = s["test"]; b = s["test_mean_bootstrap"]
            vd = s.get("vs_dense"); vdt = f"{vd['mean_diff_bootstrap']['estimate']:+.2e} [{vd['mean_diff_bootstrap']['ci_low']:+.2e}, {vd['mean_diff_bootstrap']['ci_high']:+.2e}]" if vd else "—"
            fb = f"{vd['fraction_better']:.2f}" if vd else "—"
            fails = f"{s['failures']}/{len(st['seeds'])} [{s['failure_ci95'][0]:.2f}, {s['failure_ci95'][1]:.2f}]" if "failures" in s else "—"
            rp = f"{s['routing_purity_mean']:.3f}" if "routing_purity_mean" in s else "—"
            rb = f"{s['router_matches_bayes_mean']:.3f}" if "router_matches_bayes_mean" in s else "—"
            en = f"{s['entropy_mean']:.2f}" if "entropy_mean" in s else "—"
            ut = "/".join(f"{u:.2f}" for u in s["utilization_mean_sorted"]) if "utilization_mean_sorted" in s else "—"
            L.append(f"| {LABEL[m]} | {s['trainable_parameters']} | {t['mean']:.3e} [{b['ci_low']:.2e}, {b['ci_high']:.2e}] | {t['median']:.3e} | {t['min']:.2e}–{t['max']:.2e} | {vdt} | {fb} | {fails} | "
                     f"{s['test_ambiguous_mean']:.2e} / {s['test_unambiguous_mean']:.2e} | {rp} | {rb} | {en} | {ut} |")
        L.append("")
    text = "\n".join(L) + "\n"
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot", "study", "analyze"])
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "phase4_latent_regime.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+")
    args = ap.parse_args(argv)
    cfg = copy.deepcopy(yaml.safe_load(open(args.config)))
    root = os.path.join(REPO_ROOT, cfg["experiment"]["output_dir"])
    if args.command == "pilot":
        d = os.path.join(root, cfg["experiment"]["name"] + "_pilot", dt.datetime.now().strftime("%Y%m%d-%H%M%S")); os.makedirs(d, exist_ok=True)
        print("[molora] writing to", d); pilot(cfg, d); return 0
    if args.command == "study":
        lrs = cfg["training"]["learning_rate"]
        if any(lrs.get(m) is None for m, v in cfg["methods"].items() if v["kind"] != "frozen"):
            raise SystemExit("run the pilot and record learning rates in the config first")
        d = args.run_dir or os.path.join(root, cfg["experiment"]["name"], dt.datetime.now().strftime("%Y%m%d-%H%M%S")); os.makedirs(d, exist_ok=True)
        print("[molora] writing to", d)
        run_grid(cfg, d, args.seeds or cfg["experiment"]["seeds"], cfg["conditions"], cfg["methods"], lrs, "study"); return 0
    if args.command == "analyze":
        analyze(cfg, args.run_dir); return 0


if __name__ == "__main__":
    raise SystemExit(main())
