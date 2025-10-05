import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_alpha_sweep(jsonl_path: str, out_png: str):
    rows = [json.loads(l) for l in open(jsonl_path)]
    rows = [r for r in rows if r["method"] == "delta_bestlayer"]
    # average across clips for each alpha
    by_alpha = {}
    for r in rows:
        key = (r["alpha"])
        by_alpha.setdefault(key, []).append(r)
    alphas = sorted(by_alpha)
    gains  = [np.mean([x["affect_gain"] for x in by_alpha[a]]) for a in alphas]
    kls    = [np.mean([x.get("talker_kl", 0.0) for x in by_alpha[a]]) for a in alphas]
    plt.figure(figsize=(5,3.2))
    plt.plot(alphas, gains, marker="o", label="Affect gain (Δ probe margin)")
    plt.plot(alphas, kls, marker="o", label="Talker KL (if measured)")
    plt.xlabel("alpha"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160)

def plot_method_bars(jsonl_path: str, out_png: str):
    rows = [json.loads(l) for l in open(jsonl_path)]
    # aggregate
    methods = sorted({r["method"] for r in rows})
    f0shift = [np.mean([x["delta_f0_mean"] for x in rows if x["method"]==m]) for m in methods]
    drift   = [np.mean([x["content_f1"] for x in rows if x["method"]==m]) for m in methods]
    x = np.arange(len(methods))
    plt.figure(figsize=(6,3.2))
    plt.bar(x-0.2, f0shift, width=0.4, label="ΔF0 mean")
    plt.bar(x+0.2, drift,   width=0.4, label="Token overlap F1")
    plt.xticks(x, methods, rotation=20)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160)

def plot_drift_vs_gain(jsonl_path: str, out_png: str):
    rows = [json.loads(l) for l in open(jsonl_path)]
    x = [r["content_f1"] for r in rows]
    y = [r["affect_gain"] for r in rows]
    plt.figure(figsize=(4,3.2))
    plt.scatter(x, y, s=12)
    plt.xlabel("Content overlap F1 (higher is better)")
    plt.ylabel("Affect gain (Δ joy-sad probe margin)")
    plt.tight_layout(); plt.savefig(out_png, dpi=160)
