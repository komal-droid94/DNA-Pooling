"""Experiment: does max pooling matter for finding motifs in DNA?

Two test conditions, each over several random seeds:
  1. Standard:  motif can be anywhere in train and test sequences.
  2. Position shift: models are trained ONLY on sequences with the motif near the center
     (bases 80-120), then tested on sequences with the motif anywhere. A translation-invariant
     model should still find it; a position-specific model should fail on unfamiliar positions.

Pooling variants: max pooling, average pooling, and no pooling (flatten).
Writes results/pooling_comparison.csv and results/pooling_comparison.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dnacnn import data  # noqa: E402
from dnacnn.baseline import pwm_scan_scores  # noqa: E402
from dnacnn.model import DNACNN  # noqa: E402
from dnacnn.train import evaluate, roc_auc, train  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = [0, 1, 2]
POOLINGS = ["max", "avg", "none"]
N_TRAIN, N_TEST = 12000, 2000


def build(condition: str, seed: int):
    train_range = (80, 120) if condition == "position_shift" else None
    tr_s, tr_y, _ = data.make_synthetic_dataset(N_TRAIN, position_range=train_range, seed=seed)
    va_s, va_y, _ = data.make_synthetic_dataset(1000, position_range=train_range, seed=seed + 100)
    te_s, te_y, te_pos = data.make_synthetic_dataset(N_TEST, seed=seed + 200)  # motif anywhere
    return (data.to_tensor_dataset(tr_s, tr_y), data.to_tensor_dataset(va_s, va_y),
            te_s, te_y, te_pos)


def main():
    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = []
    for condition in ["standard", "position_shift"]:
        for seed in SEEDS:
            train_ds, val_ds, te_s, te_y, te_pos = build(condition, seed)
            test_ds = data.to_tensor_dataset(te_s, te_y)
            # "outside" subset: positives whose motif lies outside the training range
            outside = (te_y == 0) | (te_pos < 70) | (te_pos > 130)
            out_ds = data.to_tensor_dataset([s for s, o in zip(te_s, outside) if o], te_y[outside])
            oracle = roc_auc(te_y, pwm_scan_scores(te_s, data.DEFAULT_MOTIF, data.DEFAULT_STRENGTH))
            rows.append({"condition": condition, "pooling": "oracle", "seed": seed,
                         "test_acc": float("nan"), "test_auc": oracle,
                         "outside_range_auc": float("nan"), "params": 0})
            for pooling in POOLINGS:
                torch.manual_seed(seed)
                model = DNACNN(pooling=pooling)
                train(model, train_ds, val_ds, epochs=25, seed=seed, verbose=False)
                full, out = evaluate(model, test_ds), evaluate(model, out_ds)
                rows.append({"condition": condition, "pooling": pooling, "seed": seed,
                             "test_acc": full["accuracy"], "test_auc": full["auc"],
                             "outside_range_auc": out["auc"],
                             "params": sum(p.numel() for p in model.parameters())})
                print(f"{condition:15s} seed {seed}  {pooling:4s}  acc {full['accuracy']:.4f}  "
                      f"auc {full['auc']:.4f}  auc(motif outside train range) {out['auc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "pooling_comparison.csv", index=False)
    summary = (df.groupby(["condition", "pooling"])
               .agg(test_acc=("test_acc", "mean"), test_acc_sd=("test_acc", "std"),
                    test_auc=("test_auc", "mean"), test_auc_sd=("test_auc", "std"),
                    outside_auc=("outside_range_auc", "mean"), params=("params", "first"))
               .round(4))
    print("\n", summary.to_string())
    summary.to_csv(RESULTS / "pooling_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    titles = {"standard": "Motif anywhere (train & test)",
              "position_shift": "Trained on centered motifs,\ntested on motif anywhere"}
    colors = {"max": "#1f77b4", "avg": "#ff7f0e", "none": "#7f7f7f"}
    for ax, cond in zip(axes, ["standard", "position_shift"]):
        s = summary.loc[cond].reindex(POOLINGS)
        ax.axhline(summary.loc[(cond, "oracle"), "test_auc"], ls="--", color="k", lw=1,
                   label="Oracle (true-motif PWM scan)")
        ax.legend(loc="lower left", fontsize=8)
        ax.bar(POOLINGS, s["test_auc"], yerr=s["test_auc_sd"].fillna(0), capsize=4,
               color=[colors[p] for p in POOLINGS])
        for x, v in enumerate(s["test_auc"]):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set(title=titles[cond], xlabel="Pooling", ylim=(0.4, 1.05))
    axes[0].set_ylabel(f"Test ROC-AUC (mean of {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(RESULTS / "pooling_comparison.png", dpi=130)
    print(f"Saved to {RESULTS}")


if __name__ == "__main__":
    main()
