"""Train one CNN, evaluate it on a held-out test set, and check what its filters learned.

Usage:
    python experiments/train_model.py                       # synthetic GATA-motif task
    python experiments/train_model.py --pos pos.fa --neg neg.fa   # your own FASTA files
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dnacnn import data, interpret  # noqa: E402
from dnacnn.baseline import pwm_scan_scores  # noqa: E402
from dnacnn.train import roc_auc  # noqa: E402
from dnacnn.model import DNACNN  # noqa: E402
from dnacnn.train import evaluate, train  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def split(seqs, labels, seed=0):
    """70 / 15 / 15 train / validation / test split."""
    idx = np.random.default_rng(seed).permutation(len(seqs))
    a, b = int(0.7 * len(idx)), int(0.85 * len(idx))
    parts = idx[:a], idx[a:b], idx[b:]
    return [([seqs[i] for i in p], labels[p]) for p in parts]


def plot_pwm(ax, pwm, title):
    im = ax.imshow(pwm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(4), list(data.BASES))
    ax.set_xticks(range(pwm.shape[1]), [str(i + 1) for i in range(pwm.shape[1])], fontsize=7)
    ax.set_title(title, fontsize=9)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", help="FASTA of positive (bound) sequences")
    parser.add_argument("--neg", help="FASTA of negative sequences")
    parser.add_argument("--pooling", default="max", choices=["max", "avg", "none"])
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.pos and args.neg:
        pos, neg = data.read_fasta(args.pos), data.read_fasta(args.neg)
        L = min(map(len, pos + neg))
        seqs = [s[:L] for s in pos + neg]
        labels = np.array([1] * len(pos) + [0] * len(neg))
        motif = None
    else:
        seqs, labels, _ = data.make_synthetic_dataset(n=args.n, seed=args.seed)
        L, motif = len(seqs[0]), data.DEFAULT_MOTIF

    (tr_s, tr_y), (va_s, va_y), (te_s, te_y) = split(seqs, labels, args.seed)
    torch.manual_seed(args.seed)
    model = DNACNN(seq_len=L, pooling=args.pooling)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{len(tr_s)} train / {len(va_s)} val / {len(te_s)} test sequences of length {L}; "
          f"{n_params:,} parameters")

    train(model, data.to_tensor_dataset(tr_s, tr_y), data.to_tensor_dataset(va_s, va_y),
          epochs=args.epochs, seed=args.seed)
    test = evaluate(model, data.to_tensor_dataset(te_s, te_y))
    print(f"TEST accuracy {test['accuracy']:.4f}  ROC-AUC {test['auc']:.4f}")
    if motif:
        test["oracle_pwm_auc"] = roc_auc(te_y, pwm_scan_scores(te_s, motif, data.DEFAULT_STRENGTH))
        test["pct_of_oracle"] = test["auc"] / test["oracle_pwm_auc"]
        print(f"Oracle PWM-scan AUC {test['oracle_pwm_auc']:.4f} -> CNN reaches "
              f"{100 * test['pct_of_oracle']:.1f}% of it without being told the motif")
    torch.save(model.state_dict(), RESULTS / f"model_{args.pooling}.pt")

    # ---- Interpretation: which filters learned a motif? ----
    pos_seqs = [s for s, y in zip(te_s, te_y) if y == 1]
    filters = [f for f in interpret.filter_pwms(model, pos_seqs) if f["pwm"] is not None]
    for f in filters:
        f["ic"] = interpret.information_content(f["pwm"])
        f["consensus"] = interpret.consensus(f["pwm"])
        if motif:
            f["match_r"], f["offset"], f["strand"] = interpret.best_match(f["pwm"], motif)
    top = sorted(filters, key=lambda f: f.get("match_r", f["ic"]), reverse=True)[:4]

    report = {"test": test, "n_params": n_params, "pooling": args.pooling,
              "top_filters": [{k: f[k] for k in ("filter", "consensus", "ic", "n_sites",
                                                  "match_r", "strand") if k in f} for f in top]}
    if motif:
        report["planted_motif"] = motif
        print(f"Planted motif: {motif} (reverse complement {data.reverse_complement(motif)})")
    for f in report["top_filters"]:
        print("  filter {filter:2d}  consensus {consensus}  IC {ic:.1f} bits".format(**f)
              + (f"  match r={f['match_r']:.2f} ({f['strand']} strand)" if motif else ""))
    (RESULTS / f"train_report_{args.pooling}.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(len(top) + (1 if motif else 0), 1, figsize=(7, 1.6 * (len(top) + 1)))
    axes = np.atleast_1d(axes)
    i = 0
    if motif:
        plot_pwm(axes[0], data.motif_pwm(motif), f"Planted motif: {motif}")
        i = 1
    for ax, f in zip(axes[i:], top):
        label = f"Filter {f['filter']}: {f['consensus']}"
        if motif:
            label += f"  (r={f['match_r']:.2f}, {f['strand']} strand)"
        plot_pwm(ax, f["pwm"], label)
    fig.suptitle("What the first convolutional layer learned", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "learned_filters.png", dpi=130)
    print(f"Saved results to {RESULTS}")


if __name__ == "__main__":
    main()
