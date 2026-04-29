# ============================================================
# generate_report_plots.py
# Generates comparison charts for the final report:
#   1. Headline accuracy comparison (all models x all datasets)
#   2. Hashtag leakage drop bar chart
#   3. RoBERTa training history (val F1 curves)
#
# Output: outputs/plots/report_*.png
#
# Run: python Code/aggregation/generate_report_plots.py
# ============================================================

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data hard-coded from the aggregation output (already verified)
ACC = {
    "Logistic Regression": {"SARC": 0.7238, "Twitter (with hashtags)": 0.9898, "Twitter (no hashtags)": 0.8218},
    "Linear SVM":          {"SARC": 0.7230, "Twitter (with hashtags)": 0.9910, "Twitter (no hashtags)": 0.8187},
    "Naive Bayes":         {"SARC": 0.6873, "Twitter (with hashtags)": 0.9250, "Twitter (no hashtags)": 0.7993},
    "BiLSTM (GloVe)":      {"SARC": 0.7514, "Twitter (with hashtags)": 0.9917, "Twitter (no hashtags)": 0.8296},
    "DistilBERT":          {"SARC": 0.7760, "Twitter (with hashtags)": 0.9921, "Twitter (no hashtags)": 0.8625},
    "RoBERTa-base":        {"SARC": 0.7987, "Twitter (with hashtags)": None,    "Twitter (no hashtags)": 0.8722},
}


def fig1_headline_comparison():
    models = list(ACC.keys())
    datasets = ["SARC", "Twitter (with hashtags)", "Twitter (no hashtags)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(len(models))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (ds, color) in enumerate(zip(datasets, colors)):
        vals = [ACC[m][ds] if ACC[m][ds] is not None else 0 for m in models]
        positions = x + (i - 1) * width
        bars = ax.bar(positions, vals, width, label=ds, color=color, edgecolor="black", linewidth=0.5)
        for bar, v, has_data in zip(bars, vals, [ACC[m][ds] is not None for m in models]):
            if has_data:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "n/a",
                        ha="center", va="bottom", fontsize=8, color="gray")

    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy across all model families and datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0.6, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "report_fig1_headline_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


def fig2_hashtag_leakage():
    models = ["Logistic Regression", "Linear SVM", "Naive Bayes", "BiLSTM (GloVe)", "DistilBERT"]
    with_h = [ACC[m]["Twitter (with hashtags)"] for m in models]
    no_h = [ACC[m]["Twitter (no hashtags)"] for m in models]
    drops = [(w - n) * 100 for w, n in zip(with_h, no_h)]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))

    b1 = ax.bar(x - width / 2, with_h, width, label="With hashtags",
                color="#C44E52", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, no_h, width, label="No hashtags",
                color="#4C72B0", edgecolor="black", linewidth=0.5)

    for bar, v in zip(b1, with_h):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    for bar, v in zip(b2, no_h):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)

    for xi, w, n, d in zip(x, with_h, no_h, drops):
        midpoint = (w + n) / 2
        ax.annotate(f"-{d:.1f} pp",
                    xy=(xi, midpoint),
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#333",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#666", alpha=0.95))

    ax.set_ylabel("Test accuracy on Twitter")
    ax.set_title("Hashtag leakage on Twitter: accuracy drop when all hashtags are removed")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "report_fig2_hashtag_leakage.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


def fig3_roberta_history():
    sarc_path = os.path.join(RESULTS_DIR, "roberta_sarc_history.json")
    twitter_path = os.path.join(RESULTS_DIR, "roberta_twitter_no_hashtags_history.json")

    with open(sarc_path) as f:
        sarc_hist = json.load(f)
    with open(twitter_path) as f:
        twitter_hist = json.load(f)

    sarc_epochs = [r["epoch"] for r in sarc_hist]
    sarc_train = [r["train_acc"] for r in sarc_hist]
    sarc_val = [r["val_f1_macro"] for r in sarc_hist]

    twitter_epochs = [r["epoch"] for r in twitter_hist]
    twitter_train = [r["train_acc"] for r in twitter_hist]
    twitter_val = [r["val_f1_macro"] for r in twitter_hist]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    axes[0].plot(sarc_epochs, sarc_train, marker="o", label="Train accuracy", color="#4C72B0", linewidth=2)
    axes[0].plot(sarc_epochs, sarc_val, marker="s", label="Val F1 (macro)", color="#C44E52", linewidth=2)
    axes[0].set_xticks(sarc_epochs)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].set_title("RoBERTa on SARC (with parent context)")
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="lower right")
    best_idx = sarc_val.index(max(sarc_val))
    axes[0].annotate(f"Best: epoch {sarc_epochs[best_idx]}\nVal F1 {sarc_val[best_idx]:.4f}",
                     xy=(sarc_epochs[best_idx], sarc_val[best_idx]),
                     xytext=(sarc_epochs[best_idx] - 0.5, sarc_val[best_idx] - 0.05),
                     fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#666"))

    axes[1].plot(twitter_epochs, twitter_train, marker="o", label="Train accuracy", color="#4C72B0", linewidth=2)
    axes[1].plot(twitter_epochs, twitter_val, marker="s", label="Val F1 (macro)", color="#C44E52", linewidth=2)
    axes[1].set_xticks(twitter_epochs)
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("RoBERTa on Twitter (no hashtags)")
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="lower right")
    best_idx = twitter_val.index(max(twitter_val))
    axes[1].annotate(f"Best: epoch {twitter_epochs[best_idx]}\nVal F1 {twitter_val[best_idx]:.4f}\n(early stop at epoch {twitter_epochs[-1]})",
                     xy=(twitter_epochs[best_idx], twitter_val[best_idx]),
                     xytext=(twitter_epochs[best_idx] - 0.4, twitter_val[best_idx] - 0.08),
                     fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#666"))

    plt.suptitle("RoBERTa training dynamics: validation F1 plateau marks the right stopping point", fontsize=11)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "report_fig3_roberta_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


def main():
    fig1_headline_comparison()
    fig2_hashtag_leakage()
    fig3_roberta_history()
    print("\n[done] All report plots generated in", PLOTS_DIR)


if __name__ == "__main__":
    main()
