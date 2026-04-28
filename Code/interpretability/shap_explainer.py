# ============================================================
# shap_explainer.py
# SHAP explanations for sarcasm predictions.
#
# Two outputs per (model, dataset):
#   1. Local explanations on 20 examples (5 per category, same as LIME)
#   2. Global summary: top words pushing toward sarcasm vs not, aggregated
#
# Outputs to outputs/interpretability/:
#   shap_<model>_<dataset>_summary_plot.png       (global feature importance)
#   shap_<model>_<dataset>_local_<i>_<category>.png (one per local example)
#   shap_<model>_<dataset>_summary.json
#
# Run:
#   python Code/interpretability/shap_explainer.py --model logistic_regression --dataset sarc
#   python Code/interpretability/shap_explainer.py --model distilbert --dataset twitter_no_hashtags
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

from utils.seed import set_seed
from utils.config import load_config
from utils.logger import get_logger

# Reuse the predictor + example-selection code from lime_explainer
from interpretability.lime_explainer import (
    get_predictor, load_test_data, select_examples, CLASS_NAMES,
)


def run_shap(predict_proba, examples, model_name, dataset_name, config, output_dir, logger):
    # SHAP's masker handles tokenization for text models.
    # We use a regex masker that splits on whitespace - works uniformly across all models.
    masker = shap.maskers.Text(r"\W+")

    # Wrap predict_proba so SHAP gets only the sarcasm-class probability
    def f(texts):
        return predict_proba(list(texts))[:, 1]

    explainer = shap.Explainer(f, masker, output_names=["Sarcasm"])

    # Local explanations on the picked examples
    texts = [ex["text"] for ex in examples]
    n_samples = config["interpretability"].get("shap_num_samples", 50)
    logger.info(f"[shap] Computing SHAP values on {len(texts)} examples (n_samples={n_samples})...")
    shap_values = explainer(texts, max_evals=n_samples, silent=True)

    # Save one local plot per example
    for i, ex in enumerate(examples):
        try:
            plt.figure(figsize=(10, 3))
            shap.plots.text(shap_values[i], display=False)
            local_path = os.path.join(
                output_dir,
                f"shap_{model_name}_{dataset_name}_local_{i+1:02d}_{ex['category']}.html",
            )
            # SHAP's text plot is HTML-based; save manually
            html = shap.plots.text(shap_values[i], display=False)
            with open(local_path, "w") as fh:
                fh.write(html if isinstance(html, str) else "")
            plt.close()
        except Exception as e:
            logger.warning(f"[shap] Local plot {i+1} failed: {e}")
            plt.close()

    # Aggregate per-token SHAP values into global importance counters
    sarcasm_pushers = Counter()
    not_pushers = Counter()
    for i in range(len(examples)):
        tokens = shap_values[i].data        # list of strings
        values = shap_values[i].values      # array of per-token shap values
        for tok, val in zip(tokens, values):
            tok_clean = tok.strip().lower()
            if not tok_clean or len(tok_clean) < 2:
                continue
            if val > 0:
                sarcasm_pushers[tok_clean] += float(val)
            else:
                not_pushers[tok_clean] += float(abs(val))

    # Build a horizontal bar chart of the top global pushers
    top_n = 20
    sarc_top = sarcasm_pushers.most_common(top_n)
    not_top = not_pushers.most_common(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if sarc_top:
        words, vals = zip(*sarc_top)
        axes[0].barh(range(len(words)), vals, color="#d62728")
        axes[0].set_yticks(range(len(words)))
        axes[0].set_yticklabels(words)
        axes[0].invert_yaxis()
        axes[0].set_title(f"Top words pushing toward Sarcasm\n({model_name}, {dataset_name})")
        axes[0].set_xlabel("Cumulative SHAP value")
    if not_top:
        words, vals = zip(*not_top)
        axes[1].barh(range(len(words)), vals, color="#1f77b4")
        axes[1].set_yticks(range(len(words)))
        axes[1].set_yticklabels(words)
        axes[1].invert_yaxis()
        axes[1].set_title(f"Top words pushing toward Not-Sarcasm\n({model_name}, {dataset_name})")
        axes[1].set_xlabel("Cumulative |SHAP value|")
    plt.tight_layout()
    summary_plot_path = os.path.join(
        output_dir, f"shap_{model_name}_{dataset_name}_summary_plot.png"
    )
    plt.savefig(summary_plot_path, dpi=120)
    plt.close()
    logger.info(f"[shap] Saved summary plot to {summary_plot_path}")

    # Save the JSON summary
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_explanations": len(examples),
        "shap_config": {"num_samples": n_samples},
        "top_words_pushing_toward_sarcasm": sarc_top,
        "top_words_pushing_toward_not_sarcasm": not_top,
        "examples": [
            {
                **ex,
                "tokens": list(shap_values[i].data),
                "shap_values": [float(v) for v in shap_values[i].values],
            }
            for i, ex in enumerate(examples)
        ],
    }
    summary_path = os.path.join(output_dir, f"shap_{model_name}_{dataset_name}_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"[shap] Saved summary JSON to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["logistic_regression", "linear_svm", "naive_bayes",
                                 "lstm", "distilbert", "roberta"])
    parser.add_argument("--dataset", required=True,
                        choices=["sarc", "twitter", "twitter_no_hashtags"])
    parser.add_argument("--n_per_category", type=int, default=5)
    parser.add_argument("--max_predict", type=int, default=5000)
    args = parser.parse_args()

    config = load_config()
    set_seed(config["seed"])
    logger = get_logger(f"shap_{args.model}_{args.dataset}",
                        log_dir=config["paths"]["logs"])
    logger.info(f"[shap] Model: {args.model} | Dataset: {args.dataset}")

    output_dir = os.path.join(PROJECT_ROOT, "outputs", "interpretability")
    os.makedirs(output_dir, exist_ok=True)

    test_df = load_test_data(args.dataset, config, logger)
    predict_proba = get_predictor(args.model, args.dataset, config)
    examples = select_examples(test_df, predict_proba, args.n_per_category,
                                args.max_predict, logger)
    run_shap(predict_proba, examples, args.model, args.dataset, config, output_dir, logger)
    logger.info("[shap] Done.")


if __name__ == "__main__":
    main()
