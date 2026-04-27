# ============================================================
# train_classical_with_contrast.py
# Trains classical models on TF-IDF features + sentiment-contrast features.
#
# This tests whether the proposed "contextual contrast" signal adds
# predictive value over plain TF-IDF. We compare side-by-side against
# the previously trained TF-IDF-only classical models.
#
# Run with:
#   python Code/training/train_classical_with_contrast.py
# ============================================================

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

from utils.seed import set_seed
from utils.config import load_config
from utils.logger import get_logger

from data.loader import load_sarc, load_twitter
from data.preprocessor import clean_dataframe, split_data

from features.classical_features import fit_features, transform_features
from features.contrast_features import ContrastFeaturizer, FEATURE_NAMES

from models.classical import get_model

from evaluation.metrics import (
    compute_metrics, save_metrics, plot_confusion_matrix,
    print_metrics, get_classification_report,
)


def add_contrast_features(X_tfidf, texts: list, featurizer: ContrastFeaturizer):
    # Compute contrast features for each text and horizontally stack with TF-IDF.
    # Naive Bayes requires non-negative features, so we'll handle that downstream.
    contrast_matrix = featurizer.transform(texts)  # shape (n, 11)
    # Convert to sparse so we can hstack with TF-IDF (TF-IDF is a sparse matrix)
    contrast_sparse = csr_matrix(contrast_matrix)
    return hstack([X_tfidf, contrast_sparse]).tocsr()


def add_contrast_features_nonneg(X_tfidf, texts: list, featurizer: ContrastFeaturizer):
    # Naive Bayes can't accept negative features (compound is in [-1, 1], strongest_neg is negative).
    # Shift these features so they're non-negative for NB only.
    contrast_matrix = featurizer.transform(texts)
    # Shift columns that can be negative: compound (col 2), strongest_neg (col 7)
    contrast_matrix[:, 2] = contrast_matrix[:, 2] + 1.0  # compound: [-1,1] -> [0,2]
    contrast_matrix[:, 7] = contrast_matrix[:, 7] + 4.0  # strongest_neg: ~[-4,0] -> [0,4]
    contrast_sparse = csr_matrix(contrast_matrix)
    return hstack([X_tfidf, contrast_sparse]).tocsr()


def run_dataset(dataset_name: str, df, config, logger, featurizer: ContrastFeaturizer):
    logger.info(f"\n{'='*70}\n  Running classical+contrast pipeline for: {dataset_name}\n{'='*70}")

    # Standard cleaning + split
    logger.info(f"[{dataset_name}] Cleaning text...")
    df = clean_dataframe(df, config, add_tokens=False)
    train, val, test = split_data(df, val_split=config["data"]["val_split"],
                                   test_split=0.1, seed=config["seed"])

    # TF-IDF features (same as before)
    logger.info(f"[{dataset_name}] Fitting TF-IDF features...")
    X_train_tfidf, word_vec, char_vec = fit_features(train["text"].tolist(), config)
    X_val_tfidf = transform_features(val["text"].tolist(), word_vec, char_vec)
    X_test_tfidf = transform_features(test["text"].tolist(), word_vec, char_vec)

    # Contrast features (compute on the SAME texts)
    logger.info(f"[{dataset_name}] Computing contrast features...")
    t0 = time.time()
    X_train_full = add_contrast_features(X_train_tfidf, train["text"].tolist(), featurizer)
    X_val_full = add_contrast_features(X_val_tfidf, val["text"].tolist(), featurizer)
    X_test_full = add_contrast_features(X_test_tfidf, test["text"].tolist(), featurizer)
    logger.info(f"[{dataset_name}] Contrast features computed in {time.time()-t0:.1f}s")
    logger.info(f"[{dataset_name}] Combined feature shape: {X_train_full.shape}")

    # NB-specific non-negative version
    X_train_full_nb = add_contrast_features_nonneg(X_train_tfidf, train["text"].tolist(), featurizer)
    X_val_full_nb = add_contrast_features_nonneg(X_val_tfidf, val["text"].tolist(), featurizer)
    X_test_full_nb = add_contrast_features_nonneg(X_test_tfidf, test["text"].tolist(), featurizer)

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    results = {}
    for model_name in config["classical"]["models"]:
        # NB needs non-negative features, others use the regular version
        if model_name == "naive_bayes":
            Xtr, Xte = X_train_full_nb, X_test_full_nb
        else:
            Xtr, Xte = X_train_full, X_test_full

        logger.info(f"\n[{dataset_name}/{model_name}+contrast] Training...")
        start = time.time()
        model = get_model(model_name, seed=config["seed"])
        model.fit(Xtr, y_train)
        fit_time = time.time() - start
        logger.info(f"[{dataset_name}/{model_name}+contrast] Trained in {fit_time:.1f}s")

        y_pred = model.predict(Xte)
        metrics = compute_metrics(y_test, y_pred)
        metrics["model"] = f"{model_name}_with_contrast"
        metrics["dataset"] = dataset_name
        metrics["split"] = "test"
        metrics["fit_time_seconds"] = fit_time

        print_metrics(metrics, name=f"{dataset_name} | {model_name}+contrast | TEST")
        save_metrics(metrics, save_dir=config["paths"]["results"],
                     name=f"classical_{dataset_name}_{model_name}_with_contrast_test")
        plot_confusion_matrix(y_test, y_pred, save_dir=config["paths"]["plots"],
                              name=f"classical_{dataset_name}_{model_name}_with_contrast_test")

        results[model_name] = metrics["accuracy"]

    return results


def main():
    config = load_config()
    set_seed(config["seed"])
    logger = get_logger("train_classical_contrast", log_dir=config["paths"]["logs"])

    overall_start = time.time()
    featurizer = ContrastFeaturizer()

    # SARC
    logger.info("Loading SARC...")
    sarc_df = load_sarc(
        sarc_dir=config["paths"]["sarc_dir"],
        sample_size=config["data"].get("sarc_train_sample"),
        seed=config["seed"],
    )
    sarc_results = run_dataset("sarc", sarc_df, config, logger, featurizer)

    # Twitter
    logger.info("\nLoading Twitter...")
    tw_train = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="train")
    tw_test = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="test")
    twitter_df = pd.concat([tw_train, tw_test], ignore_index=True)
    twitter_results = run_dataset("twitter", twitter_df, config, logger, featurizer)

    # Comparison vs TF-IDF only baseline (numbers from previous run)
    baseline = {
        "sarc": {"logistic_regression": 0.7238, "linear_svm": 0.7230, "naive_bayes": 0.6873},
        "twitter": {"logistic_regression": 0.9898, "linear_svm": 0.9910, "naive_bayes": 0.9250},
    }

    logger.info(f"\n{'='*70}\n  CONTRAST FEATURES ABLATION (test accuracy)\n{'='*70}")
    logger.info(f"{'Dataset':<10} {'Model':<22} {'TF-IDF only':<14} {'+ contrast':<14} {'Delta':<10}")
    logger.info("-" * 70)
    for ds_name, res in [("sarc", sarc_results), ("twitter", twitter_results)]:
        for model_name, new_acc in res.items():
            base = baseline[ds_name][model_name]
            delta = new_acc - base
            logger.info(f"{ds_name:<10} {model_name:<22} {base:.4f}         {new_acc:.4f}         {delta:+.4f}")

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
