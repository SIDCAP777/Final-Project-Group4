# ============================================================
# twitter_leakage_test_classical.py
# Tests whether Twitter's high accuracy is from hashtag leakage on classical models.
#
# Strips ALL hashtags from tweets, retrains classical models,
# and compares to the original Twitter results.
#
# Run with:
#   python Code/training/twitter_leakage_test_classical.py
# ============================================================

import os
import sys
import time
import pickle
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

from utils.seed import set_seed
from utils.config import load_config
from utils.logger import get_logger

from data.loader import load_twitter
from data.preprocessor import clean_dataframe, split_data

from features.classical_features import fit_features, transform_features, save_vectorizers

from models.classical import get_model

from evaluation.metrics import (
    compute_metrics, save_metrics, plot_confusion_matrix,
    print_metrics, get_classification_report,
)


def main():
    config = load_config()
    set_seed(config["seed"])
    logger = get_logger("twitter_leakage_test", log_dir=config["paths"]["logs"])

    # Force the aggressive hashtag stripping flag for this run only.
    # We DON'T modify config.yaml - we just override in memory.
    config["preprocessing"]["remove_all_hashtags"] = True
    logger.info("Override: remove_all_hashtags = True (testing leakage hypothesis)")

    # Load Twitter (combine train + test, we'll re-split)
    logger.info("\nLoading Twitter dataset...")
    tw_train = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="train")
    tw_test = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="test")
    df = pd.concat([tw_train, tw_test], ignore_index=True)

    # Clean (with ALL hashtags stripped) and split
    logger.info("Cleaning tweets with ALL hashtags removed...")
    df = clean_dataframe(df, config, add_tokens=False)
    train, val, test = split_data(df, val_split=config["data"]["val_split"],
                                   test_split=0.1, seed=config["seed"])

    # Fit features on cleaned train
    logger.info("Fitting TF-IDF features on cleaned training data...")
    X_train, word_vec, char_vec = fit_features(train["text"].tolist(), config)
    X_val = transform_features(val["text"].tolist(), word_vec, char_vec)
    X_test = transform_features(test["text"].tolist(), word_vec, char_vec)

    # Save the no-hashtags vectorizers so LIME / SHAP can reload them later
    save_vectorizers(
        word_vec, char_vec,
        save_dir=config["paths"]["saved_models"],
        tag="classical_twitter_no_hashtags",
    )

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    # Train each classical model and collect results
    results_summary = {}

    for model_name in config["classical"]["models"]:
        logger.info(f"\n[twitter_clean/{model_name}] Training...")
        start = time.time()
        model = get_model(model_name, seed=config["seed"])
        model.fit(X_train, y_train)
        fit_time = time.time() - start
        logger.info(f"[twitter_clean/{model_name}] Trained in {fit_time:.1f}s")

        # Save the trained classifier so LIME / SHAP can reload it later
        import pickle
        model_path = os.path.join(
            config["paths"]["saved_models"],
            f"classical_twitter_no_hashtags_{model_name}.pkl",
        )
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"[twitter_clean/{model_name}] Saved model to {model_path}")

        # Evaluate on test set
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics["model"] = model_name
        metrics["dataset"] = "twitter_no_hashtags"
        metrics["split"] = "test"
        metrics["fit_time_seconds"] = fit_time

        print_metrics(metrics, name=f"twitter_no_hashtags | {model_name} | TEST")

        # Save metrics + plots
        save_metrics(metrics, save_dir=config["paths"]["results"],
                     name=f"classical_twitter_no_hashtags_{model_name}_test")
        plot_confusion_matrix(y_test, y_pred, save_dir=config["paths"]["plots"],
                              name=f"classical_twitter_no_hashtags_{model_name}_test")

        results_summary[model_name] = metrics["accuracy"]

    # Print comparison vs original (we have those numbers from earlier runs)
    original_results = {
        "logistic_regression": 0.9898,
        "linear_svm": 0.9910,
        "naive_bayes": 0.9250,
    }

    logger.info(f"\n{'='*70}\n  TWITTER LEAKAGE COMPARISON (test accuracy)\n{'='*70}")
    logger.info(f"{'Model':<22} {'With hashtags':<16} {'No hashtags':<14} {'Drop':<10}")
    logger.info("-" * 70)
    for model_name, new_acc in results_summary.items():
        orig = original_results.get(model_name, 0.0)
        drop = orig - new_acc
        logger.info(f"{model_name:<22} {orig:.4f}           {new_acc:.4f}         {drop:+.4f}")


if __name__ == "__main__":
    main()
