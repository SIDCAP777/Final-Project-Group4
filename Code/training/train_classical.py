# ============================================================
# train_classical.py
# Trains and evaluates all classical models on both datasets.
#
# Pipeline (per dataset):
#   1. Load + clean + split data
#   2. Fit TF-IDF features on train, transform val/test
#   3. For each model (LR, SVM, NB):
#        - fit on train
#        - evaluate on val and test
#        - save metrics, confusion matrix, and the trained model
#
# Run with:
#   python Code/training/train_classical.py
# ============================================================

import os
import sys
import time
import pickle

# Add project root to Python path so we can import Code.* modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

from utils.seed import set_seed
from utils.config import load_config
from utils.logger import get_logger

from data.loader import load_sarc, load_twitter
from data.preprocessor import clean_dataframe, split_data

from features.classical_features import fit_features, transform_features, save_vectorizers

from models.classical import get_model

from evaluation.metrics import (
    compute_metrics,
    save_metrics,
    plot_confusion_matrix,
    print_metrics,
    get_classification_report,
)


def prepare_dataset(name: str, df, config: dict, logger):
    # Clean the data, then split into train/val/test
    logger.info(f"[{name}] Cleaning text...")
    df = clean_dataframe(df, config, add_tokens=False)

    logger.info(f"[{name}] Splitting train/val/test...")
    train, val, test = split_data(
        df,
        val_split=config["data"]["val_split"],
        test_split=0.1,
        seed=config["seed"],
    )
    return train, val, test


def train_one_model(model_name: str, dataset_name: str,
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    config: dict, logger):
    # Build a fresh model
    logger.info(f"[{dataset_name}/{model_name}] Building model...")
    model = get_model(model_name, seed=config["seed"])

    # Fit on training data
    logger.info(f"[{dataset_name}/{model_name}] Training...")
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start
    logger.info(f"[{dataset_name}/{model_name}] Trained in {fit_time:.1f}s")

    # Predict on val and test
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Compute metrics for both sets
    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    # Add metadata so the JSON files are self-describing later
    val_metrics["model"] = model_name
    val_metrics["dataset"] = dataset_name
    val_metrics["split"] = "val"
    val_metrics["fit_time_seconds"] = fit_time

    test_metrics["model"] = model_name
    test_metrics["dataset"] = dataset_name
    test_metrics["split"] = "test"
    test_metrics["fit_time_seconds"] = fit_time

    # Pretty console output
    print_metrics(val_metrics, name=f"{dataset_name} | {model_name} | VAL")
    print_metrics(test_metrics, name=f"{dataset_name} | {model_name} | TEST")

    # Save metrics JSONs
    results_dir = config["paths"]["results"]
    save_metrics(val_metrics, save_dir=results_dir, name=f"classical_{dataset_name}_{model_name}_val")
    save_metrics(test_metrics, save_dir=results_dir, name=f"classical_{dataset_name}_{model_name}_test")

    # Save confusion matrix plots (test set only - that's what we report)
    plots_dir = config["paths"]["plots"]
    plot_confusion_matrix(
        y_test, y_test_pred,
        save_dir=plots_dir,
        name=f"classical_{dataset_name}_{model_name}_test",
    )

    # Save the trained model itself
    models_dir = config["paths"]["saved_models"]
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"classical_{dataset_name}_{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"[{dataset_name}/{model_name}] Saved model to {model_path}")

    # Also save a full classification report as a text file (handy for the report write-up)
    report_path = os.path.join(results_dir, f"classical_{dataset_name}_{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Fit time: {fit_time:.1f}s\n")
        f.write("\n--- VAL ---\n")
        f.write(get_classification_report(y_val, y_val_pred))
        f.write("\n--- TEST ---\n")
        f.write(get_classification_report(y_test, y_test_pred))

    return {"val": val_metrics, "test": test_metrics}


def run_dataset(dataset_name: str, df, config: dict, logger):
    # Run the full pipeline for one dataset across all 3 classical models
    logger.info(f"\n{'='*70}\n  Running classical pipeline for: {dataset_name}\n{'='*70}")

    # 1. Clean + split
    train, val, test = prepare_dataset(dataset_name, df, config, logger)

    # 2. Fit features on train only, transform val/test using the same vectorizers
    logger.info(f"[{dataset_name}] Fitting TF-IDF features...")
    X_train, word_vec, char_vec = fit_features(train["text"].tolist(), config)
    X_val = transform_features(val["text"].tolist(), word_vec, char_vec)
    X_test = transform_features(test["text"].tolist(), word_vec, char_vec)

    # Save the vectorizers (we'll need them later for inference + LIME/SHAP)
    save_vectorizers(
        word_vec, char_vec,
        save_dir=config["paths"]["saved_models"],
        tag=f"classical_{dataset_name}",
    )

    # 3. Pull labels as numpy arrays
    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    # 4. Train each model and collect its results
    all_results = {}
    for model_name in config["classical"]["models"]:
        results = train_one_model(
            model_name, dataset_name,
            X_train, y_train, X_val, y_val, X_test, y_test,
            config, logger,
        )
        all_results[model_name] = results

    return all_results


def main():
    # Load config + set seeds + start logger
    config = load_config()
    set_seed(config["seed"])
    logger = get_logger("train_classical", log_dir=config["paths"]["logs"])

    overall_start = time.time()

    # Storage for the final summary table
    all_results = {}

    # ---- SARC ----
    logger.info("Loading SARC dataset...")
    sarc_df = load_sarc(
        sarc_dir=config["paths"]["sarc_dir"],
        sample_size=config["data"].get("sarc_train_sample"),
        seed=config["seed"],
    )
    all_results["sarc"] = run_dataset("sarc", sarc_df, config, logger)

    # ---- Twitter ----
    logger.info("Loading Twitter dataset (combining train + test)...")
    # Twitter has predefined train/test, but we re-split together for our own val set
    import pandas as pd
    tw_train = load_twitter(
        twitter_dir=config["paths"]["twitter_dir"],
        split="train",
        sample_size=config["data"].get("twitter_train_sample"),
        seed=config["seed"],
    )
    tw_test = load_twitter(
        twitter_dir=config["paths"]["twitter_dir"],
        split="test",
        sample_size=config["data"].get("twitter_test_sample"),
        seed=config["seed"],
    )
    twitter_df = pd.concat([tw_train, tw_test], ignore_index=True)
    all_results["twitter"] = run_dataset("twitter", twitter_df, config, logger)

    # ---- Final summary table ----
    logger.info(f"\n{'='*70}\n  FINAL SUMMARY (test set)\n{'='*70}")
    logger.info(f"{'Dataset':<10} {'Model':<22} {'Accuracy':<10} {'F1 (macro)':<12} {'Fit time':<10}")
    logger.info("-" * 70)
    for ds_name, ds_results in all_results.items():
        for model_name, splits in ds_results.items():
            test = splits["test"]
            logger.info(
                f"{ds_name:<10} {model_name:<22} "
                f"{test['accuracy']:.4f}     {test['f1_macro']:.4f}       "
                f"{test['fit_time_seconds']:.1f}s"
            )

    total_time = time.time() - overall_start
    logger.info(f"\nTotal classical training time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
