# ============================================================
# train_lstm.py
# Trains the BiLSTM sarcasm classifier on SARC and Twitter.
#
# Pipeline (per dataset):
#   1. Load + clean + tokenize + split data
#   2. Build vocab from training tokens
#   3. Load GloVe and build embedding matrix
#   4. Convert texts to padded sequences -> PyTorch DataLoaders
#   5. Train for N epochs, tracking val F1; save best model
#   6. Evaluate best model on test set; save metrics + confusion matrix
#
# Run with:
#   python Code/training/train_lstm.py
# ============================================================

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project Code/ to path so imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

from utils.seed import set_seed
from utils.config import load_config
from utils.logger import get_logger

from data.loader import load_sarc, load_twitter
from data.preprocessor import clean_dataframe, split_data

from features.glove_embeddings import (
    build_vocab, load_glove, build_embedding_matrix, texts_to_sequences,
    PAD_IDX,
)

from models.lstm import build_lstm

from evaluation.metrics import (
    compute_metrics, save_metrics, plot_confusion_matrix, print_metrics, get_classification_report
)


# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloader(sequences: np.ndarray, labels: np.ndarray,
                    batch_size: int, shuffle: bool) -> DataLoader:
    # Wrap numpy arrays into a PyTorch TensorDataset for batching
    ds = TensorDataset(
        torch.from_numpy(sequences).long(),
        torch.from_numpy(labels).long(),
    )
    # num_workers=2 parallelizes batch prep across CPU cores
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def train_one_epoch(model, loader, optimizer, criterion, logger, epoch_num):
    # Set model to training mode (enables dropout)
    model.train()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        # Standard training step: zero grads, forward, loss, backward, step
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # Gradient clipping helps prevent exploding gradients in RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track running loss + accuracy
        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        # Log progress every ~5% of an epoch
        if batch_idx > 0 and batch_idx % max(1, len(loader) // 20) == 0:
            running_acc = correct / total
            running_loss = total_loss / n_batches
            logger.info(f"  Epoch {epoch_num} | batch {batch_idx}/{len(loader)} | "
                        f"loss={running_loss:.4f} | acc={running_acc:.4f}")

    return total_loss / n_batches, correct / total


@torch.no_grad()
def evaluate(model, loader):
    # Set model to eval mode (disables dropout, freezes batchnorm if any)
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / n_batches
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, np.array(all_labels), np.array(all_preds)


def run_dataset(dataset_name: str, df: pd.DataFrame, glove_dict: dict,
                config: dict, logger):
    # End-to-end pipeline for one dataset
    logger.info(f"\n{'='*70}\n  Running LSTM pipeline for: {dataset_name}\n{'='*70}")

    # 1. Clean + tokenize + split
    logger.info(f"[{dataset_name}] Cleaning and tokenizing...")
    df = clean_dataframe(df, config, add_tokens=True)
    train_df, val_df, test_df = split_data(
        df,
        val_split=config["data"]["val_split"],
        test_split=0.1,
        seed=config["seed"],
    )

    # 2. Build vocab from training tokens only (no peeking at val/test)
    vocab = build_vocab(train_df["tokens"].tolist(),
                        max_vocab_size=config["lstm"]["max_vocab_size"])

    # 3. Build embedding matrix (GloVe-initialized, random for OOV)
    emb_matrix = build_embedding_matrix(
        vocab, glove_dict,
        embedding_dim=config["lstm"]["embedding_dim"],
        seed=config["seed"],
    )

    # 4. Convert tokens -> integer sequences
    max_len = config["lstm"]["max_seq_length"]
    X_train = texts_to_sequences(train_df["tokens"].tolist(), vocab, max_seq_length=max_len)
    X_val = texts_to_sequences(val_df["tokens"].tolist(), vocab, max_seq_length=max_len)
    X_test = texts_to_sequences(test_df["tokens"].tolist(), vocab, max_seq_length=max_len)
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # 5. PyTorch DataLoaders
    batch_size = config["lstm"]["batch_size"]
    train_loader = make_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, batch_size, shuffle=False)

    # 6. Build model
    model = build_lstm(config, vocab_size=len(vocab), pretrained_embeddings=emb_matrix)
    model = model.to(DEVICE)
    logger.info(f"[{dataset_name}] Model on {DEVICE}, "
                f"params={sum(p.numel() for p in model.parameters()):,}")

    # 7. Optimizer + loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lstm"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # 8. Training loop with best-model tracking on val F1
    epochs = config["lstm"]["epochs"]
    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{epochs} ({dataset_name}) ---")
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, logger, epoch)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader)

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} done in {epoch_time:.1f}s | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
                    f"val_f1={val_metrics['f1_macro']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "epoch_time_seconds": epoch_time,
        })

        # Track best model by val F1 (more robust than accuracy on imbalanced sets)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  ** New best val F1 = {best_val_f1:.4f} (saving) **")

    # 9. Restore best model and evaluate on test
    logger.info(f"\n[{dataset_name}] Restoring best model (val F1={best_val_f1:.4f})")
    model.load_state_dict(best_state)
    test_loss, test_metrics, y_test_true, y_test_pred = evaluate(model, test_loader)

    test_metrics["model"] = "lstm"
    test_metrics["dataset"] = dataset_name
    test_metrics["split"] = "test"
    test_metrics["best_val_f1_macro"] = best_val_f1

    print_metrics(test_metrics, name=f"{dataset_name} | LSTM | TEST")

    # 10. Save everything
    results_dir = config["paths"]["results"]
    plots_dir = config["paths"]["plots"]
    models_dir = config["paths"]["saved_models"]
    os.makedirs(models_dir, exist_ok=True)

    # Metrics + confusion matrix
    save_metrics(test_metrics, save_dir=results_dir, name=f"lstm_{dataset_name}_test")
    plot_confusion_matrix(y_test_true, y_test_pred, save_dir=plots_dir,
                          name=f"lstm_{dataset_name}_test")

    # Training history (loss curves come from this)
    history_path = os.path.join(results_dir, f"lstm_{dataset_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"[{dataset_name}] Saved training history to {history_path}")

    # Classification report
    report_path = os.path.join(results_dir, f"lstm_{dataset_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: LSTM\nDataset: {dataset_name}\nBest val F1: {best_val_f1:.4f}\n\n")
        f.write("--- TEST ---\n")
        f.write(get_classification_report(y_test_true, y_test_pred))

    # Save model weights + vocab (so we can reload for inference / interpretability)
    model_path = os.path.join(models_dir, f"lstm_{dataset_name}.pt")
    torch.save(model.state_dict(), model_path)
    vocab_path = os.path.join(models_dir, f"lstm_{dataset_name}_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logger.info(f"[{dataset_name}] Saved model to {model_path}")
    logger.info(f"[{dataset_name}] Saved vocab to {vocab_path}")

    return test_metrics


def main():
    config = load_config()
    set_seed(config["seed"])
    logger = get_logger("train_lstm", log_dir=config["paths"]["logs"])
    logger.info(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    overall_start = time.time()

    # Load GloVe ONCE (it's big - reuse across both datasets)
    glove_dict = load_glove(config["paths"]["glove"],
                            embedding_dim=config["lstm"]["embedding_dim"])

    all_results = {}

    # ---- SARC ----
    logger.info("\nLoading SARC dataset...")
    sarc_df = load_sarc(
        sarc_dir=config["paths"]["sarc_dir"],
        sample_size=config["data"].get("sarc_train_sample"),
        seed=config["seed"],
    )
    all_results["sarc"] = run_dataset("sarc", sarc_df, glove_dict, config, logger)

    # ---- Twitter ----
    logger.info("\nLoading Twitter dataset (train + test combined)...")
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
    all_results["twitter"] = run_dataset("twitter", twitter_df, glove_dict, config, logger)

    # ---- Final summary ----
    logger.info(f"\n{'='*70}\n  FINAL LSTM SUMMARY (test set)\n{'='*70}")
    logger.info(f"{'Dataset':<10} {'Accuracy':<10} {'F1 (macro)':<12} {'Best val F1':<12}")
    logger.info("-" * 70)
    for ds_name, m in all_results.items():
        logger.info(f"{ds_name:<10} {m['accuracy']:.4f}     {m['f1_macro']:.4f}       "
                    f"{m['best_val_f1_macro']:.4f}")

    total_time = time.time() - overall_start
    logger.info(f"\nTotal LSTM training time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
