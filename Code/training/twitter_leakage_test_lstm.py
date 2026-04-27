# ============================================================
# twitter_leakage_test_lstm.py
# Re-runs the LSTM on Twitter with ALL hashtags stripped, to get
# an honest Twitter accuracy number (matching what we did for the
# classical models in twitter_leakage_test.py).
#
# The original Twitter LSTM run (~99.2%) was inflated by hashtag
# label leakage. The classical leakage test showed accuracy drops
# of ~17 points once #sarcasm/#irony/etc. are removed. We re-run
# the LSTM under the same fair conditions here.
#
# Run with:
#   python Code/training/twitter_leakage_test_lstm.py
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

from utils.seed import set_seed
from utils.config import load_config
from utils.logger import get_logger

from data.loader import load_twitter
from data.preprocessor import clean_dataframe, split_data

from features.glove_embeddings import (
    build_vocab, load_glove, build_embedding_matrix, texts_to_sequences,
    PAD_IDX,
)
from models.lstm import build_lstm

from evaluation.metrics import (
    compute_metrics, save_metrics, plot_confusion_matrix,
    print_metrics, get_classification_report,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloader(sequences: np.ndarray, labels: np.ndarray,
                    batch_size: int, shuffle: bool) -> DataLoader:
    # Wrap numpy arrays into a PyTorch TensorDataset for batching
    ds = TensorDataset(
        torch.from_numpy(sequences).long(),
        torch.from_numpy(labels).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def train_one_epoch(model, loader, optimizer, criterion, logger, epoch_num):
    # Training mode (enables dropout)
    model.train()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        # Twitter is small - log every ~10% of an epoch
        if batch_idx > 0 and batch_idx % max(1, len(loader) // 10) == 0:
            logger.info(f"  Epoch {epoch_num} | batch {batch_idx}/{len(loader)} | "
                        f"loss={total_loss/n_batches:.4f} | acc={correct/total:.4f}")

    return total_loss / n_batches, correct / total


@torch.no_grad()
def evaluate(model, loader):
    # Eval mode (disables dropout)
    model.eval()
    all_preds, all_labels = [], []
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

    return total_loss / n_batches, compute_metrics(all_labels, all_preds), \
           np.array(all_labels), np.array(all_preds)


def main():
    config = load_config()
    set_seed(config["seed"])
    logger = get_logger("twitter_leakage_test_lstm", log_dir=config["paths"]["logs"])
    logger.info(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Override hashtag stripping in memory (don't modify config.yaml)
    config["preprocessing"]["remove_all_hashtags"] = True
    logger.info("Override: remove_all_hashtags = True (testing leakage hypothesis on LSTM)")

    overall_start = time.time()

    # 1. Load Twitter (combine train + test, we'll re-split using our own seed)
    logger.info("\nLoading Twitter dataset...")
    tw_train = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="train")
    tw_test = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="test")
    df = pd.concat([tw_train, tw_test], ignore_index=True)

    # 2. Clean (with ALL hashtags stripped) and split
    logger.info("Cleaning tweets with ALL hashtags removed...")
    df = clean_dataframe(df, config, add_tokens=True)
    train_df, val_df, test_df = split_data(
        df, val_split=config["data"]["val_split"],
        test_split=0.1, seed=config["seed"],
    )

    # 3. Build vocab from cleaned training tokens
    vocab = build_vocab(train_df["tokens"].tolist(),
                        max_vocab_size=config["lstm"]["max_vocab_size"])

    # 4. Load GloVe + build embedding matrix
    glove_dict = load_glove(config["paths"]["glove"],
                            embedding_dim=config["lstm"]["embedding_dim"])
    emb_matrix = build_embedding_matrix(
        vocab, glove_dict,
        embedding_dim=config["lstm"]["embedding_dim"],
        seed=config["seed"],
    )

    # 5. Token sequences
    max_len = config["lstm"]["max_seq_length"]
    X_train = texts_to_sequences(train_df["tokens"].tolist(), vocab, max_seq_length=max_len)
    X_val = texts_to_sequences(val_df["tokens"].tolist(), vocab, max_seq_length=max_len)
    X_test = texts_to_sequences(test_df["tokens"].tolist(), vocab, max_seq_length=max_len)
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # 6. DataLoaders
    batch_size = config["lstm"]["batch_size"]
    train_loader = make_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, batch_size, shuffle=False)

    # 7. Build model
    model = build_lstm(config, vocab_size=len(vocab), pretrained_embeddings=emb_matrix)
    model = model.to(DEVICE)
    logger.info(f"[twitter_no_hashtags] Model on {DEVICE}, "
                f"params={sum(p.numel() for p in model.parameters()):,}")

    # 8. Optimizer + loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lstm"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # 9. Training loop with best-model tracking on val F1
    epochs = config["lstm"]["epochs"]
    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{epochs} (twitter_no_hashtags) ---")
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

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  ** New best val F1 = {best_val_f1:.4f} (saving) **")

    # 10. Restore best + evaluate on test
    logger.info(f"\n[twitter_no_hashtags] Restoring best model (val F1={best_val_f1:.4f})")
    model.load_state_dict(best_state)
    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader)

    test_metrics["model"] = "lstm"
    test_metrics["dataset"] = "twitter_no_hashtags"
    test_metrics["split"] = "test"
    test_metrics["best_val_f1_macro"] = best_val_f1

    print_metrics(test_metrics, name="twitter_no_hashtags | LSTM | TEST")

    # 11. Save everything (note dataset name: twitter_no_hashtags)
    results_dir = config["paths"]["results"]
    plots_dir = config["paths"]["plots"]
    models_dir = config["paths"]["saved_models"]
    os.makedirs(models_dir, exist_ok=True)

    save_metrics(test_metrics, save_dir=results_dir, name="lstm_twitter_no_hashtags_test")
    plot_confusion_matrix(y_true, y_pred, save_dir=plots_dir, name="lstm_twitter_no_hashtags_test")

    history_path = os.path.join(results_dir, "lstm_twitter_no_hashtags_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    report_path = os.path.join(results_dir, "lstm_twitter_no_hashtags_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: LSTM\nDataset: twitter_no_hashtags\nBest val F1: {best_val_f1:.4f}\n\n")
        f.write("--- TEST ---\n")
        f.write(get_classification_report(y_true, y_pred))

    model_path = os.path.join(models_dir, "lstm_twitter_no_hashtags.pt")
    torch.save(model.state_dict(), model_path)
    vocab_path = os.path.join(models_dir, "lstm_twitter_no_hashtags_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved vocab to {vocab_path}")

    # 12. Comparison vs original (with-hashtag) Twitter LSTM
    original_acc = 0.9917  # from earlier full Twitter LSTM run
    new_acc = test_metrics["accuracy"]
    drop = original_acc - new_acc
    logger.info(f"\n{'='*70}\n  TWITTER LSTM LEAKAGE COMPARISON (test accuracy)\n{'='*70}")
    logger.info(f"{'Model':<22} {'With hashtags':<16} {'No hashtags':<14} {'Drop':<10}")
    logger.info("-" * 70)
    logger.info(f"{'lstm':<22} {original_acc:.4f}           {new_acc:.4f}         {drop:+.4f}")

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
