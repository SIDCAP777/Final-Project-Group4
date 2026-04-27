# ============================================================
# twitter_leakage_test_distilbert.py
# Re-runs DistilBERT on Twitter with ALL hashtags stripped, to get
# an honest Twitter accuracy number (matching what we did for the
# classical models in twitter_leakage_test.py and the LSTM in
# twitter_leakage_test_lstm.py).
#
# The original Twitter DistilBERT run (~99.2%) was inflated by
# hashtag label leakage. We re-run under fair conditions here.
#
# Run with:
#   python Code/training/twitter_leakage_test_distilbert.py
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

from utils.seed import set_seed
from utils.config import load_config
from utils.logger import get_logger

from data.loader import load_twitter
from data.preprocessor import clean_dataframe, split_data
from models.distilbert import build_distilbert
from evaluation.metrics import (
    compute_metrics, save_metrics, plot_confusion_matrix,
    print_metrics, get_classification_report,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    # Tokenize all texts upfront so the training loop is fast (no per-step tokenization)
    def __init__(self, texts, labels, tokenizer, max_seq_length: int):
        encoded = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def train_one_epoch(model, loader, optimizer, scheduler, logger, epoch_num):
    # Training mode (enables dropout)
    model.train()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # update learning rate (linear warmup + decay)

        total_loss += loss.item()
        n_batches += 1
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Twitter is small - log every ~10% of an epoch
        if batch_idx > 0 and batch_idx % max(1, len(loader) // 10) == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"  Epoch {epoch_num} | batch {batch_idx}/{len(loader)} | "
                        f"loss={total_loss/n_batches:.4f} | acc={correct/total:.4f} | lr={current_lr:.2e}")

    return total_loss / n_batches, correct / total


@torch.no_grad()
def evaluate(model, loader):
    # Eval mode (disables dropout)
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1
        preds = outputs.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / n_batches, compute_metrics(all_labels, all_preds), \
           np.array(all_labels), np.array(all_preds)


def main():
    config = load_config()
    set_seed(config["seed"])
    logger = get_logger("twitter_leakage_test_distilbert", log_dir=config["paths"]["logs"])
    logger.info(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Override hashtag stripping in memory (don't modify config.yaml)
    config["preprocessing"]["remove_all_hashtags"] = True
    logger.info("Override: remove_all_hashtags = True (testing leakage hypothesis on DistilBERT)")

    overall_start = time.time()

    # 1. Load Twitter
    logger.info("\nLoading Twitter dataset...")
    tw_train = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="train")
    tw_test = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="test")
    df = pd.concat([tw_train, tw_test], ignore_index=True)

    # 2. Clean (with ALL hashtags stripped) and split
    logger.info("Cleaning tweets with ALL hashtags removed...")
    df = clean_dataframe(df, config, add_tokens=False)
    train_df, val_df, test_df = split_data(
        df, val_split=config["data"]["val_split"],
        test_split=0.1, seed=config["seed"],
    )

    # 3. Build model + tokenizer
    tokenizer, model = build_distilbert(config, num_classes=2)
    model = model.to(DEVICE)

    # 4. Tokenize
    max_len = config["distilbert"]["max_seq_length"]
    logger.info(f"[twitter_no_hashtags] Tokenizing train ({len(train_df)} rows)...")
    t0 = time.time()
    train_ds = TextDataset(train_df["text"], train_df["label"], tokenizer, max_len)
    logger.info(f"[twitter_no_hashtags] Train tokenized in {time.time()-t0:.1f}s")
    logger.info(f"[twitter_no_hashtags] Tokenizing val + test...")
    val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer, max_len)
    test_ds = TextDataset(test_df["text"], test_df["label"], tokenizer, max_len)

    # 5. DataLoaders
    batch_size = config["distilbert"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # 6. Optimizer (AdamW) + linear warmup scheduler
    epochs = config["distilbert"]["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(config["distilbert"]["warmup_ratio"] * total_steps)
    optimizer = AdamW(
        model.parameters(),
        lr=config["distilbert"]["learning_rate"],
        weight_decay=config["distilbert"]["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    logger.info(f"[twitter_no_hashtags] Training: {epochs} epochs, "
                f"batch_size={batch_size}, total_steps={total_steps}, warmup={warmup_steps}")

    # 7. Training loop with best-model tracking on val F1
    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{epochs} (twitter_no_hashtags) ---")
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, logger, epoch)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader)
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} done in {epoch_time:.1f}s ({epoch_time/60:.1f} min) | "
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

    # 8. Restore best + evaluate test
    logger.info(f"\n[twitter_no_hashtags] Restoring best model (val F1={best_val_f1:.4f})")
    model.load_state_dict(best_state)
    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader)

    test_metrics["model"] = "distilbert"
    test_metrics["dataset"] = "twitter_no_hashtags"
    test_metrics["split"] = "test"
    test_metrics["best_val_f1_macro"] = best_val_f1

    print_metrics(test_metrics, name="twitter_no_hashtags | DistilBERT | TEST")

    # 9. Save everything (note dataset name: twitter_no_hashtags)
    results_dir = config["paths"]["results"]
    plots_dir = config["paths"]["plots"]
    models_dir = config["paths"]["saved_models"]

    save_metrics(test_metrics, save_dir=results_dir, name="distilbert_twitter_no_hashtags_test")
    plot_confusion_matrix(y_true, y_pred, save_dir=plots_dir, name="distilbert_twitter_no_hashtags_test")

    history_path = os.path.join(results_dir, "distilbert_twitter_no_hashtags_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    report_path = os.path.join(results_dir, "distilbert_twitter_no_hashtags_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: DistilBERT\nDataset: twitter_no_hashtags\nBest val F1: {best_val_f1:.4f}\n\n")
        f.write("--- TEST ---\n")
        f.write(get_classification_report(y_true, y_pred))

    # Save fine-tuned model + tokenizer using HuggingFace's standard format
    model_dir = os.path.join(models_dir, "distilbert_twitter_no_hashtags")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"Saved model + tokenizer to {model_dir}")

    # 10. Comparison vs original (with-hashtag) Twitter DistilBERT
    original_acc = 0.9921  # from earlier full Twitter DistilBERT run
    new_acc = test_metrics["accuracy"]
    drop = original_acc - new_acc
    logger.info(f"\n{'='*70}\n  TWITTER DISTILBERT LEAKAGE COMPARISON (test accuracy)\n{'='*70}")
    logger.info(f"{'Model':<22} {'With hashtags':<16} {'No hashtags':<14} {'Drop':<10}")
    logger.info("-" * 70)
    logger.info(f"{'distilbert':<22} {original_acc:.4f}           {new_acc:.4f}         {drop:+.4f}")

    total_time = time.time() - overall_start
    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
