# ============================================================
# train_distilbert.py
# Fine-tunes DistilBERT for sarcasm detection on SARC and Twitter.
#
# Pipeline (per dataset):
#   1. Load + clean + split data
#   2. Tokenize text with DistilBERT's WordPiece tokenizer
#   3. Build PyTorch DataLoaders
#   4. Fine-tune with AdamW + linear warmup
#   5. Track val F1, save best checkpoint
#   6. Evaluate on test, save metrics + confusion matrix
#
# Run with:
#   python Code/training/train_distilbert.py
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF parallelism warnings
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Add project Code/ to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

from utils.seed import set_seed
from utils.config import load_config
from utils.logger import get_logger

from data.loader import load_sarc, load_twitter
from data.preprocessor import clean_dataframe, split_data

from models.distilbert import build_distilbert

from evaluation.metrics import (
    compute_metrics, save_metrics, plot_confusion_matrix, print_metrics, get_classification_report
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    # PyTorch Dataset that holds tokenized input_ids + attention_masks + labels.
    # We tokenize all texts upfront to keep training loop fast (no per-step tokenization).
    def __init__(self, texts, labels, tokenizer, max_seq_length: int):
        # Tokenize entire dataset in one go - this is the fast path
        encoded = tokenizer(
            list(texts),
            padding="max_length",     # pad to max_seq_length
            truncation=True,          # cut texts longer than max_seq_length
            max_length=max_seq_length,
            return_tensors="pt",      # return PyTorch tensors
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
        # Move tensors to GPU
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        # Standard step
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        # Clip gradients to prevent instability during transformer fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # update learning rate (linear warmup + decay)

        total_loss += loss.item()
        n_batches += 1

        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Log progress every ~5% of an epoch
        if batch_idx > 0 and batch_idx % max(1, len(loader) // 20) == 0:
            running_acc = correct / total
            running_loss = total_loss / n_batches
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"  Epoch {epoch_num} | batch {batch_idx}/{len(loader)} | "
                        f"loss={running_loss:.4f} | acc={running_acc:.4f} | lr={current_lr:.2e}")

    return total_loss / n_batches, correct / total


@torch.no_grad()
def evaluate(model, loader):
    # Eval mode (disables dropout)
    model.eval()
    all_preds = []
    all_labels = []
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

    avg_loss = total_loss / n_batches
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, np.array(all_labels), np.array(all_preds)


def run_dataset(dataset_name: str, df: pd.DataFrame, config: dict, logger):
    logger.info(f"\n{'='*70}\n  Running DistilBERT pipeline for: {dataset_name}\n{'='*70}")

    # 1. Clean + split (DistilBERT has its own tokenizer, so no add_tokens needed here)
    logger.info(f"[{dataset_name}] Cleaning text...")
    df = clean_dataframe(df, config, add_tokens=False)
    train_df, val_df, test_df = split_data(
        df,
        val_split=config["data"]["val_split"],
        test_split=0.1,
        seed=config["seed"],
    )

    # 2. Build model + tokenizer
    tokenizer, model = build_distilbert(config, num_classes=2)
    model = model.to(DEVICE)

    # 3. Tokenize each split into datasets (this can take a minute on 1M rows)
    max_len = config["distilbert"]["max_seq_length"]
    logger.info(f"[{dataset_name}] Tokenizing train ({len(train_df)} rows)...")
    t0 = time.time()
    train_ds = TextDataset(train_df["text"], train_df["label"], tokenizer, max_len)
    logger.info(f"[{dataset_name}] Train tokenized in {time.time()-t0:.1f}s")

    logger.info(f"[{dataset_name}] Tokenizing val + test...")
    val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer, max_len)
    test_ds = TextDataset(test_df["text"], test_df["label"], tokenizer, max_len)

    # 4. DataLoaders
    batch_size = config["distilbert"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # 5. Optimizer (AdamW) + linear warmup scheduler.
    # Warmup helps transformers train stably - learning rate ramps up from 0
    # over the first warmup_ratio of steps, then linearly decays to 0.
    epochs = config["distilbert"]["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(config["distilbert"]["warmup_ratio"] * total_steps)

    optimizer = AdamW(
        model.parameters(),
        lr=config["distilbert"]["learning_rate"],
        weight_decay=config["distilbert"]["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"[{dataset_name}] Training: {epochs} epochs, "
                f"batch_size={batch_size}, total_steps={total_steps}, warmup={warmup_steps}")

    # 6. Training loop with best-model tracking
    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{epochs} ({dataset_name}) ---")
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
            # State dict copy (CPU side) so we don't blow up GPU memory
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  ** New best val F1 = {best_val_f1:.4f} (saving) **")

    # 7. Restore best model + evaluate on test
    logger.info(f"\n[{dataset_name}] Restoring best model (val F1={best_val_f1:.4f})")
    model.load_state_dict(best_state)
    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader)

    test_metrics["model"] = "distilbert"
    test_metrics["dataset"] = dataset_name
    test_metrics["split"] = "test"
    test_metrics["best_val_f1_macro"] = best_val_f1

    print_metrics(test_metrics, name=f"{dataset_name} | DistilBERT | TEST")

    # 8. Save everything
    results_dir = config["paths"]["results"]
    plots_dir = config["paths"]["plots"]
    models_dir = config["paths"]["saved_models"]

    save_metrics(test_metrics, save_dir=results_dir, name=f"distilbert_{dataset_name}_test")
    plot_confusion_matrix(y_true, y_pred, save_dir=plots_dir,
                          name=f"distilbert_{dataset_name}_test")

    history_path = os.path.join(results_dir, f"distilbert_{dataset_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"[{dataset_name}] Saved training history to {history_path}")

    report_path = os.path.join(results_dir, f"distilbert_{dataset_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: DistilBERT\nDataset: {dataset_name}\nBest val F1: {best_val_f1:.4f}\n\n")
        f.write("--- TEST ---\n")
        f.write(get_classification_report(y_true, y_pred))

    # Save fine-tuned model + tokenizer using HuggingFace's standard format
    model_dir = os.path.join(models_dir, f"distilbert_{dataset_name}")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"[{dataset_name}] Saved model + tokenizer to {model_dir}")

    return test_metrics


def main():
    config = load_config()
    set_seed(config["seed"])
    logger = get_logger("train_distilbert", log_dir=config["paths"]["logs"])
    logger.info(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    overall_start = time.time()
    all_results = {}

    # ---- SARC ----
    logger.info("\nLoading SARC dataset...")
    sarc_df = load_sarc(
        sarc_dir=config["paths"]["sarc_dir"],
        sample_size=config["data"].get("sarc_train_sample"),
        seed=config["seed"],
    )
    all_results["sarc"] = run_dataset("sarc", sarc_df, config, logger)

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
    all_results["twitter"] = run_dataset("twitter", twitter_df, config, logger)

    # ---- Final summary ----
    logger.info(f"\n{'='*70}\n  FINAL DISTILBERT SUMMARY (test set)\n{'='*70}")
    logger.info(f"{'Dataset':<10} {'Accuracy':<10} {'F1 (macro)':<12} {'Best val F1':<12}")
    logger.info("-" * 70)
    for ds, m in all_results.items():
        logger.info(f"{ds:<10} {m['accuracy']:.4f}     {m['f1_macro']:.4f}       "
                    f"{m['best_val_f1_macro']:.4f}")

    total_time = time.time() - overall_start
    logger.info(f"\nTotal DistilBERT training time: {total_time:.1f}s "
                f"({total_time/60:.1f} min, {total_time/3600:.1f} hr)")


if __name__ == "__main__":
    main()
