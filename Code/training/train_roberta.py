# ============================================================
# train_roberta.py
# Fine-tunes RoBERTa-base for sarcasm detection on SARC and Twitter.
#
# Key differences vs train_distilbert.py:
#   1. Uses RoBERTa-base (125M params) instead of DistilBERT (66M)
#   2. Optionally concatenates parent comment context for SARC,
#      formatted as "<parent> </s></s> <comment>" using RoBERTa's
#      paired-input convention (sep_token = </s>)
#   3. Early stopping on val F1 (patience configurable in config)
#   4. Twitter is run with hashtags stripped (matches the leakage test
#      protocol used for LSTM and DistilBERT)
#
# Run with:
#   python Code/training/train_roberta.py
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
from data.preprocessor import clean_text, clean_dataframe, split_data
from models.roberta import build_roberta
from evaluation.metrics import (
    compute_metrics, save_metrics, plot_confusion_matrix,
    print_metrics, get_classification_report,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- SARC loader with parent comment ----------
def load_sarc_with_parent(sarc_dir: str, sample_size: int = None, seed: int = 42) -> pd.DataFrame:
    # Custom SARC loader that keeps the parent_comment column.
    # The standard load_sarc() in data/loader.py only keeps comment+label;
    # for RoBERTa we want the parent comment too as conversation context.
    file_path = os.path.join(sarc_dir, "train-balanced-sarcasm.csv")
    df = pd.read_csv(file_path)
    df = df[["comment", "label", "parent_comment"]].dropna(subset=["comment", "label"])
    df = df.rename(columns={"comment": "text"})
    # Empty parent comments stay as empty string (handled later)
    df["parent_comment"] = df["parent_comment"].fillna("")
    df["source"] = "sarc"
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    print(f"[loader] Loaded SARC w/ parent: {len(df)} rows | label distribution:\n"
          f"{df['label'].value_counts().to_dict()}")
    return df


def clean_sarc_with_parent(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Apply the same text cleaning to BOTH the comment and the parent_comment
    # so the model sees consistently formatted text on both sides of [SEP].
    df = df.copy()
    prep_cfg = config["preprocessing"]

    def _clean(t):
        return clean_text(
            t,
            lowercase=prep_cfg["lowercase"],
            remove_urls=prep_cfg["remove_urls"],
            remove_mentions=prep_cfg["remove_mentions"],
            remove_special_chars=prep_cfg["remove_special_chars"],
            remove_label_hashtags=prep_cfg.get("remove_label_hashtags", True),
            remove_all_hashtags=prep_cfg.get("remove_all_hashtags", False),
            convert_emojis=prep_cfg.get("convert_emojis", True),
        )

    df["text"] = df["text"].apply(_clean)
    df["parent_comment"] = df["parent_comment"].apply(_clean)

    # Filter on COMMENT length (don't filter on parent - many parents are long, that's fine)
    min_len = prep_cfg["min_text_length"]
    max_len = prep_cfg["max_text_length"]
    original_len = len(df)
    df = df[df["text"].str.len().between(min_len, max_len)].reset_index(drop=True)
    dropped = original_len - len(df)
    print(f"[preprocessor] SARC w/ parent: cleaned {original_len} rows, dropped {dropped} "
          f"({dropped/original_len*100:.1f}%) outside length [{min_len}, {max_len}]")
    return df


# ---------- Dataset class ----------
class TextDataset(Dataset):
    # Tokenizes single texts OR (parent, text) pairs upfront.
    # For pairs, RoBERTa's tokenizer formats them as "<parent></s></s><text>" automatically.
    def __init__(self, texts, labels, tokenizer, max_seq_length: int, parent_texts=None):
        if parent_texts is not None:
            # Pair encoding - RoBERTa handles the <s>...</s></s>...</s> formatting
            encoded = tokenizer(
                list(parent_texts),
                list(texts),
                padding="max_length",
                truncation="only_first",  # truncate parent (the longer side) first, keep comment intact
                max_length=max_seq_length,
                return_tensors="pt",
            )
        else:
            # Single text encoding (used for Twitter)
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


# ---------- Train / eval loops ----------
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
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Log every ~5% of an epoch (SARC is huge)
        if batch_idx > 0 and batch_idx % max(1, len(loader) // 20) == 0:
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


# ---------- Per-dataset orchestration ----------
def run_dataset(dataset_name: str, train_df, val_df, test_df,
                config: dict, logger, use_parent_context: bool):
    logger.info(f"\n{'='*70}\n  Running RoBERTa pipeline for: {dataset_name} "
                f"(parent_context={use_parent_context})\n{'='*70}")

    # 1. Build model + tokenizer
    tokenizer, model = build_roberta(config, num_classes=2)
    model = model.to(DEVICE)

    # 2. Tokenize - SARC uses paired (parent, comment); Twitter uses single text
    max_len = config["roberta"]["max_seq_length"]
    logger.info(f"[{dataset_name}] Tokenizing train ({len(train_df)} rows)...")
    t0 = time.time()
    if use_parent_context:
        train_ds = TextDataset(train_df["text"], train_df["label"], tokenizer, max_len,
                               parent_texts=train_df["parent_comment"])
        val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer, max_len,
                             parent_texts=val_df["parent_comment"])
        test_ds = TextDataset(test_df["text"], test_df["label"], tokenizer, max_len,
                              parent_texts=test_df["parent_comment"])
    else:
        train_ds = TextDataset(train_df["text"], train_df["label"], tokenizer, max_len)
        val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer, max_len)
        test_ds = TextDataset(test_df["text"], test_df["label"], tokenizer, max_len)
    logger.info(f"[{dataset_name}] Tokenization done in {time.time()-t0:.1f}s")

    # 3. DataLoaders
    batch_size = config["roberta"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # 4. Optimizer + scheduler
    epochs = config["roberta"]["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(config["roberta"]["warmup_ratio"] * total_steps)
    optimizer = AdamW(
        model.parameters(),
        lr=config["roberta"]["learning_rate"],
        weight_decay=config["roberta"]["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    logger.info(f"[{dataset_name}] Training: max {epochs} epochs, "
                f"batch_size={batch_size}, total_steps={total_steps}, warmup={warmup_steps}, "
                f"early_stopping_patience={config['roberta']['early_stopping_patience']}")

    # 5. Training loop with best-model tracking + early stopping
    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    history = []
    patience = config["roberta"]["early_stopping_patience"]
    actual_epochs_run = 0

    for epoch in range(1, epochs + 1):
        actual_epochs_run = epoch
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
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
            logger.info(f"  ** New best val F1 = {best_val_f1:.4f} (saving) **")
        else:
            epochs_since_improve += 1
            logger.info(f"  No improvement ({epochs_since_improve}/{patience} patience used)")
            # Early stop if patience exhausted (and we have at least 2 epochs done)
            if epochs_since_improve >= patience and epoch >= 2:
                logger.info(f"  ** Early stopping triggered at epoch {epoch} (best epoch={best_epoch}) **")
                break

    # 6. Restore best + evaluate test
    logger.info(f"\n[{dataset_name}] Restoring best model "
                f"(epoch={best_epoch}, val F1={best_val_f1:.4f})")
    model.load_state_dict(best_state)
    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader)

    test_metrics["model"] = "roberta"
    test_metrics["dataset"] = dataset_name
    test_metrics["split"] = "test"
    test_metrics["best_val_f1_macro"] = best_val_f1
    test_metrics["best_epoch"] = best_epoch
    test_metrics["epochs_run"] = actual_epochs_run
    test_metrics["use_parent_context"] = use_parent_context

    print_metrics(test_metrics, name=f"{dataset_name} | RoBERTa | TEST")

    # 7. Save everything
    results_dir = config["paths"]["results"]
    plots_dir = config["paths"]["plots"]
    models_dir = config["paths"]["saved_models"]

    save_metrics(test_metrics, save_dir=results_dir, name=f"roberta_{dataset_name}_test")
    plot_confusion_matrix(y_true, y_pred, save_dir=plots_dir, name=f"roberta_{dataset_name}_test")

    history_path = os.path.join(results_dir, f"roberta_{dataset_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"[{dataset_name}] Saved training history to {history_path}")

    report_path = os.path.join(results_dir, f"roberta_{dataset_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: RoBERTa-base\nDataset: {dataset_name}\n"
                f"Use parent context: {use_parent_context}\n"
                f"Best val F1: {best_val_f1:.4f} (epoch {best_epoch})\n"
                f"Epochs run: {actual_epochs_run}/{epochs}\n\n")
        f.write("--- TEST ---\n")
        f.write(get_classification_report(y_true, y_pred))

    # Save fine-tuned model + tokenizer (HuggingFace standard format)
    model_dir = os.path.join(models_dir, f"roberta_{dataset_name}")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"[{dataset_name}] Saved model + tokenizer to {model_dir}")

    return test_metrics


def main():
    config = load_config()
    set_seed(config["seed"])
    logger = get_logger("train_roberta", log_dir=config["paths"]["logs"])
    logger.info(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    overall_start = time.time()
    all_results = {}

    use_parent = config["roberta"].get("use_parent_context", True)

    # ---- SARC (with parent comment context) ----
    logger.info("\nLoading SARC dataset (with parent comments)...")
    sarc_df = load_sarc_with_parent(
        sarc_dir=config["paths"]["sarc_dir"],
        sample_size=config["data"].get("sarc_train_sample"),
        seed=config["seed"],
    )
    logger.info("Cleaning SARC text + parent comments...")
    sarc_df = clean_sarc_with_parent(sarc_df, config)
    sarc_train, sarc_val, sarc_test = split_data(
        sarc_df, val_split=config["data"]["val_split"],
        test_split=0.1, seed=config["seed"],
    )
    all_results["sarc"] = run_dataset(
        "sarc", sarc_train, sarc_val, sarc_test,
        config, logger, use_parent_context=use_parent,
    )

    # ---- Twitter (no hashtags, single text - no parent column exists) ----
    logger.info("\nLoading Twitter dataset (with hashtags removed for fair comparison)...")
    # Force hashtag stripping for Twitter (matches the LSTM and DistilBERT leakage tests)
    tw_config = {**config, "preprocessing": {**config["preprocessing"], "remove_all_hashtags": True}}
    tw_train = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="train")
    tw_test = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="test")
    twitter_df = pd.concat([tw_train, tw_test], ignore_index=True)
    twitter_df = clean_dataframe(twitter_df, tw_config, add_tokens=False)
    tw_train_df, tw_val_df, tw_test_df = split_data(
        twitter_df, val_split=config["data"]["val_split"],
        test_split=0.1, seed=config["seed"],
    )
    # Twitter has no parent context, so we override use_parent_context=False here
    all_results["twitter_no_hashtags"] = run_dataset(
        "twitter_no_hashtags", tw_train_df, tw_val_df, tw_test_df,
        config, logger, use_parent_context=False,
    )

    # ---- Final summary ----
    logger.info(f"\n{'='*70}\n  FINAL ROBERTA SUMMARY (test set)\n{'='*70}")
    logger.info(f"{'Dataset':<22} {'Accuracy':<10} {'F1 (macro)':<12} {'Best val F1':<12} {'Best ep':<8}")
    logger.info("-" * 70)
    for ds, m in all_results.items():
        logger.info(f"{ds:<22} {m['accuracy']:.4f}     {m['f1_macro']:.4f}       "
                    f"{m['best_val_f1_macro']:.4f}       {m['best_epoch']}")

    total_time = time.time() - overall_start
    logger.info(f"\nTotal RoBERTa training time: {total_time:.1f}s "
                f"({total_time/60:.1f} min, {total_time/3600:.1f} hr)")


if __name__ == "__main__":
    main()
