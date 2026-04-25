# ============================================================
# loader.py
# Loads the raw SARC and Twitter datasets into pandas DataFrames.
# Both loaders return a unified format with columns: [text, label, source]
# - text: the raw text of the comment/tweet
# - label: 1 for sarcasm, 0 for not sarcasm
# - source: "sarc" or "twitter" for tracking which dataset the row came from
# ============================================================

import os
import pandas as pd


def load_sarc(sarc_dir: str, sample_size: int = None, seed: int = 42) -> pd.DataFrame:
    # Path to the only SARC file with actual text comments
    file_path = os.path.join(sarc_dir, "train-balanced-sarcasm.csv")

    # Load the CSV (pandas handles the header automatically)
    df = pd.read_csv(file_path)

    # Keep only the columns we need and drop rows with missing text
    df = df[["comment", "label"]].dropna()

    # Rename 'comment' to 'text' for a unified schema
    df = df.rename(columns={"comment": "text"})

    # Add a source tag so we can track where each row came from
    df["source"] = "sarc"

    # Optionally subsample for faster training (especially useful for DistilBERT)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    print(f"[loader] Loaded SARC: {len(df)} rows | label distribution:\n{df['label'].value_counts().to_dict()}")
    return df


def load_twitter(twitter_dir: str, split: str = "train", sample_size: int = None, seed: int = 42) -> pd.DataFrame:
    # Path to the train or test file
    file_path = os.path.join(twitter_dir, f"{split}.csv")

    # Load the CSV
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    df = df.dropna(subset=["tweets", "class"])

    # Binarize the 4-class labels: sarcasm -> 1, regular -> 0
    # Drop 'irony' and 'figurative' to avoid label noise (they are ambiguous middle cases)
    df = df[df["class"].isin(["sarcasm", "regular"])]
    df["label"] = (df["class"] == "sarcasm").astype(int)

    # Keep only the columns we need and rename for unified schema
    df = df[["tweets", "label"]].rename(columns={"tweets": "text"})

    # Add source tag
    df["source"] = "twitter"

    # Optionally subsample
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    print(f"[loader] Loaded Twitter ({split}): {len(df)} rows | label distribution:\n{df['label'].value_counts().to_dict()}")
    return df


def load_all(config: dict) -> dict:
    # Loads all datasets using paths and sample sizes from config.
    # Returns a dict with keys: 'sarc', 'twitter_train', 'twitter_test'
    seed = config["seed"]
    paths = config["paths"]
    data_cfg = config["data"]

    datasets = {
        "sarc": load_sarc(
            sarc_dir=paths["sarc_dir"],
            sample_size=data_cfg.get("sarc_train_sample"),
            seed=seed,
        ),
        "twitter_train": load_twitter(
            twitter_dir=paths["twitter_dir"],
            split="train",
            sample_size=data_cfg.get("twitter_train_sample"),
            seed=seed,
        ),
        "twitter_test": load_twitter(
            twitter_dir=paths["twitter_dir"],
            split="test",
            sample_size=data_cfg.get("twitter_test_sample"),
            seed=seed,
        ),
    }
    return datasets
