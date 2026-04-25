# ============================================================
# classical_features.py
# Builds TF-IDF features for classical models (LR, SVM, NB).
#
# Two feature types combined:
#   1. Word-level TF-IDF (1-2 grams): captures phrases like "not great"
#   2. Char-level TF-IDF (3-5 grams): captures stretched spellings
#      like "yeah/yeaah/yeeaah" which signal sarcasm in Twitter data
#
# The two matrices are horizontally stacked into a single sparse matrix.
# The fitted vectorizers are returned so the SAME vocab can be applied
# to val and test data (critical - never refit on test).
# ============================================================

import os
import pickle
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def build_word_vectorizer(config: dict) -> TfidfVectorizer:
    # Word-level TF-IDF: captures word sequences ("not good", "so fun")
    cfg = config["classical"]
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(cfg["tfidf_ngram_range"]),
        max_features=cfg["tfidf_max_features"],
        min_df=2,             # ignore words appearing in fewer than 2 docs (noise)
        max_df=0.95,          # ignore words in more than 95% of docs (stopword-like)
        sublinear_tf=True,    # log-scale term frequency (dampens very common terms)
        lowercase=False,      # already lowercased in preprocessor
    )


def build_char_vectorizer(config: dict) -> TfidfVectorizer:
    # Char-level TF-IDF: captures stretched/misspelled words via 3-5 char n-grams
    # "yeah" -> "yea", "eah", "yeah"; "yeaah" -> "yea", "eaa", "aah", "yeaa", "eaah", "yeaah"
    # so similar-but-not-identical spellings land near each other in feature space
    return TfidfVectorizer(
        analyzer="char_wb",   # char n-grams constrained to word boundaries
        ngram_range=(3, 5),
        max_features=10000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
    )


def fit_features(train_texts, config: dict):
    # Fit both vectorizers on training text only (never on val/test - prevents leakage)
    word_vec = build_word_vectorizer(config)
    char_vec = build_char_vectorizer(config)

    print("[features] Fitting word-level TF-IDF...")
    X_word = word_vec.fit_transform(train_texts)
    print(f"[features] Word TF-IDF shape: {X_word.shape}")

    print("[features] Fitting char-level TF-IDF (3-5 grams)...")
    X_char = char_vec.fit_transform(train_texts)
    print(f"[features] Char TF-IDF shape: {X_char.shape}")

    # Combine both feature matrices side-by-side into one wider sparse matrix
    X_combined = hstack([X_word, X_char]).tocsr()
    print(f"[features] Combined shape: {X_combined.shape}")

    return X_combined, word_vec, char_vec


def transform_features(texts, word_vec: TfidfVectorizer, char_vec: TfidfVectorizer):
    # Apply already-fitted vectorizers to new text (val / test / inference)
    X_word = word_vec.transform(texts)
    X_char = char_vec.transform(texts)
    X_combined = hstack([X_word, X_char]).tocsr()
    return X_combined


def save_vectorizers(word_vec: TfidfVectorizer, char_vec: TfidfVectorizer, save_dir: str, tag: str = ""):
    # Save fitted vectorizers so we can load them later for inference / interpretability
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    word_path = os.path.join(save_dir, f"word_vectorizer{suffix}.pkl")
    char_path = os.path.join(save_dir, f"char_vectorizer{suffix}.pkl")

    with open(word_path, "wb") as f:
        pickle.dump(word_vec, f)
    with open(char_path, "wb") as f:
        pickle.dump(char_vec, f)

    print(f"[features] Saved vectorizers to {save_dir}")


def load_vectorizers(save_dir: str, tag: str = ""):
    # Load previously saved vectorizers
    suffix = f"_{tag}" if tag else ""
    with open(os.path.join(save_dir, f"word_vectorizer{suffix}.pkl"), "rb") as f:
        word_vec = pickle.load(f)
    with open(os.path.join(save_dir, f"char_vectorizer{suffix}.pkl"), "rb") as f:
        char_vec = pickle.load(f)
    return word_vec, char_vec
