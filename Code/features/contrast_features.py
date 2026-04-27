# ============================================================
# contrast_features.py
# Computes "sentiment contrast" features for sarcasm detection.
#
# Hypothesis: sarcastic text often contains BOTH positive and
# negative sentiment words (e.g., "I LOVE being STUCK in traffic"),
# while genuine text usually has consistent polarity.
#
# Uses VADER sentiment lexicon (designed for social media text).
# For each text we extract:
#   - pos_score, neg_score: VADER's positive/negative ratios
#   - has_both: 1 if both pos>0 and neg>0 (the core contrast signal)
#   - contrast_score: min(pos, neg) - high when both polarities coexist
#   - polarity_swing: pos + neg (total polarized content)
#   - compound: VADER's overall sentiment in [-1, 1]
#   - strongest_pos / strongest_neg: max polarity word scores in text
#   - num_pos_words, num_neg_words: counts of polarized words
# ============================================================

import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize


class ContrastFeaturizer:
    # Wraps VADER + adds clause-level contrast features.
    # Initialize once, reuse across many texts (VADER setup is cheap but not free).

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def features_for_text(self, text: str) -> dict:
        # Guard against bad input
        if not isinstance(text, str) or not text.strip():
            return self._empty_features()

        # Whole-text VADER scores
        scores = self.sia.polarity_scores(text)
        pos = scores["pos"]
        neg = scores["neg"]
        compound = scores["compound"]

        # Per-word sentiment scan: walk through tokens and look up each
        # in VADER's internal lexicon. Track the strongest positive and
        # strongest negative individual word scores.
        tokens = word_tokenize(text.lower())
        word_scores = []
        for tok in tokens:
            if tok in self.sia.lexicon:
                word_scores.append(self.sia.lexicon[tok])

        if word_scores:
            num_pos = sum(1 for s in word_scores if s > 0)
            num_neg = sum(1 for s in word_scores if s < 0)
            strongest_pos = max(word_scores) if any(s > 0 for s in word_scores) else 0.0
            strongest_neg = min(word_scores) if any(s < 0 for s in word_scores) else 0.0
        else:
            num_pos = 0
            num_neg = 0
            strongest_pos = 0.0
            strongest_neg = 0.0

        # Core derived features
        has_both = int(pos > 0 and neg > 0)
        contrast_score = min(pos, neg)         # High when both polarities present strongly
        polarity_swing = pos + neg              # Total non-neutral content
        # Clash flag: at least one positive AND one negative word, AND meaningful intensity
        word_clash = int(num_pos >= 1 and num_neg >= 1)

        return {
            "pos_score": pos,
            "neg_score": neg,
            "compound": compound,
            "has_both": has_both,
            "contrast_score": contrast_score,
            "polarity_swing": polarity_swing,
            "strongest_pos": float(strongest_pos),
            "strongest_neg": float(strongest_neg),
            "num_pos_words": num_pos,
            "num_neg_words": num_neg,
            "word_clash": word_clash,
        }

    def _empty_features(self) -> dict:
        # All-zero feature vector for empty/invalid inputs
        return {
            "pos_score": 0.0, "neg_score": 0.0, "compound": 0.0,
            "has_both": 0, "contrast_score": 0.0, "polarity_swing": 0.0,
            "strongest_pos": 0.0, "strongest_neg": 0.0,
            "num_pos_words": 0, "num_neg_words": 0, "word_clash": 0,
        }

    def transform(self, texts) -> np.ndarray:
        # Convert a list of texts into a numpy matrix of features.
        # Returns shape (n_texts, n_features). Order of feature columns is
        # given by FEATURE_NAMES below - keep this stable for reproducibility.
        rows = []
        for t in texts:
            feats = self.features_for_text(t)
            rows.append([feats[name] for name in FEATURE_NAMES])
        return np.array(rows, dtype=np.float32)


# Order matters: when we hstack with TF-IDF, these become the trailing columns
FEATURE_NAMES = [
    "pos_score",
    "neg_score",
    "compound",
    "has_both",
    "contrast_score",
    "polarity_swing",
    "strongest_pos",
    "strongest_neg",
    "num_pos_words",
    "num_neg_words",
    "word_clash",
]


def get_feature_names() -> list:
    # Used by interpretability code (LIME/SHAP) so it knows what each column means
    return FEATURE_NAMES.copy()
