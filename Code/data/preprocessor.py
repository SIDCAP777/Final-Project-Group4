# ============================================================
# preprocessor.py
# Cleans raw text and creates train/val/test splits.
#
# Pipeline:
#   1. clean_text: normalize a single string
#        - strip URLs and @mentions
#        - optionally convert emojis to text tokens
#        - optionally strip label-leaking hashtags (#sarcasm, #irony, etc.)
#        - lowercase, collapse whitespace
#   2. tokenize_text: split cleaned text into tokens using NLTK
#        - used by classical features and LSTM
#   3. clean_dataframe: applies cleaning to a whole DataFrame,
#        filters by length, optionally adds a 'tokens' column
#   4. split_data: stratified train/val/test split
# ============================================================

import re
import pandas as pd
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


# Hashtags that act as labels — keeping them leaks the answer to the model
LABEL_HASHTAGS = re.compile(
    r"#(sarcasm|sarcastic|irony|ironic|not|joke|jk|kidding|lol)\b",
    flags=re.IGNORECASE,
)

# Precompile regexes for speed (called 1M+ times)
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
SPECIAL_CHAR_RE = re.compile(r"[^a-zA-Z0-9#\s.,!?]")

# Cache stopwords so we don't re-load them every call
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str,
               lowercase: bool = True,
               remove_urls: bool = True,
               remove_mentions: bool = True,
               remove_special_chars: bool = False,
               remove_label_hashtags: bool = True,
               convert_emojis: bool = True) -> str:
    # Guard against NaN or non-string inputs
    if not isinstance(text, str):
        return ""

    # Remove URLs first (they can contain @ and # which would confuse later steps)
    if remove_urls:
        text = URL_RE.sub(" ", text)

    # Remove @mentions (Twitter usernames — never useful signal for sarcasm)
    if remove_mentions:
        text = MENTION_RE.sub(" ", text)

    # Strip label-leaking hashtags before lowercasing so the regex is simpler
    # Keeps other hashtags (e.g., #politics) intact — those ARE useful features
    if remove_label_hashtags:
        text = LABEL_HASHTAGS.sub(" ", text)

    # Convert emojis to their text descriptions so classical models can use them
    # e.g., "😏" -> ":smirking_face:"
    if convert_emojis:
        text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove rare special characters if asked (we keep hashtags and basic punct)
    if remove_special_chars:
        text = SPECIAL_CHAR_RE.sub(" ", text)

    # Lowercase AFTER all hashtag logic
    if lowercase:
        text = text.lower()

    # Collapse repeated whitespace into a single space and trim ends
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text


def tokenize_text(text: str, remove_stopwords_flag: bool = False) -> list:
    # Use NLTK's word_tokenize for proper tokenization (handles contractions, punct)
    tokens = word_tokenize(text)

    # Rejoin hashtags: NLTK splits "#quote" into ["#", "quote"], but "#" + word
    # is a single semantic unit (hashtags carry sarcasm-relevant meaning)
    merged = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "#" and i + 1 < len(tokens):
            merged.append("#" + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    tokens = merged

    # Optionally filter stopwords. Disabled by default for sarcasm detection:
    # words like "really", "so", "totally" carry sarcasm signal and shouldn't be dropped.
    if remove_stopwords_flag:
        tokens = [t for t in tokens if t not in STOPWORDS]

    return tokens


def clean_dataframe(df: pd.DataFrame, config: dict, add_tokens: bool = False) -> pd.DataFrame:
    # Work on a copy so the caller's DataFrame is never mutated
    df = df.copy()
    prep_cfg = config["preprocessing"]

    # Apply text cleaning row-by-row
    df["text"] = df["text"].apply(
        lambda t: clean_text(
            t,
            lowercase=prep_cfg["lowercase"],
            remove_urls=prep_cfg["remove_urls"],
            remove_mentions=prep_cfg["remove_mentions"],
            remove_special_chars=prep_cfg["remove_special_chars"],
            remove_label_hashtags=prep_cfg.get("remove_label_hashtags", True),
            convert_emojis=prep_cfg.get("convert_emojis", True),
        )
    )

    # Filter by length (drop empty/too-short and absurdly long rows)
    min_len = prep_cfg["min_text_length"]
    max_len = prep_cfg["max_text_length"]
    original_len = len(df)
    df = df[df["text"].str.len().between(min_len, max_len)].reset_index(drop=True)
    dropped = original_len - len(df)

    print(f"[preprocessor] Cleaned {original_len} rows, dropped {dropped} "
          f"({dropped/original_len*100:.1f}%) outside length [{min_len}, {max_len}]")

    # Optionally tokenize. Only needed by classical/LSTM — DistilBERT has its own tokenizer.
    if add_tokens:
        remove_sw = prep_cfg.get("remove_stopwords", False)
        df["tokens"] = df["text"].apply(lambda t: tokenize_text(t, remove_stopwords_flag=remove_sw))
        print(f"[preprocessor] Tokenized {len(df)} rows (remove_stopwords={remove_sw})")

    return df


def split_data(df: pd.DataFrame, val_split: float = 0.1, test_split: float = 0.1,
               seed: int = 42, stratify: bool = True) -> tuple:
    # Stratify on label so class balance is preserved in every split
    stratify_col = df["label"] if stratify else None

    # First split: peel off the test set
    train_val, test = train_test_split(
        df, test_size=test_split, random_state=seed, stratify=stratify_col
    )

    # Second split: carve val out of what's left.
    # Adjust val ratio because train_val is smaller than the original df.
    adjusted_val = val_split / (1 - test_split)
    stratify_col2 = train_val["label"] if stratify else None
    train, val = train_test_split(
        train_val, test_size=adjusted_val, random_state=seed, stratify=stratify_col2
    )

    # Reset indices for cleanliness
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(f"[preprocessor] Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test
