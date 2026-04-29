# ============================================================
# lime_explainer.py
# LIME explanations for individual sarcasm predictions.
#
# For each (model, dataset), explains 20 examples:
#   5 correct sarcasm | 5 correct not-sarcasm | 5 false positives | 5 false negatives
#
# Outputs to outputs/interpretability/:
#   lime_<model>_<dataset>_<i>_<category>.html
#   lime_<model>_<dataset>_summary.json
#
# Run:
#   python Code/interpretability/lime_explainer.py --model logistic_regression --dataset sarc
#   python Code/interpretability/lime_explainer.py --model distilbert --dataset twitter_no_hashtags
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

from utils.seed import set_seed
from utils.config import load_config
from utils.logger import get_logger

from data.loader import load_sarc, load_twitter
from data.preprocessor import clean_dataframe, split_data
from features.classical_features import load_vectorizers, transform_features


CLASS_NAMES = ["Not Sarcasm", "Sarcasm"]


# ---------- Model loaders ----------
def load_classical_predictor(model_name: str, dataset_name: str, config: dict):
    # Load the trained sklearn classifier and matching TF-IDF vectorizers.
    # The "tag" matches what was used at training time for naming.
    models_dir = config["paths"]["saved_models"]
    word_vec, char_vec = load_vectorizers(models_dir, tag=f"classical_{dataset_name}")
    model_path = os.path.join(models_dir, f"classical_{dataset_name}_{model_name}.pkl")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    def predict_proba(texts):
        X = transform_features(list(texts), word_vec, char_vec)
        return clf.predict_proba(X)

    return predict_proba


def load_lstm_predictor(dataset_name: str, config: dict):
    from data.preprocessor import tokenize_text
    from features.glove_embeddings import texts_to_sequences
    from models.lstm import build_lstm

    models_dir = config["paths"]["saved_models"]
    model_path = os.path.join(models_dir, f"lstm_{dataset_name}.pt")
    vocab_path = os.path.join(models_dir, f"lstm_{dataset_name}_vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lstm(config, vocab_size=len(vocab), pretrained_embeddings=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    max_len = config["lstm"]["max_seq_length"]

    def predict_proba(texts):
        token_lists = [tokenize_text(t) for t in texts]
        seqs = texts_to_sequences(token_lists, vocab, max_seq_length=max_len)
        x = torch.from_numpy(seqs).long().to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    return predict_proba


def load_transformer_predictor(model_name: str, dataset_name: str, config: dict):
    # Generic loader for distilbert and roberta
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    models_dir = config["paths"]["saved_models"]
    model_dir = os.path.join(models_dir, f"{model_name}_{dataset_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    max_len = config[model_name]["max_seq_length"]

    def predict_proba(texts):
        # LIME sends ~5000 perturbations; batch to avoid GPU OOM
        all_probs = []
        for i in range(0, len(texts), 64):
            batch = list(texts[i:i+64])
            enc = tokenizer(batch, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    return predict_proba


def get_predictor(model_name: str, dataset_name: str, config: dict):
    if model_name in ("logistic_regression", "linear_svm", "naive_bayes"):
        return load_classical_predictor(model_name, dataset_name, config)
    elif model_name == "lstm":
        return load_lstm_predictor(dataset_name, config)
    elif model_name in ("distilbert", "roberta"):
        return load_transformer_predictor(model_name, dataset_name, config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------- Test data ----------
def load_test_data(dataset_name: str, config: dict, logger):
    # Reproduce the same train/val/test split that was used at training time
    if dataset_name == "sarc":
        df = load_sarc(sarc_dir=config["paths"]["sarc_dir"],
                       sample_size=config["data"].get("sarc_train_sample"),
                       seed=config["seed"])
        df = clean_dataframe(df, config, add_tokens=False)
    elif dataset_name == "twitter_no_hashtags":
        cfg = {**config, "preprocessing": {**config["preprocessing"], "remove_all_hashtags": True}}
        tw_train = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="train")
        tw_test = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="test")
        df = pd.concat([tw_train, tw_test], ignore_index=True)
        df = clean_dataframe(df, cfg, add_tokens=False)
    else:
        tw_train = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="train")
        tw_test = load_twitter(twitter_dir=config["paths"]["twitter_dir"], split="test")
        df = pd.concat([tw_train, tw_test], ignore_index=True)
        df = clean_dataframe(df, config, add_tokens=False)

    _, _, test_df = split_data(df, val_split=config["data"]["val_split"],
                                test_split=0.1, seed=config["seed"])
    logger.info(f"[lime] Test set: {len(test_df)} rows")
    return test_df


def select_examples(test_df, predict_proba, n_per_category: int, max_predict: int, logger):
    # Predict on a manageable subset, then bucket by (true, predicted) labels
    sample = test_df.sample(n=min(max_predict, len(test_df)), random_state=42).reset_index(drop=True)
    texts = sample["text"].tolist()
    labels = sample["label"].values

    logger.info(f"[lime] Predicting on {len(texts)} sampled rows...")
    probs = predict_proba(texts)
    preds = probs.argmax(axis=1)

    correct_sarc = [i for i in range(len(preds)) if labels[i] == 1 and preds[i] == 1]
    correct_not = [i for i in range(len(preds)) if labels[i] == 0 and preds[i] == 0]
    false_pos = [i for i in range(len(preds)) if labels[i] == 0 and preds[i] == 1]
    false_neg = [i for i in range(len(preds)) if labels[i] == 1 and preds[i] == 0]

    logger.info(f"[lime] Buckets: correct_sarc={len(correct_sarc)} correct_not={len(correct_not)} "
                f"FP={len(false_pos)} FN={len(false_neg)}")

    picked = []
    for category, bucket in [
        ("correct_sarcasm", correct_sarc),
        ("correct_not_sarcasm", correct_not),
        ("false_positive", false_pos),
        ("false_negative", false_neg),
    ]:
        for i, idx in enumerate(bucket[:n_per_category]):
            picked.append({
                "category": category,
                "rank": i + 1,
                "text": texts[idx],
                "true_label": int(labels[idx]),
                "predicted_label": int(preds[idx]),
                "prob_not_sarcasm": float(probs[idx][0]),
                "prob_sarcasm": float(probs[idx][1]),
            })

    logger.info(f"[lime] Selected {len(picked)} examples")
    return picked


# ---------- Run LIME ----------
def explain_examples(examples, predict_proba, model_name, dataset_name,
                     config, output_dir, logger):
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, bow=False,
                                  random_state=config["seed"])
    num_features = config["interpretability"]["lime_num_features"]
    num_samples = config["interpretability"]["lime_num_samples"]

    sarcasm_pushers = Counter()
    not_pushers = Counter()
    summary_records = []

    for i, ex in enumerate(examples):
        logger.info(f"[lime] {i+1}/{len(examples)} ({ex['category']}): {ex['text'][:60]!r}...")
        try:
            exp = explainer.explain_instance(
                text_instance=ex["text"],
                classifier_fn=predict_proba,
                num_features=num_features,
                num_samples=num_samples,
                labels=(0, 1),
            )
            html_path = os.path.join(
                output_dir,
                f"{i+1:02d}_{ex['category']}.html",
            )
            exp.save_to_file(html_path)

            # Per-word weights for the Sarcasm class (positive = pushes toward sarcasm)
            word_weights = exp.as_list(label=1)
            for word, weight in word_weights:
                if weight > 0:
                    sarcasm_pushers[word] += weight
                else:
                    not_pushers[word] += abs(weight)

            summary_records.append({
                **ex,
                "word_weights_for_sarcasm": word_weights,
                "html_file": os.path.basename(html_path),
            })
        except Exception as e:
            logger.warning(f"[lime] Skipped example {i+1}: {e}")
            continue

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_explanations": len(summary_records),
        "lime_config": {"num_features": num_features, "num_samples": num_samples},
        "top_words_pushing_toward_sarcasm": sarcasm_pushers.most_common(30),
        "top_words_pushing_toward_not_sarcasm": not_pushers.most_common(30),
        "explanations": summary_records,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[lime] Saved summary to {summary_path}")
    return summary


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["logistic_regression", "linear_svm", "naive_bayes",
                                 "lstm", "distilbert", "roberta"])
    parser.add_argument("--dataset", required=True,
                        choices=["sarc", "twitter", "twitter_no_hashtags"])
    parser.add_argument("--n_per_category", type=int, default=5)
    parser.add_argument("--max_predict", type=int, default=5000)
    args = parser.parse_args()

    config = load_config()
    set_seed(config["seed"])
    logger = get_logger(f"lime_{args.model}_{args.dataset}",
                        log_dir=config["paths"]["logs"])
    logger.info(f"[lime] Model: {args.model} | Dataset: {args.dataset}")

    output_dir = os.path.join(PROJECT_ROOT, "outputs", "interpretability", "lime", f"{args.model}_{args.dataset}")
    os.makedirs(output_dir, exist_ok=True)

    test_df = load_test_data(args.dataset, config, logger)
    predict_proba = get_predictor(args.model, args.dataset, config)
    examples = select_examples(test_df, predict_proba, args.n_per_category,
                                args.max_predict, logger)
    explain_examples(examples, predict_proba, args.model, args.dataset,
                     config, output_dir, logger)
    logger.info("[lime] Done.")


if __name__ == "__main__":
    main()
