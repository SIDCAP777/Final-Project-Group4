# ============================================================
# aggregate_results.py
# Walks outputs/results/*.json and outputs/interpretability/**/summary.json,
# builds master comparison artifacts for the report.
#
# Outputs to outputs/aggregation/:
#   all_results.csv               (test-set metrics, one row per model/dataset/variant)
#   all_results.md                (markdown table for the report)
#   interpretability_summary.md   (LIME vs SHAP top-words side by side)
#
# Run:
#   python Code/aggregation/aggregate_results.py
# ============================================================

import os
import re
import sys
import json
import glob
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
INTERP_DIR = os.path.join(PROJECT_ROOT, "outputs", "interpretability")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "aggregation")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- pretty-name maps ----------
MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "naive_bayes": "Naive Bayes",
    "lstm": "BiLSTM (GloVe)",
    "distilbert": "DistilBERT",
    "roberta": "RoBERTa-base",
}
DATASET_DISPLAY = {
    "sarc": "SARC",
    "twitter": "Twitter (with hashtags)",
    "twitter_no_hashtags": "Twitter (no hashtags)",
}
# Order for sorting rows in the comparison table
MODEL_ORDER = ["logistic_regression", "linear_svm", "naive_bayes",
               "lstm", "distilbert", "roberta"]
DATASET_ORDER = ["sarc", "twitter", "twitter_no_hashtags"]
VARIANT_ORDER = ["tfidf", "tfidf+contrast"]


# ---------- filename parser ----------
def parse_test_filename(fname):
    """
    Returns (model, dataset, variant) or None if not a test-metric JSON.

    Supported patterns:
      classical_<dataset>_<model>_test_<ts>.json                    → variant=tfidf
      classical_<dataset>_<model>_with_contrast_test_<ts>.json      → variant=tfidf+contrast
      <lstm|distilbert|roberta>_<dataset>_test_<ts>.json            → variant=base
    where <dataset> is one of: sarc, twitter, twitter_no_hashtags
    """
    base = os.path.basename(fname)
    if not base.endswith(".json") or "_test_" not in base:
        return None
    if "_history" in base or "_val_" in base:
        return None

    # Strip trailing _test_<ts>.json
    stem = re.sub(r"_test_\d{8}_\d{6}\.json$", "", base)

    # Classical pattern (with optional +contrast)
    m = re.match(r"^classical_(sarc|twitter|twitter_no_hashtags)_(logistic_regression|linear_svm|naive_bayes)(_with_contrast)?$", stem)
    if m:
        dataset = m.group(1)
        model = m.group(2)
        variant = "tfidf+contrast" if m.group(3) else "tfidf"
        return model, dataset, variant

    # Deep model pattern
    m = re.match(r"^(lstm|distilbert|roberta)_(sarc|twitter|twitter_no_hashtags)$", stem)
    if m:
        return m.group(1), m.group(2), "base"

    return None


# ---------- collect test-metric rows ----------
def collect_test_results():
    rows = []
    seen = {}  # (model, dataset, variant) -> (timestamp_str, row)

    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        parsed = parse_test_filename(path)
        if parsed is None:
            continue
        model, dataset, variant = parsed

        with open(path) as f:
            metrics = json.load(f)

        # Extract timestamp for de-dup (keep latest)
        ts_match = re.search(r"_test_(\d{8}_\d{6})\.json$", path)
        ts = ts_match.group(1) if ts_match else "00000000_000000"

        per_class = metrics.get("per_class", {})
        c0 = per_class.get("0", {}) or per_class.get("Not Sarcasm", {}) or {}
        c1 = per_class.get("1", {}) or per_class.get("Sarcasm", {}) or {}

        row = {
            "model": model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "dataset": dataset,
            "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
            "variant": variant,
            "timestamp": ts,
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "class0_precision": c0.get("precision"),
            "class0_recall": c0.get("recall"),
            "class0_f1": c0.get("f1"),
            "class1_precision": c1.get("precision"),
            "class1_recall": c1.get("recall"),
            "class1_f1": c1.get("f1"),
            "source_file": os.path.basename(path),
        }
        key = (model, dataset, variant)
        if key in seen and seen[key][0] >= ts:
            continue
        seen[key] = (ts, row)

    return [r for _, r in seen.values()]


# ---------- writers ----------
def write_csv(rows, out_path):
    if not rows:
        print("[warn] No rows to write to CSV")
        return
    cols = ["model_display", "dataset_display", "variant",
            "accuracy", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro",
            "class0_precision", "class0_recall", "class0_f1",
            "class1_precision", "class1_recall", "class1_f1",
            "source_file"]

    def sort_key(r):
        return (DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 99,
                MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
                VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99)

    rows = sorted(rows, key=sort_key)
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c)
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"[ok] CSV: {out_path}")


def write_markdown_main(rows, out_path):
    """Compact comparison table: model rows × dataset columns."""
    # Collapse to (model, dataset) -> accuracy when variant=base or tfidf
    pivot_acc = defaultdict(dict)
    pivot_f1 = defaultdict(dict)
    for r in rows:
        if r["variant"] not in ("base", "tfidf"):
            continue
        pivot_acc[r["model"]][r["dataset"]] = r["accuracy"]
        pivot_f1[r["model"]][r["dataset"]] = r["f1_macro"]

    lines = []
    lines.append("# Master Results Comparison")
    lines.append("")
    lines.append("All numbers are **test-set** metrics. Twitter no-hashtags is the honest evaluation; Twitter with hashtags shows leakage.")
    lines.append("")
    lines.append("## Test Accuracy")
    lines.append("")
    header = "| Model | " + " | ".join(DATASET_DISPLAY[d] for d in DATASET_ORDER) + " |"
    sep = "|---" * (1 + len(DATASET_ORDER)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in MODEL_ORDER:
        if m not in pivot_acc:
            continue
        row = ["**" + MODEL_DISPLAY[m] + "**"]
        for d in DATASET_ORDER:
            v = pivot_acc[m].get(d)
            row.append(f"{v:.4f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Test F1 (macro)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for m in MODEL_ORDER:
        if m not in pivot_f1:
            continue
        row = ["**" + MODEL_DISPLAY[m] + "**"]
        for d in DATASET_ORDER:
            v = pivot_f1[m].get(d)
            row.append(f"{v:.4f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Hashtag leakage drop
    lines.append("## Hashtag-Leakage Drop on Twitter")
    lines.append("")
    lines.append("Stripping all hashtags exposes label leakage. Drop = (with hashtags) − (no hashtags).")
    lines.append("")
    lines.append("| Model | Twitter w/ hashtags | Twitter no hashtags | Drop (pp) |")
    lines.append("|---|---|---|---|")
    for m in MODEL_ORDER:
        if m not in pivot_acc:
            continue
        with_h = pivot_acc[m].get("twitter")
        no_h = pivot_acc[m].get("twitter_no_hashtags")
        if with_h is None or no_h is None:
            continue
        drop_pp = (with_h - no_h) * 100
        lines.append(f"| **{MODEL_DISPLAY[m]}** | {with_h:.4f} | {no_h:.4f} | {drop_pp:+.2f} |")
    lines.append("")

    # Contrast features ablation
    lines.append("## Contrast Features Ablation (TF-IDF + 11 VADER features)")
    lines.append("")
    lines.append("Comparison vs the same model without contrast features. Δ = with-contrast − baseline.")
    lines.append("")
    lines.append("| Model | Dataset | TF-IDF only | + contrast | Δ |")
    lines.append("|---|---|---|---|---|")
    by_key = {(r["model"], r["dataset"], r["variant"]): r["accuracy"] for r in rows}
    for d in ["sarc", "twitter"]:
        for m in ["logistic_regression", "linear_svm", "naive_bayes"]:
            base = by_key.get((m, d, "tfidf"))
            ctr = by_key.get((m, d, "tfidf+contrast"))
            if base is None or ctr is None:
                continue
            delta = (ctr - base) * 100
            lines.append(f"| **{MODEL_DISPLAY[m]}** | {DATASET_DISPLAY[d]} | {base:.4f} | {ctr:.4f} | {delta:+.2f} pp |")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[ok] Markdown: {out_path}")


def write_interpretability_summary(out_path):
    """Side-by-side LIME vs SHAP top words for each (model, dataset)."""
    combos = []
    lime_root = os.path.join(INTERP_DIR, "lime")
    if os.path.isdir(lime_root):
        for sub in sorted(os.listdir(lime_root)):
            shap_summary = os.path.join(INTERP_DIR, "shap", sub, "summary.json")
            lime_summary = os.path.join(lime_root, sub, "summary.json")
            entry = {"combo": sub, "lime": None, "shap": None}
            if os.path.isfile(lime_summary):
                with open(lime_summary) as f:
                    entry["lime"] = json.load(f)
            if os.path.isfile(shap_summary):
                with open(shap_summary) as f:
                    entry["shap"] = json.load(f)
            combos.append(entry)

    lines = ["# Interpretability Summary (LIME vs SHAP)", ""]
    lines.append("Top 15 words pushing toward each class, aggregated across 20 explained examples per combo.")
    lines.append("Cross-method agreement (words in BOTH lists) signals a robust learned cue.")
    lines.append("")

    for entry in combos:
        lines.append(f"## {entry['combo']}")
        lines.append("")
        lime_data = entry["lime"]
        shap_data = entry["shap"]

        # Sarcasm pushers side-by-side
        lines.append("### Top sarcasm pushers")
        lines.append("")
        lines.append("| Rank | LIME word | LIME weight | SHAP word | SHAP weight |")
        lines.append("|---|---|---|---|---|")
        lime_top = lime_data["top_words_pushing_toward_sarcasm"][:15] if lime_data else []
        shap_top = shap_data["top_words_pushing_toward_sarcasm"][:15] if shap_data else []
        # Compute overlap for highlighting
        lime_words = {w for w, _ in lime_top}
        shap_words = {w for w, _ in shap_top}
        overlap = lime_words & shap_words
        for i in range(15):
            lw, lv = lime_top[i] if i < len(lime_top) else ("", "")
            sw, sv = shap_top[i] if i < len(shap_top) else ("", "")
            lw_disp = f"**{lw}**" if lw in overlap else lw
            sw_disp = f"**{sw}**" if sw in overlap else sw
            lv_disp = f"{lv:.4f}" if isinstance(lv, float) else lv
            sv_disp = f"{sv:.4f}" if isinstance(sv, float) else sv
            lines.append(f"| {i+1} | {lw_disp} | {lv_disp} | {sw_disp} | {sv_disp} |")
        lines.append("")
        lines.append(f"**Cross-method agreement (sarcasm):** {len(overlap)} / 15 top words appear in both lists: `{', '.join(sorted(overlap))}`")
        lines.append("")

        # Not-sarcasm pushers side-by-side
        lines.append("### Top not-sarcasm pushers")
        lines.append("")
        lines.append("| Rank | LIME word | LIME weight | SHAP word | SHAP weight |")
        lines.append("|---|---|---|---|---|")
        lime_top = lime_data["top_words_pushing_toward_not_sarcasm"][:15] if lime_data else []
        shap_top = shap_data["top_words_pushing_toward_not_sarcasm"][:15] if shap_data else []
        lime_words = {w for w, _ in lime_top}
        shap_words = {w for w, _ in shap_top}
        overlap_n = lime_words & shap_words
        for i in range(15):
            lw, lv = lime_top[i] if i < len(lime_top) else ("", "")
            sw, sv = shap_top[i] if i < len(shap_top) else ("", "")
            lw_disp = f"**{lw}**" if lw in overlap_n else lw
            sw_disp = f"**{sw}**" if sw in overlap_n else sw
            lv_disp = f"{lv:.4f}" if isinstance(lv, float) else lv
            sv_disp = f"{sv:.4f}" if isinstance(sv, float) else sv
            lines.append(f"| {i+1} | {lw_disp} | {lv_disp} | {sw_disp} | {sv_disp} |")
        lines.append("")
        lines.append(f"**Cross-method agreement (not-sarcasm):** {len(overlap_n)} / 15 top words: `{', '.join(sorted(overlap_n))}`")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[ok] Interp summary: {out_path}")


def print_terminal_summary(rows):
    print("\n" + "=" * 72)
    print("  HEADLINE COMPARISON (test accuracy / F1 macro)")
    print("=" * 72)
    pivot_acc = defaultdict(dict)
    pivot_f1 = defaultdict(dict)
    for r in rows:
        if r["variant"] in ("base", "tfidf"):
            pivot_acc[r["model"]][r["dataset"]] = r["accuracy"]
            pivot_f1[r["model"]][r["dataset"]] = r["f1_macro"]
    print(f"\n{'Model':<22} | {'SARC':<14} | {'Twit (w/hash)':<14} | {'Twit (no hash)':<14}")
    print("-" * 72)
    for m in MODEL_ORDER:
        if m not in pivot_acc:
            continue
        cells = []
        for d in DATASET_ORDER:
            a = pivot_acc[m].get(d)
            f1 = pivot_f1[m].get(d)
            cells.append(f"{a:.4f}/{f1:.4f}" if a is not None else "      —      ")
        print(f"{MODEL_DISPLAY[m]:<22} | {cells[0]:<14} | {cells[1]:<14} | {cells[2]:<14}")
    print()


def main():
    print(f"[agg] Reading from {RESULTS_DIR}")
    rows = collect_test_results()
    print(f"[agg] Parsed {len(rows)} test-metric rows")

    write_csv(rows, os.path.join(OUT_DIR, "all_results.csv"))
    write_markdown_main(rows, os.path.join(OUT_DIR, "all_results.md"))
    write_interpretability_summary(os.path.join(OUT_DIR, "interpretability_summary.md"))
    print_terminal_summary(rows)


if __name__ == "__main__":
    main()
