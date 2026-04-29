# ============================================================
# app.py
# Side-by-side sarcasm detection demo: RoBERTa on Twitter vs RoBERTa on SARC.
# Same input, two models, two LIME explanations.
#
# Run from project root:
#   streamlit run Code/streamlit_demo/app.py
# Default port: 8501.
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import numpy as np
import pandas as pd
import streamlit as st
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Code"))

# ---------- Constants ----------
MODELS = {
    "twitter": {
        "dir": os.path.join(PROJECT_ROOT, "saved_models", "roberta_twitter_no_hashtags"),
        "label": "RoBERTa on Twitter",
        "test_acc": 0.8722,
        "test_f1": 0.8716,
        "trained_on": "34,552 tweets",
    },
    "sarc": {
        "dir": os.path.join(PROJECT_ROOT, "saved_models", "roberta_sarc"),
        "label": "RoBERTa on SARC",
        "test_acc": 0.7987,
        "test_f1": 0.7987,
        "trained_on": "500K Reddit comments",
    },
}
MAX_SEQ_LENGTH = 256
LIME_NUM_SAMPLES = 300
LIME_NUM_FEATURES = 12
CLASS_NAMES = ["Not Sarcasm", "Sarcasm"]

st.set_page_config(
    page_title="Sarcasm Detection Demo",
    layout="wide",
)


# ---------- Cached model load (both models loaded once) ----------
@st.cache_resource(show_spinner="Loading both RoBERTa models (one time, ~15 sec)...")
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for key, cfg in MODELS.items():
        tok = AutoTokenizer.from_pretrained(cfg["dir"])
        mdl = AutoModelForSequenceClassification.from_pretrained(cfg["dir"]).to(device)
        mdl.eval()
        loaded[key] = {"tokenizer": tok, "model": mdl}
    return loaded, device


# ---------- Inference ----------
def predict_proba(texts, tokenizer, model, device):
    if isinstance(texts, str):
        texts = [texts]
    all_probs = []
    for i in range(0, len(texts), 64):
        batch = list(texts[i:i + 64])
        enc = tokenizer(batch, padding="max_length", truncation=True,
                        max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def run_lime(text, tokenizer, model, device):
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, bow=False, random_state=42)
    def classifier_fn(texts):
        return predict_proba(texts, tokenizer, model, device)
    return explainer.explain_instance(
        text_instance=text,
        classifier_fn=classifier_fn,
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
        labels=(0, 1),
    )


def render_highlighted_text(exp, text):
    word_weights = dict(exp.as_list(label=1))
    max_abs = max((abs(v) for v in word_weights.values()), default=1.0) or 1.0

    html_parts = []
    for word in text.split():
        clean = word.strip(".,!?;:\"'()[]")
        weight = word_weights.get(clean, 0.0)
        intensity = min(abs(weight) / max_abs, 1.0)
        if weight > 0.02:
            alpha = int(intensity * 200) + 30
            color = f"rgba(220, 50, 50, {alpha / 255:.2f})"
            html_parts.append(
                f'<span style="background:{color}; padding:2px 4px; border-radius:3px; margin:1px; display:inline-block;" '
                f'title="{clean}: +{weight:.3f}">{word}</span>'
            )
        elif weight < -0.02:
            alpha = int(intensity * 200) + 30
            color = f"rgba(50, 100, 220, {alpha / 255:.2f})"
            html_parts.append(
                f'<span style="background:{color}; padding:2px 4px; border-radius:3px; margin:1px; display:inline-block;" '
                f'title="{clean}: {weight:.3f}">{word}</span>'
            )
        else:
            html_parts.append(f'<span style="margin:1px; display:inline-block;">{word}</span>')
    return " ".join(html_parts)


def render_model_panel(col, model_key, text, models_dict, device):
    cfg = MODELS[model_key]
    tok = models_dict[model_key]["tokenizer"]
    mdl = models_dict[model_key]["model"]

    with col:
        st.markdown(f"### {cfg['label']}")
        st.caption(f"Test accuracy {cfg['test_acc']*100:.2f}%   Test F1 {cfg['test_f1']*100:.2f}%   Trained on {cfg['trained_on']}")

        with st.spinner(f"Predicting with {model_key}..."):
            probs = predict_proba(text, tok, mdl, device)[0]
        pred_idx = int(np.argmax(probs))

        if pred_idx == 1:
            st.error(f"**Predicted: SARCASM** ({probs[1]*100:.1f}%)")
        else:
            st.success(f"**Predicted: NOT SARCASM** ({probs[0]*100:.1f}%)")

        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": [float(probs[0]), float(probs[1])],
        })
        st.dataframe(
            prob_df, hide_index=True, use_container_width=True,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability", format="%.3f", min_value=0, max_value=1
                ),
            },
        )

        with st.spinner("LIME..."):
            exp = run_lime(text, tok, mdl, device)
        word_weights = exp.as_list(label=1)
        sarc_words = [(w, v) for w, v in word_weights if v > 0]
        not_words = [(w, v) for w, v in word_weights if v < 0]

        st.markdown("**Pushing toward SARCASM**")
        if sarc_words:
            st.dataframe(
                pd.DataFrame(sarc_words, columns=["Word", "Weight"]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("_no words contributed strongly_")

        st.markdown("**Pushing toward NOT SARCASM**")
        if not_words:
            st.dataframe(
                pd.DataFrame(not_words, columns=["Word", "Weight"]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("_no words contributed strongly_")

        st.markdown("**Highlighted text**")
        st.markdown(render_highlighted_text(exp, text), unsafe_allow_html=True)

        return pred_idx, probs


# ---------- UI ----------
st.title("Sarcasm Detection: Twitter vs SARC")
st.caption(
    "Two RoBERTa models, same architecture, different training data. "
    "Each model reflects what its training corpus considers sarcasm."
)

with st.form("input_form", clear_on_submit=False):
    text = st.text_area(
        "Enter text",
        value="",
        height=80,
        placeholder="Type a sentence and press Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux) to run both models",
    )
    submitted = st.form_submit_button("Run both models")

if submitted and text.strip():
    models_dict, device = load_all_models()

    st.markdown("---")
    col_tw, col_sarc = st.columns(2)
    pred_tw, probs_tw = render_model_panel(col_tw, "twitter", text, models_dict, device)
    pred_sarc, probs_sarc = render_model_panel(col_sarc, "sarc", text, models_dict, device)

    # Comparison callout
    st.markdown("---")
    if pred_tw == pred_sarc:
        if pred_tw == 1:
            st.success(f"### Both models agree: SARCASM (Twitter {probs_tw[1]*100:.1f}%, SARC {probs_sarc[1]*100:.1f}%)")
        else:
            st.success(f"### Both models agree: NOT SARCASM (Twitter {probs_tw[0]*100:.1f}%, SARC {probs_sarc[0]*100:.1f}%)")
    else:
        tw_label = "SARCASM" if pred_tw == 1 else "NOT SARCASM"
        sarc_label = "SARCASM" if pred_sarc == 1 else "NOT SARCASM"
        st.warning(
            f"### Models disagree\n\n"
            f"**Twitter** predicts **{tw_label}** ({max(probs_tw)*100:.1f}%). "
            f"**SARC** predicts **{sarc_label}** ({max(probs_sarc)*100:.1f}%). "
            f"The two training corpora taught different definitions of sarcasm."
        )
