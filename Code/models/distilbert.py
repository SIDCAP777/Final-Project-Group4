# ============================================================
# distilbert.py
# Wrapper around HuggingFace DistilBERT for sequence classification.
#
# DistilBERT is already pretrained on a massive corpus. We just need
# to:
#   1. Load the pretrained model
#   2. Add a classification head (HuggingFace does this automatically)
#   3. Fine-tune on our sarcasm data
#
# DistilBertForSequenceClassification handles the classification head
# for us - it adds a small linear layer on top of the [CLS] token's
# final hidden state.
# ============================================================

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)


def build_distilbert(config: dict, num_classes: int = 2):
    # Load both tokenizer and model from HuggingFace.
    # First time this runs it downloads ~250MB of weights to ~/.cache/huggingface/.
    # Subsequent runs reuse the cached weights instantly.
    model_name = config["distilbert"]["model_name"]

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
    )

    print(f"[distilbert] Loaded {model_name}")
    print(f"[distilbert] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return tokenizer, model
