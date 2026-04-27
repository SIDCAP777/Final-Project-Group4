# ============================================================
# roberta.py
# Wrapper around HuggingFace RoBERTa for sequence classification.
#
# RoBERTa is a more thoroughly pretrained variant of BERT (more
# data, longer training, dynamic masking, no NSP). It typically
# beats DistilBERT by 1-2 points on text classification at the
# cost of ~2x training time (125M params vs 66M).
#
# RobertaForSequenceClassification adds a small classification
# head on top of the [CLS] token's final hidden state.
# ============================================================
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
)


def build_roberta(config: dict, num_classes: int = 2):
    # Load tokenizer + model from HuggingFace.
    # First time this runs it downloads ~500MB to ~/.cache/huggingface/.
    # Subsequent runs reuse the cached weights instantly.
    model_name = config["roberta"]["model_name"]
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
    )
    print(f"[roberta] Loaded {model_name}")
    print(f"[roberta] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model
