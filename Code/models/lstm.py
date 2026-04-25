# ============================================================
# lstm.py
# Bidirectional LSTM classifier for sarcasm detection.
#
# Architecture:
#   embedding (GloVe-init) -> BiLSTM -> dropout -> linear -> 2 classes
#
# Key design choices:
#   - Embedding layer initialized with pretrained GloVe vectors
#   - Pad token (id=0) is masked so it doesn't affect the loss
#   - Bidirectional LSTM reads text both forward and backward,
#     final representation = concat of last-forward + first-backward
#   - Dropout applied between LSTM output and the classifier
# ============================================================

import torch
import torch.nn as nn


class SarcasmLSTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 num_classes: int = 2,
                 pad_idx: int = 0,
                 pretrained_embeddings=None,
                 freeze_embeddings: bool = False):
        super().__init__()

        # Embedding layer. padding_idx tells PyTorch to keep this row at zero
        # (gradient never flows through it, so <pad> stays uninformative)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        # Initialize embedding weights with GloVe vectors if provided
        if pretrained_embeddings is not None:
            # Copy the matrix into the embedding layer's weight tensor
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Optionally freeze (don't update during training)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # Bidirectional LSTM
        # batch_first=True means input shape is (batch, seq_len, embedding_dim)
        # dropout argument applies BETWEEN stacked layers (only when num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout after LSTM, before the classifier
        self.dropout = nn.Dropout(dropout)

        # If bidirectional, the LSTM output is 2 * hidden_dim (forward + backward concat)
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Final classifier: linear layer mapping LSTM output -> 2 class logits
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids shape: (batch, seq_len)

        # Look up embeddings -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # Run through LSTM. Returns (output, (h_n, c_n)).
        # output shape: (batch, seq_len, hidden_dim * num_directions)
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        lstm_out, (h_n, _) = self.lstm(embedded)

        # Strategy: take the last hidden state from BOTH directions of the TOP layer.
        # h_n is organized as (layer_0_fwd, layer_0_bwd, layer_1_fwd, layer_1_bwd, ...)
        # So the last 2 entries are the top-layer forward + backward states.
        if self.lstm.bidirectional:
            # Concatenate the top forward and top backward final states
            forward_last = h_n[-2]   # (batch, hidden_dim)
            backward_last = h_n[-1]  # (batch, hidden_dim)
            sentence_repr = torch.cat([forward_last, backward_last], dim=1)  # (batch, 2*hidden_dim)
        else:
            sentence_repr = h_n[-1]  # (batch, hidden_dim)

        # Apply dropout, then classify
        sentence_repr = self.dropout(sentence_repr)
        logits = self.classifier(sentence_repr)  # (batch, num_classes)
        return logits


def build_lstm(config: dict, vocab_size: int, pretrained_embeddings=None) -> SarcasmLSTM:
    # Convenience builder that pulls hyperparameters from config
    cfg = config["lstm"]
    return SarcasmLSTM(
        vocab_size=vocab_size,
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
        num_classes=2,
        pad_idx=0,  # matches PAD_IDX in glove_embeddings.py
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=cfg.get("freeze_embeddings", False),
    )
