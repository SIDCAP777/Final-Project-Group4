# ============================================================
# glove_embeddings.py
# Builds vocabulary from training data and creates an embedding
# matrix initialized with pretrained GloVe vectors.
#
# Functions:
#   - build_vocab: count tokens in training corpus, keep top-K
#   - load_glove: read GloVe file into a {word: vector} dict
#   - build_embedding_matrix: create (vocab_size, dim) matrix where
#       row i = GloVe vector for vocab token with id i (random if not in GloVe)
#   - texts_to_sequences: convert tokenized texts to padded integer arrays
# ============================================================

import os
import numpy as np
from collections import Counter
from tqdm import tqdm


# Special tokens: index 0 is padding, index 1 is unknown
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def build_vocab(token_lists, max_vocab_size: int = 20000) -> dict:
    # Count frequency of every token across all training documents
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    # Keep only the top-K most common tokens (rarer ones become <unk>)
    most_common = counter.most_common(max_vocab_size - 2)  # -2 for <pad> and <unk>

    # Build word -> id mapping. <pad>=0, <unk>=1, then frequent words follow.
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in most_common:
        vocab[word] = len(vocab)

    print(f"[glove] Built vocab of {len(vocab)} tokens "
          f"(from {len(counter)} unique tokens in corpus)")
    return vocab


def load_glove(glove_path: str, embedding_dim: int = 100) -> dict:
    # Load GloVe file into memory as {word: numpy vector}
    print(f"[glove] Loading GloVe from {glove_path}...")
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=400000, desc="Loading GloVe"):
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            # Sanity check: every line should have exactly `embedding_dim` numbers
            if len(vec) == embedding_dim:
                embeddings[word] = vec
    print(f"[glove] Loaded {len(embeddings)} GloVe vectors")
    return embeddings


def build_embedding_matrix(vocab: dict, glove_dict: dict,
                           embedding_dim: int = 100, seed: int = 42) -> np.ndarray:
    # Create matrix shape (vocab_size, embedding_dim)
    # Row i = GloVe vector for the word with id i, or a small random vector if not in GloVe
    rng = np.random.RandomState(seed)
    vocab_size = len(vocab)

    # Initialize with small uniform random values (typical for embedding init)
    matrix = rng.uniform(-0.05, 0.05, size=(vocab_size, embedding_dim)).astype(np.float32)

    # Pad token row should be all zeros (so it contributes nothing during training)
    matrix[PAD_IDX] = np.zeros(embedding_dim, dtype=np.float32)

    # Fill rows for tokens that exist in GloVe
    found = 0
    for word, idx in vocab.items():
        if word in glove_dict:
            matrix[idx] = glove_dict[word]
            found += 1

    coverage = 100.0 * found / vocab_size
    print(f"[glove] Embedding matrix shape: {matrix.shape}")
    print(f"[glove] GloVe coverage: {found}/{vocab_size} tokens ({coverage:.1f}%)")
    return matrix


def texts_to_sequences(token_lists, vocab: dict, max_seq_length: int = 100) -> np.ndarray:
    # Convert each list of tokens into a fixed-length array of integer ids
    # Truncate longer texts; pad shorter ones with PAD_IDX
    n = len(token_lists)
    sequences = np.full((n, max_seq_length), PAD_IDX, dtype=np.int64)

    for i, tokens in enumerate(token_lists):
        # Truncate if too long (keep the first max_seq_length tokens)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        # Map each token to its vocab id, falling back to UNK for OOV tokens
        ids = [vocab.get(tok, UNK_IDX) for tok in tokens]
        sequences[i, :len(ids)] = ids

    return sequences
