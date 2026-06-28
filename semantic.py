"""
semantic.py
-----------
Adds a third, "ML-powered" similarity signal on top of the structural
matching in matcher.py: semantic code embeddings.

Why this matters: sequence/feature similarity (matcher.py) compares the
SHAPE of the code. But two functions can implement the same logic with a
genuinely different shape -- e.g. one uses a for-loop, the other uses
recursion, to compute the same sum. A human grader would call that
plagiarism (or at least "suspiciously identical logic"); pure AST-shape
comparison might score it low. Embedding the *normalized source text* with
a pretrained code-aware language model and comparing vector similarity
catches this case.

This module is intentionally optional / fail-soft: if `sentence-transformers`
isn't installed or a model can't be downloaded (e.g. no internet on a grading
machine), the rest of the detector still works fine using sequence + feature
similarity alone. This is checked once at import time via `EMBEDDINGS_AVAILABLE`.
"""

import math

EMBEDDINGS_AVAILABLE = False
_model = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    np = None


def _get_model():
    """
    Lazily loads the embedding model on first use (avoids paying model
    load time if the caller never asks for semantic similarity).

    Uses a small general-purpose sentence embedding model. For a more
    code-specialized result you could swap this for "microsoft/codebert-base"
    via the `transformers` library directly, but that requires more setup
    (mean-pooling token embeddings yourself). This model is chosen because
    it installs/runs quickly and still captures meaningful similarity
    between code snippets that share structure and naming patterns.
    """
    global _model, EMBEDDINGS_AVAILABLE
    if _model is None:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            # Model couldn't be downloaded/loaded (e.g. no internet access,
            # blocked registry). Disable the layer rather than crashing --
            # the rest of the detector still works fine without it.
            EMBEDDINGS_AVAILABLE = False
            _model = False
    return _model


def cosine_similarity(vec_a, vec_b) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_similarity(source_a: str, source_b: str) -> float:
    """
    Returns a 0-1 similarity score between two normalized source snippets
    using sentence embeddings. Returns None if embeddings aren't available
    so callers can distinguish "scored 0" from "couldn't score."
    """
    if not EMBEDDINGS_AVAILABLE:
        return None
    if not source_a.strip() or not source_b.strip():
        return None

    model = _get_model()
    if not model:
        return None
    emb_a, emb_b = model.encode([source_a, source_b])
    sim = cosine_similarity(emb_a, emb_b)
    # Cosine similarity for this model is typically in [~ -0.1, 1.0] for
    # unrelated-to-identical text; clip defensively for display purposes.
    return max(0.0, min(1.0, sim))
