"""
matcher.py
----------
Computes similarity scores between normalized code representations.

Two complementary signals are combined:
  1. Sequence similarity  -- how similar are the flattened AST node-type
     sequences (catches structural copying, even with renamed vars/loop-
     vs-while swaps will reduce this score, which is intentional: it's a
     SYNTACTIC structure signal).
  2. Feature similarity   -- how similar are complexity/shape features
     (branch count, nesting depth, cyclomatic complexity, etc). This catches
     logic-equivalent code that was restructured (e.g. recursion rewritten
     as iteration) since the *complexity profile* often still lines up even
     when the exact AST shape doesn't.

The final score is a weighted blend of both.
"""

from difflib import SequenceMatcher
import math

# Weights for combining signals into one final score.
SEQUENCE_WEIGHT = 0.65
FEATURE_WEIGHT = 0.35


def sequence_similarity(seq_a: str, seq_b: str) -> float:
    """
    Ratio in [0, 1] of how similar two flattened node-type sequences are,
    using difflib's SequenceMatcher (Ratcliff-Obershelp algorithm).
    This is robust to insertions/deletions, unlike naive position-by-
    position comparison, so reordering unrelated statements only partially
    affects the score rather than breaking the match entirely.
    """
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    a_tokens = seq_a.split(",")
    b_tokens = seq_b.split(",")
    return SequenceMatcher(None, a_tokens, b_tokens).ratio()


def feature_similarity(features_a: dict, features_b: dict) -> float:
    """
    Compares two feature dicts (from normalizer.extract_features) using
    normalized distance per feature, averaged. Returns a value in [0, 1].
    """
    keys = set(features_a.keys()) | set(features_b.keys())
    if not keys:
        return 1.0

    sims = []
    for key in keys:
        a = features_a.get(key, 0)
        b = features_b.get(key, 0)
        max_val = max(a, b)
        if max_val == 0:
            sims.append(1.0)  # both zero -> identical on this feature
        else:
            sims.append(1 - abs(a - b) / max_val)
    return sum(sims) / len(sims)


def combined_similarity(func_a, func_b) -> dict:
    """
    Takes two NormalizedFunction objects and returns a dict with the
    individual signal scores plus a final weighted score (0-100 scale,
    easier to read in reports than a 0-1 float).
    """
    seq_sim = sequence_similarity(func_a.node_sequence, func_b.node_sequence)
    feat_sim = feature_similarity(func_a.features, func_b.features)
    final = SEQUENCE_WEIGHT * seq_sim + FEATURE_WEIGHT * feat_sim

    return {
        "sequence_similarity": round(seq_sim * 100, 1),
        "feature_similarity": round(feat_sim * 100, 1),
        "final_score": round(final * 100, 1),
    }


def verdict_for_score(score: float) -> str:
    """Maps a 0-100 score to a human-readable verdict label."""
    if score >= 85:
        return "HIGH SIMILARITY - manual review strongly recommended"
    elif score >= 65:
        return "MODERATE SIMILARITY - review recommended"
    elif score >= 40:
        return "LOW SIMILARITY - likely coincidental"
    else:
        return "MINIMAL SIMILARITY"


def match_all_functions(functions_a: list, functions_b: list) -> list:
    """
    Compares every function in file A against every function in file B
    (all-pairs), returning a sorted list of match dicts, highest score
    first. This lets the report show "function foo() in A matches bar()
    in B at 91%" even when comparing whole files that differ elsewhere.
    """
    results = []
    for fa in functions_a:
        for fb in functions_b:
            scores = combined_similarity(fa, fb)
            results.append(
                {
                    "func_a": fa.name,
                    "func_b": fb.name,
                    "lineno_a": fa.lineno,
                    "lineno_b": fb.lineno,
                    **scores,
                    "verdict": verdict_for_score(scores["final_score"]),
                }
            )
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results


def whole_file_score(seq_a: str, seq_b: str) -> dict:
    """Whole-file-level score, useful as a quick top-line summary."""
    seq_sim = sequence_similarity(seq_a, seq_b)
    score = round(seq_sim * 100, 1)
    return {"final_score": score, "verdict": verdict_for_score(score)}
