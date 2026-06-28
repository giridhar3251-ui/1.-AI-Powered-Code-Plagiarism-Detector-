#!/usr/bin/env python3
"""
detect.py
---------
Main CLI entry point for the plagiarism detector.

Usage:
    python detect.py file1.py file2.py
    python detect.py file1.py file2.py --report report.html
    python detect.py --batch samples/  (compares all .py files pairwise)

This script wires together:
    normalizer.py  -> parse + normalize ASTs, extract functions/features
    matcher.py     -> sequence similarity + feature similarity -> combined score
    semantic.py    -> optional ML embedding similarity (graceful fallback)
    report.py      -> HTML report generation
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalizer import normalize_source
from matcher import match_all_functions, whole_file_score
import semantic
from semantic import semantic_similarity
from report import generate_report


def analyze_file(path: str):
    """Reads and normalizes a single file. Returns (whole_seq, functions) or
    raises a clear error if the file has a syntax error (common with student
    submissions -- we don't want the whole batch to crash on one bad file)."""
    with open(path, "r") as f:
        source = f.read()
    try:
        return normalize_source(source)
    except SyntaxError as e:
        raise SyntaxError(f"Could not parse {path}: {e}")


def compare_pair(path_a: str, path_b: str, report_path: str = None, verbose: bool = True):
    seq_a, funcs_a = analyze_file(path_a)
    seq_b, funcs_b = analyze_file(path_b)

    file_score = whole_file_score(seq_a, seq_b)
    func_matches = match_all_functions(funcs_a, funcs_b)

    # Compute semantic similarity for the top N function pairs only --
    # embedding every pair would be wasteful; we only care about the
    # pairs that already look structurally promising, plus we cap total
    # calls to keep runtime reasonable on large batches.
    semantic_results = {}
    top_pairs = func_matches[:10]
    func_lookup_a = {f.name: f for f in funcs_a}
    func_lookup_b = {f.name: f for f in funcs_b}
    for m in top_pairs:
        fa = func_lookup_a[m["func_a"]]
        fb = func_lookup_b[m["func_b"]]
        sim = semantic_similarity(fa.source_normalized, fb.source_normalized)
        semantic_results[(m["func_a"], m["func_b"])] = sim

    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing: {path_a}  vs  {path_b}")
        print(f"{'='*60}")
        print(f"Overall file similarity: {file_score['final_score']}%  -> {file_score['verdict']}")
        if not semantic.EMBEDDINGS_AVAILABLE:
            print("(Note: semantic/ML similarity layer unavailable -- install "
                  "'sentence-transformers' and ensure internet access to enable it. "
                  "Continuing with structural similarity only.)")
        print(f"\nTop function matches:")
        for m in func_matches[:10]:
            sem = semantic_results.get((m["func_a"], m["func_b"]))
            sem_str = f", semantic={sem*100:.1f}%" if sem is not None else ""
            print(f"  {m['func_a']}() <-> {m['func_b']}()  "
                  f"score={m['final_score']}% "
                  f"(seq={m['sequence_similarity']}%, feat={m['feature_similarity']}%{sem_str})  "
                  f"-> {m['verdict']}")

    if report_path:
        out = generate_report(path_a, path_b, file_score, func_matches, semantic_results, report_path)
        if verbose:
            print(f"\nHTML report written to: {out}")

    return {
        "file_a": path_a,
        "file_b": path_b,
        "file_score": file_score,
        "function_matches": func_matches,
    }


def batch_compare(directory: str, report_dir: str = None):
    """Compares every pair of .py files in a directory (all-pairs), useful
    for an instructor checking a whole class's submissions at once."""
    files = sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".py")
    )
    if len(files) < 2:
        print(f"Need at least 2 .py files in {directory} to compare.")
        return []

    results = []
    print(f"Found {len(files)} files. Running {len(files) * (len(files)-1) // 2} pairwise comparisons...\n")
    for path_a, path_b in itertools.combinations(files, 2):
        report_path = None
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
            name_a = os.path.splitext(os.path.basename(path_a))[0]
            name_b = os.path.splitext(os.path.basename(path_b))[0]
            report_path = os.path.join(report_dir, f"{name_a}_vs_{name_b}.html")
        try:
            result = compare_pair(path_a, path_b, report_path=report_path, verbose=False)
            score = result["file_score"]["final_score"]
            print(f"{os.path.basename(path_a):30s} vs {os.path.basename(path_b):30s}  "
                  f"score={score:5.1f}%  {result['file_score']['verdict']}")
            results.append(result)
        except SyntaxError as e:
            print(f"  [skipped] {e}")

    results.sort(key=lambda r: r["file_score"]["final_score"], reverse=True)
    print("\nTop flagged pairs (highest similarity first):")
    for r in results[:5]:
        print(f"  {os.path.basename(r['file_a'])} <-> {os.path.basename(r['file_b'])}: "
              f"{r['file_score']['final_score']}%")
    return results


def main():
    parser = argparse.ArgumentParser(description="AI-powered code plagiarism detector")
    parser.add_argument("file_a", nargs="?", help="First Python file")
    parser.add_argument("file_b", nargs="?", help="Second Python file")
    parser.add_argument("--report", help="Path to write HTML report for a single comparison")
    parser.add_argument("--batch", help="Directory of .py files to compare all-pairs")
    parser.add_argument("--report-dir", help="Directory to write HTML reports for batch mode")
    args = parser.parse_args()

    if args.batch:
        batch_compare(args.batch, report_dir=args.report_dir)
    elif args.file_a and args.file_b:
        compare_pair(args.file_a, args.file_b, report_path=args.report)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
