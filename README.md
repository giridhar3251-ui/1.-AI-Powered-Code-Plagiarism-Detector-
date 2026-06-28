# AI-Powered Code Plagiarism Detector

Detects code plagiarism in Python submissions by analyzing **logic and
structure**, not just text. Catches copying even when variable names are
renamed, formatting is changed, or the control flow is rewritten (e.g. a
loop rewritten as recursion).

## Why this is different from a text-diff tool

A naive plagiarism checker (or `diff`) compares raw text, so it's trivially
defeated by renaming variables or reformatting code. This tool instead:

1. **Parses code into an Abstract Syntax Tree (AST)** — the structural
   representation a compiler would use — rather than treating code as text.
2. **Normalizes** the AST: variable names, function names, and literal
   values are replaced with placeholders, so `def add(x, y): return x+y` and
   `def sum_two(a, b): return a+b` normalize to the *same* structure.
3. **Compares structure**, using sequence alignment over the flattened AST
   (same family of algorithm used by MOSS/JPlag) combined with a
   complexity/shape profile (branch count, nesting depth, cyclomatic
   complexity).
4. **Compares semantics** using ML sentence embeddings of the normalized
   code, which catches cases where the *logic* is identical but the
   *structure* differs (e.g. iterative vs. recursive Fibonacci).
5. **Reports** function-level matches with a combined score and verdict, not
   just one whole-file number — so a student can't bury one copied function
   inside an otherwise-original file.

## Project structure

```
plagiarism_detector/
├── detector/
│   ├── normalizer.py   # AST parsing + normalization (Phase 1)
│   ├── matcher.py       # Structural similarity scoring (Phase 2)
│   ├── semantic.py      # ML embedding similarity (Phase 3, optional)
│   ├── report.py         # HTML report generation (Phase 4)
│   └── detect.py          # CLI entry point, wires everything together
├── samples/               # Example submissions for testing/demo
│   ├── student_A.py               # "Original" submission
│   ├── student_B_renamed.py       # Plagiarized: renamed variables
│   ├── student_C_restructured.py  # Plagiarized: restructured logic (recursion)
│   └── student_D_original.py      # Genuinely independent code (negative control)
├── reports/                # Generated HTML reports land here
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

The ML/semantic layer (`sentence-transformers`) is **optional** — if it
isn't installed, or if the model can't be downloaded (no internet access),
the tool automatically falls back to structural-only comparison and prints
a notice. Nothing crashes either way.

## Usage

**Compare two files:**
```bash
python detector/detect.py samples/student_A.py samples/student_B_renamed.py
```

**Compare two files and generate an HTML report:**
```bash
python detector/detect.py samples/student_A.py samples/student_B_renamed.py --report reports/result.html
```

**Batch mode — compare every pair of `.py` files in a folder** (e.g. an
instructor checking a whole class's submissions at once):
```bash
python detector/detect.py --batch samples/ --report-dir reports/batch
```

## How the scoring works

Each function pair gets three independent signals, combined into one score:

| Signal | What it measures | Catches |
|---|---|---|
| **Sequence similarity** | Ratcliff-Obershelp ratio over the flattened, normalized AST node-type sequence | Renamed variables, reformatted code, literal value changes, comment changes |
| **Feature similarity** | Distance between complexity profiles (branch count, loop count, nesting depth, cyclomatic complexity) | Code that's been reordered or lightly modified but keeps the same shape |
| **Semantic similarity** (optional, ML) | Cosine similarity between sentence-embedding vectors of normalized source | Logic-equivalent code with a genuinely different structure (loop ↔ recursion) |

Final score = `0.65 × sequence_similarity + 0.35 × feature_similarity`
(semantic similarity is reported alongside as a third signal for human
review, since it has higher variance and is best used as corroborating
evidence rather than folded into the primary score).

**Score interpretation:**
| Score | Verdict |
|---|---|
| 85-100% | High similarity — manual review strongly recommended |
| 65-84% | Moderate similarity — review recommended |
| 40-64% | Low similarity — likely coincidental |
| 0-39% | Minimal similarity |

## Demonstrated results (from the included samples)

Running batch mode on the four sample files produces:

| Pair | Score | Verdict | What it demonstrates |
|---|---|---|---|
| A vs B (renamed) | 100.0% | HIGH | Pure renaming is caught perfectly |
| A vs C (restructured) | 79.1% | MODERATE | Partial restructuring still flagged; the one truly rewritten function (fibonacci, recursive vs. iterative) scores lower on structure alone — this is exactly the gap the semantic/ML layer is designed to close |
| A vs D (independent) | 33.3% | MINIMAL | No false positive on genuinely different code |

This validates the core thesis of the project: **structural comparison
alone catches most plagiarism, but logic-preserving restructuring needs a
semantic signal to catch reliably** — which is why the project layers both.

## Known limitations (worth mentioning in a report/presentation)

- Single-file, single-language (Python) scope — no cross-file or
  cross-language detection.
- The feature/structure weights (0.65/0.35) were chosen heuristically for
  this demo, not tuned on a labeled dataset; a real deployment would want
  to calibrate these against known plagiarism/non-plagiarism pairs.
- Very short functions (a few lines) can show coincidentally high
  similarity since there's less structure to differentiate on — the tool
  is most reliable on functions with non-trivial logic.
- The semantic layer requires internet access on first run to download the
  embedding model; it is designed to fail gracefully (not crash) if
  unavailable, falling back to structural-only scoring.

## Possible extensions

- Multi-language support via `tree-sitter` instead of Python's `ast`.
- Tune scoring weights against a labeled dataset of known plagiarism cases.
- Add a web UI (FastAPI) for instructors to upload a batch of submissions.
- Replace the general sentence-embedding model with a code-specialized one
  (e.g. CodeBERT / GraphCodeBERT) for higher semantic precision.
