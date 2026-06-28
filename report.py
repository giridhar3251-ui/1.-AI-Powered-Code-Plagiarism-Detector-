"""
report.py
---------
Generates a self-contained HTML report summarizing the comparison between
two (or more) Python files: overall score, per-function breakdown, and
side-by-side normalized source for the top matches.
"""

from datetime import datetime
import html


def _score_color(score: float) -> str:
    if score >= 85:
        return "#d93025"  # red
    elif score >= 65:
        return "#f29900"  # orange
    elif score >= 40:
        return "#e8b400"  # yellow
    return "#1e8e3e"      # green


def _escape(s: str) -> str:
    return html.escape(s or "")


def generate_report(file_a: str, file_b: str, whole_file_result: dict,
                     function_matches: list, semantic_results: dict,
                     output_path: str) -> str:
    """
    Builds an HTML report and writes it to output_path.
    `semantic_results` maps (func_a_name, func_b_name) -> semantic score (0-1) or None.
    """
    top_score = whole_file_result["final_score"]
    top_color = _score_color(top_score)

    rows = []
    for m in function_matches[:25]:  # cap to top 25 matches to keep report readable
        sem_key = (m["func_a"], m["func_b"])
        sem_score = semantic_results.get(sem_key)
        sem_display = f"{sem_score * 100:.1f}%" if sem_score is not None else "N/A"
        color = _score_color(m["final_score"])
        rows.append(f"""
        <tr>
          <td>{_escape(m['func_a'])} <span class="lineno">(line {m['lineno_a']})</span></td>
          <td>{_escape(m['func_b'])} <span class="lineno">(line {m['lineno_b']})</span></td>
          <td>{m['sequence_similarity']}%</td>
          <td>{m['feature_similarity']}%</td>
          <td>{sem_display}</td>
          <td><strong style="color:{color}">{m['final_score']}%</strong></td>
          <td class="verdict">{_escape(m['verdict'])}</td>
        </tr>
        """)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Plagiarism Detection Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
         margin: 40px auto; padding: 0 20px; color: #202124; background: #fafafa; }}
  h1 {{ font-size: 1.5rem; }}
  .meta {{ color: #5f6368; font-size: 0.9rem; margin-bottom: 24px; }}
  .summary {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px;
              padding: 20px; margin-bottom: 28px; }}
  .summary .score {{ font-size: 2.2rem; font-weight: 700; color: {top_color}; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 0.88rem; }}
  th {{ background: #f1f3f4; font-weight: 600; }}
  .lineno {{ color: #9aa0a6; font-size: 0.78rem; }}
  .verdict {{ font-size: 0.8rem; color: #5f6368; }}
  footer {{ margin-top: 30px; color: #9aa0a6; font-size: 0.78rem; }}
</style>
</head>
<body>
  <h1>Code Plagiarism Detection Report</h1>
  <div class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

  <div class="summary">
    <div>Comparing <strong>{_escape(file_a)}</strong> vs <strong>{_escape(file_b)}</strong></div>
    <div class="score">{top_score}%</div>
    <div>{_escape(whole_file_result['verdict'])}</div>
  </div>

  <h2>Function-level matches</h2>
  <table>
    <tr>
      <th>Function (File A)</th>
      <th>Function (File B)</th>
      <th>Structure Sim.</th>
      <th>Feature Sim.</th>
      <th>Semantic Sim.</th>
      <th>Final Score</th>
      <th>Verdict</th>
    </tr>
    {''.join(rows) if rows else '<tr><td colspan="7">No functions found to compare.</td></tr>'}
  </table>

  <footer>
    Structure Sim. = flattened AST node-sequence match · Feature Sim. = complexity/shape profile match ·
    Semantic Sim. = embedding-based similarity (ML layer, may be N/A if model unavailable)
  </footer>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_doc)
    return output_path
