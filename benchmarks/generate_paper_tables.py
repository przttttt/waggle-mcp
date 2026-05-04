"""
Generate paper-ready Markdown tables from benchmark results.

Usage:
  python benchmarks/generate_paper_tables.py --output docs/research/tables/
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = _HERE.parent
_RESULTS = _ROOT / "benchmark_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
        return []


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|" + "|".join("---" for _ in headers) + "|"
    data_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + data_lines)


def _fmt(val: str, decimals: int = 3) -> str:
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


# ---------------------------------------------------------------------------
# Table 1: Main results
# ---------------------------------------------------------------------------


def _table_main_results(results_dir: Path) -> str:
    rows = _load_csv(results_dir / "rlm_style_waggle_results.csv")
    if not rows:
        return "_No main results available. Run `rlm_style_waggle_eval.py` first._"

    headers = ["Family", "Scale", "Method", "Score", "Tokens", "Latency (ms)"]
    data: list[list[str]] = []
    for r in rows:
        data.append([
            r.get("benchmark_family", ""),
            r.get("scale_n", ""),
            r.get("method", ""),
            _fmt(r.get("score", "0")),
            r.get("tokens_returned", ""),
            _fmt(r.get("latency_ms", "0"), 0),
        ])
    return _md_table(headers, data)


# ---------------------------------------------------------------------------
# Table 2: Ablations
# ---------------------------------------------------------------------------


def _table_ablations(results_dir: Path) -> str:
    rows = _load_csv(results_dir / "ablation_results.csv")
    if not rows:
        return "_No ablation results available. Run `run_ablation.py` first._"

    headers = ["Family", "Scale", "Variant", "Score", "Δ vs full", "Interpretation"]
    data: list[list[str]] = []
    for r in rows:
        delta = r.get("delta_vs_full", "0")
        try:
            delta_str = f"{float(delta):+.4f}"
        except (ValueError, TypeError):
            delta_str = str(delta)
        data.append([
            r.get("benchmark_family", ""),
            r.get("scale_n", ""),
            r.get("ablation_variant", ""),
            _fmt(r.get("score", "0")),
            delta_str,
            r.get("annotation", ""),
        ])
    return _md_table(headers, data)


# ---------------------------------------------------------------------------
# Table 3: Groq answer-level
# ---------------------------------------------------------------------------


def _table_groq_answer_level(results_dir: Path) -> str:
    # Prefer groq-specific file, fall back to generic
    path = results_dir / "groq_answer_level_results.csv"
    if not path.exists():
        path = results_dir / "answer_level_results.csv"
    rows = _load_csv(path)
    if not rows:
        return "_No answer-level results available. Run `answer_level_eval.py` first._"

    headers = ["Scale", "Method", "Mean F1 ± Std", "Hall. Rate", "Insuff. Rate", "Tokens"]
    # Group by (scale, method) to compute mean/std
    from collections import defaultdict
    import statistics

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("seed", "") == "-1":
            continue  # skip aggregated rows
        key = (r.get("scale_n", ""), r.get("method", ""))
        groups[key].append(r)

    data: list[list[str]] = []
    for (scale, method), group in sorted(groups.items()):
        f1_vals = [float(r.get("final_answer_f1", 0)) for r in group]
        hall_vals = [float(r.get("hallucination_rate", 0)) for r in group]
        insuff_vals = [1.0 if str(r.get("insufficient_context", "False")).lower() == "true" else 0.0 for r in group]
        tokens_vals = [int(r.get("tokens_injected", 0)) for r in group]

        mean_f1 = statistics.mean(f1_vals)
        std_f1 = statistics.stdev(f1_vals) if len(f1_vals) > 1 else 0.0
        mean_hall = statistics.mean(hall_vals)
        mean_insuff = statistics.mean(insuff_vals)
        mean_tokens = int(statistics.mean(tokens_vals))

        data.append([
            scale,
            method,
            f"{mean_f1:.3f} ± {std_f1:.3f}",
            f"{mean_hall:.3f}",
            f"{mean_insuff:.3f}",
            str(mean_tokens),
        ])

    if not data:
        return "_No per-seed answer-level rows found._"
    return _md_table(headers, data)


# ---------------------------------------------------------------------------
# Table 4: ContextReset
# ---------------------------------------------------------------------------


def _table_context_reset(results_dir: Path) -> str:
    partial_dir = results_dir / "partial" / "context_reset"
    rows: list[dict[str, str]] = []
    if partial_dir.exists():
        for csv_path in sorted(partial_dir.glob("*.csv")):
            rows.extend(_load_csv(csv_path))

    if not rows:
        return "_No ContextReset results available. Run `rlm_style_waggle_eval.py --families context_reset` first._"

    headers = ["Method", "Dec. Recall", "Const. Recall", "Next-Step", "Superseded", "Ev. Cov", "Tokens"]
    import json as _json

    data: list[list[str]] = []
    for r in rows:
        notes = r.get("notes", "{}")
        try:
            scoring = _json.loads(notes)
        except Exception:
            scoring = {}
        data.append([
            r.get("method", ""),
            _fmt(scoring.get("decision_recall", r.get("f1", "0"))),
            _fmt(scoring.get("constraint_recall", "0")),
            _fmt(scoring.get("next_step_accuracy", r.get("exact_match", "0"))),
            _fmt(scoring.get("superseded_context_handling", "0")),
            _fmt(r.get("evidence_coverage", "0")),
            r.get("tokens_returned", ""),
        ])
    return _md_table(headers, data)


# ---------------------------------------------------------------------------
# Table 5: Failure claims (hardcoded)
# ---------------------------------------------------------------------------


def _table_failure_claims() -> str:
    headers = ["Claim", "Supported?", "Evidence", "Caveat"]
    rows = [
        [
            "RMCA decomposition improves pairwise",
            "✅ Yes",
            "ablation: no_decomp drops 1.0→0.0",
            "Synthetic data only",
        ],
        [
            "RMCA structured context improves LLM answerability",
            "⚠️ Partial",
            "Groq F1=0.64 vs 0.00 at scale=128",
            "Single scale, single model",
        ],
        [
            "RMCA reduces injected tokens vs raw_context",
            "✅ Yes",
            "S-NIAH: 14% of raw tokens",
            "Synthetic data only",
        ],
        [
            "Graph expansion is load-bearing",
            "❌ Not yet",
            "No delta at scale=128",
            "Need pairwise_hidden_edge",
        ],
        [
            "Conflict resolution is load-bearing",
            "❌ Not yet",
            "No delta at scale=128",
            "Need pairwise_hidden_edge",
        ],
        [
            "RMCA solves ContextReset",
            "❌ Not yet",
            "Score 0.0 in current setup",
            "Scope/query bug being fixed",
        ],
        [
            "Results generalize to real traces",
            "❌ Not yet",
            "Synthetic data only",
            "Need real dataset runs",
        ],
    ]
    return _md_table(headers, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_tables(output_dir: str, results_dir: str = "benchmark_results") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rd = Path(results_dir)

    files: dict[str, str] = {}

    # Table 1: Main results
    t1_path = out / "main_results.md"
    t1_path.write_text(
        "# Main Results\n\n"
        "_From `benchmark_results/rlm_style_waggle_results.csv`_\n\n"
        + _table_main_results(rd)
        + "\n"
    )
    files["main_results"] = str(t1_path)

    # Table 2: Ablations
    t2_path = out / "ablations.md"
    t2_path.write_text(
        "# Ablation Study Results\n\n"
        "_From `benchmark_results/ablation_results.csv`_\n\n"
        + _table_ablations(rd)
        + "\n"
    )
    files["ablations"] = str(t2_path)

    # Table 3: Groq answer-level
    t3_path = out / "groq_answer_level.md"
    t3_path.write_text(
        "# Groq Answer-Level Evaluation\n\n"
        "_From `benchmark_results/groq_answer_level_results.csv` (or `answer_level_results.csv`)_\n\n"
        + _table_groq_answer_level(rd)
        + "\n"
    )
    files["groq_answer_level"] = str(t3_path)

    # Table 4: ContextReset
    t4_path = out / "context_reset.md"
    t4_path.write_text(
        "# ContextReset Results\n\n"
        "_From `benchmark_results/partial/context_reset/*.csv`_\n\n"
        + _table_context_reset(rd)
        + "\n"
    )
    files["context_reset"] = str(t4_path)

    # Table 5: Failure claims
    t5_path = out / "failure_claims.md"
    t5_path.write_text(
        "# Failure Claims Summary\n\n"
        "_Hardcoded claims table — update as evidence accumulates._\n\n"
        + _table_failure_claims()
        + "\n"
    )
    files["failure_claims"] = str(t5_path)

    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready Markdown tables from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        default="docs/research/tables",
        help="Output directory for table files (default: docs/research/tables/)",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark_results",
        help="Directory containing result CSV files (default: benchmark_results/)",
    )
    args = parser.parse_args(argv)

    files = generate_tables(output_dir=args.output, results_dir=args.results_dir)
    print("Paper tables written to:")
    for name, path in files.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
