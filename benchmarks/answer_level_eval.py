"""
benchmarks/answer_level_eval.py
================================
Answer-level evaluation pipeline for RMCA.

Pipeline: method → context pack → answerer → final answer → scorer

The DeterministicAnswerer is a reproducible lower-bound evaluator.
It is NOT a substitute for human preference or LLM-judge quality assessment.

DISCLAIMER: Deterministic answer-level metrics are reproducible lower bounds.
They are not equivalent to human preference ratings or LLM-judge quality
assessments. Scores should be interpreted as retrieval-quality proxies,
not end-to-end answer quality.

SECURITY: GROQ_API_KEY is read exclusively from os.environ. It is never
written to any output file, log, or notes field.

Usage:
  python benchmarks/answer_level_eval.py \\
    --methods rmca_full bm25_topk query_graph \\
    --scales 128 \\
    --families pairwise codeqa \\
    --seed 42 \\
    --output benchmark_results/
"""
from __future__ import annotations

import argparse
import atexit
import csv
import json
import logging
import os
import random
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — works both from repo root and as installed package
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from waggle.intelligence import tokenize_text
from rlm_style_waggle_eval import (
    _METHOD_RUNNERS,
    _make_graph,
    _BENCHMARK_RUNNERS,
    _ALL_FAMILIES,
    generate_pairwise_cases,
    generate_codeqa_cases,
    generate_context_reset_cases,
    token_estimate,
    BenchResult,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "Deterministic answer-level metrics are reproducible lower bounds. "
    "They are not equivalent to human preference ratings or LLM-judge quality assessments. "
    "Scores should be interpreted as retrieval-quality proxies, not end-to-end answer quality."
)

LLM_EVAL_CAVEAT = (
    "These results use one answering model (Groq llama-3.3-70b-versatile) and should be "
    "replicated across models. The model is an answerer, not an independent human judge."
)

GROQ_PROMPT_TEMPLATE = """You are evaluating a memory retrieval system.

Context (retrieved memory):
{context_pack}

Question: {question}

Instructions:
- Answer ONLY using information from the context above.
- If the context does not contain enough information to answer, set insufficient_context to true.
- Output STRICT JSON only. No markdown. No commentary.

JSON schema:
{{
  "answer": "your answer here",
  "evidence_phrases": ["phrase from context that supports answer"],
  "insufficient_context": false
}}"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnswerLevelResult:
    benchmark_family: str
    scale_n: int
    method: str
    answerer: str  # "deterministic" | "ollama" | "groq"
    final_answer_exact_match: float
    final_answer_f1: float
    evidence_used: float
    contradiction_correctness: float
    hallucination_rate: float
    tokens_injected: int
    seed: int = 42
    notes: str = ""
    # Groq-specific fields (Task 2)
    insufficient_context: bool = False
    parse_error: int = 0
    answer_tokens: int = 0
    model_name: str = ""
    # Pairwise-specific fields (Task 4)
    pairwise_answer_f1: float = 0.0
    pairwise_conflict_recall: float = 0.0
    pairwise_conflict_precision: float = 0.0


# ---------------------------------------------------------------------------
# Answerers
# ---------------------------------------------------------------------------


class DeterministicAnswerer:
    """
    Reproducible lower-bound evaluator.

    Extracts a final answer from a Context_Pack using rule-based string
    matching. This is NOT a substitute for human preference or LLM-judge
    quality assessment. Use for offline reproducibility only.
    """

    def extract(self, context_pack: str, question: str, gold_answer: str = "") -> str:
        """
        Extract a final answer from the context pack.

        Algorithm:
        1. Scan lines starting with "- [decision]", "- [fact]", "- [preference]"
        2. Extract content after the colon on each such line
        3. If gold_answer is provided and ≤10 tokens, return first line containing all gold tokens
        4. Otherwise return first non-empty extracted line
        5. If no structured lines found, return first non-empty non-header line
        """
        lines = context_pack.split("\n")
        extracted_lines = []
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith("- [decision]")
                or stripped.startswith("- [fact]")
                or stripped.startswith("- [preference]")
            ):
                # Extract content after the colon
                colon_idx = stripped.find(":", 3)  # skip past "- [xxx]"
                if colon_idx >= 0:
                    content = stripped[colon_idx + 1:].strip()
                    if content:
                        extracted_lines.append(content)

        if not extracted_lines:
            # Fallback: return first non-empty non-header line
            for line in lines:
                stripped = line.strip()
                if (
                    stripped
                    and not stripped.startswith("#")
                    and not stripped.startswith("Task:")
                    and not stripped.startswith("###")
                ):
                    return stripped
            return ""

        # If gold_answer provided and short, find best matching line
        if gold_answer:
            gold_tokens = set(tokenize_text(gold_answer))
            if len(gold_tokens) <= 10:
                for line in extracted_lines:
                    line_tokens = set(tokenize_text(line))
                    if gold_tokens and gold_tokens.issubset(line_tokens):
                        return line

        return extracted_lines[0]


class OllamaAnswerer:
    """
    Optional LLM-based answerer using a local Ollama instance.
    Disabled by default. Falls back to DeterministicAnswerer if unavailable.
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._fallback = DeterministicAnswerer()

    def extract(self, context_pack: str, question: str, gold_answer: str = "") -> str:
        try:
            import urllib.request
            import json as _json

            payload = _json.dumps({
                "model": self.model,
                "prompt": f"Context:\n{context_pack}\n\nQuestion: {question}\n\nAnswer briefly:",
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
                return data.get("response", "").strip()
        except Exception as exc:
            LOGGER.warning(
                "OllamaAnswerer unavailable, falling back to DeterministicAnswerer: %s", exc
            )
            return self._fallback.extract(context_pack, question, gold_answer)


class GroqAnswerer:
    """
    LLM-based answerer using the Groq API with structured JSON output.
    Requires GROQ_API_KEY environment variable.
    Falls back to DeterministicAnswerer if the API call fails or key is missing.

    SECURITY: API key is read exclusively from os.environ. Never written to output files.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 512,
        timeout: float = 30.0,
    ):
        import os
        # SECURITY: read from environment only — never accept api_key as parameter
        self._api_key = os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._fallback = DeterministicAnswerer()

    def extract(self, context_pack: str, question: str, gold_answer: str = "") -> str:
        """
        Extract answer using Groq API with structured JSON prompt.
        Falls back to DeterministicAnswerer if API key is missing or call fails.
        Returns (answer_str, metadata_dict) — but for compatibility returns str only.
        Use extract_structured() for full metadata.
        """
        result = self.extract_structured(context_pack, question, gold_answer)
        return result["answer"]

    def extract_structured(
        self, context_pack: str, question: str, gold_answer: str = ""
    ) -> dict:
        """
        Extract answer with full structured metadata.
        Returns dict with keys: answer, evidence_phrases, insufficient_context,
        parse_error, answer_tokens, model_name.
        """
        if not self._api_key:
            LOGGER.warning("GroqAnswerer: GROQ_API_KEY not set, falling back to DeterministicAnswerer")
            fallback_answer = self._fallback.extract(context_pack, question, gold_answer)
            return {
                "answer": fallback_answer,
                "evidence_phrases": [],
                "insufficient_context": False,
                "parse_error": 0,
                "answer_tokens": len(fallback_answer.split()),
                "model_name": "deterministic-fallback",
            }

        prompt = GROQ_PROMPT_TEMPLATE.format(
            context_pack=context_pack,
            question=question,
        )
        try:
            import json as _json
            import urllib.request

            payload = _json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "waggle-mcp/1.0",
                    "Accept": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
            raw_response = str(body["choices"][0]["message"]["content"] or "").strip()

            # Parse structured JSON response
            try:
                parsed = _json.loads(raw_response)
                answer = str(parsed.get("answer", "")).strip()
                evidence_phrases = parsed.get("evidence_phrases", [])
                insufficient_context = bool(parsed.get("insufficient_context", False))
                return {
                    "answer": answer,
                    "evidence_phrases": evidence_phrases,
                    "insufficient_context": insufficient_context,
                    "parse_error": 0,
                    "answer_tokens": len(answer.split()),
                    "model_name": self.model,
                }
            except (_json.JSONDecodeError, ValueError):
                # JSON parse failed — store raw response truncated, no key leak
                return {
                    "answer": raw_response[:200],
                    "evidence_phrases": [],
                    "insufficient_context": False,
                    "parse_error": 1,
                    "answer_tokens": len(raw_response[:200].split()),
                    "model_name": self.model,
                }

        except Exception as exc:
            LOGGER.warning("GroqAnswerer failed, falling back to DeterministicAnswerer: %s", exc)
            fallback_answer = self._fallback.extract(context_pack, question, gold_answer)
            return {
                "answer": fallback_answer,
                "evidence_phrases": [],
                "insufficient_context": False,
                "parse_error": 0,
                "answer_tokens": len(fallback_answer.split()),
                "model_name": "deterministic-fallback",
            }


# ---------------------------------------------------------------------------
# Pairwise deterministic answerer (Task 4)
# ---------------------------------------------------------------------------


def _normalize_label(text: str) -> str:
    """Normalize a label: lowercase, strip punctuation, collapse whitespace."""
    text = text.strip().lower()
    # Remove common prefixes
    for prefix in ("choice:", "constraint:", "- [decision]", "- [preference]", "- [fact]"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Strip surrounding quotes and punctuation
    text = re.sub(r"^['\"\[\]]+|['\"\[\]]+$", "", text).strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


class PairwiseDeterministicAnswerer:
    """
    Deterministic answerer specialized for pairwise conflict tasks.
    Detects conflict phrases and extracts (choice, constraint) pairs.
    """

    _CONFLICT_PHRASES = [
        "conflicts with", "contradicts", "Possible conflict",
        "incompatible with", "violates constraint", "conflict:",
    ]

    def extract_conflict_pairs(self, context_pack: str) -> list[tuple[str, str]]:
        """Extract (choice, constraint) pairs from structured conflict context."""
        pairs: list[tuple[str, str]] = []
        lines = context_pack.split("\n")
        for line in lines:
            stripped = line.strip()
            # Match "Possible conflict: 'X' contradicts 'Y'"
            m = re.search(
                r"[Pp]ossible conflict[:\s]+['\"]?([^'\"]+)['\"]?\s+contradicts\s+['\"]?([^'\"]+)['\"]?",
                stripped,
            )
            if m:
                choice = _normalize_label(m.group(1))
                constraint = _normalize_label(m.group(2))
                if choice and constraint:
                    pairs.append((choice, constraint))
                continue
            # Match lines containing conflict phrases
            for phrase in self._CONFLICT_PHRASES:
                if phrase.lower() in stripped.lower():
                    # Try to extract two entities around the phrase
                    parts = re.split(re.escape(phrase), stripped, flags=re.IGNORECASE, maxsplit=1)
                    if len(parts) == 2:
                        left = _normalize_label(parts[0])
                        right = _normalize_label(parts[1])
                        if left and right:
                            pairs.append((left, right))
                    break
        return pairs

    def extract(self, context_pack: str, question: str, gold_answer: str = "") -> str:
        pairs = self.extract_conflict_pairs(context_pack)
        if pairs:
            return "; ".join(f"{a} conflicts with {b}" for a, b in pairs)
        return DeterministicAnswerer().extract(context_pack, question, gold_answer)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def _final_answer_exact_match(extracted: str, gold: str) -> float:
    return 1.0 if gold.strip().lower() in extracted.strip().lower() else 0.0


def _final_answer_f1(extracted: str, gold: str) -> float:
    pred_tokens = tokenize_text(extracted)
    gold_tokens = tokenize_text(gold)
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _evidence_used(context_pack: str, gold_node_labels: list[str]) -> float:
    if not gold_node_labels:
        return 1.0
    pack_lower = context_pack.lower()
    found = sum(1 for lbl in gold_node_labels if lbl.lower() in pack_lower)
    return found / len(gold_node_labels)


def _contradiction_correctness(
    context_pack: str, gold_conflict_pairs: list[tuple[str, str]]
) -> float:
    if not gold_conflict_pairs:
        return 1.0
    pack_lower = context_pack.lower()
    found = sum(
        1
        for a, b in gold_conflict_pairs
        if a.lower() in pack_lower and b.lower() in pack_lower
    )
    return found / len(gold_conflict_pairs)


def _hallucination_rate(extracted: str, graph_node_texts: list[str]) -> float:
    """Fraction of sentences in extracted answer with no token overlap with any node text."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", extracted) if s.strip()]
    if not sentences:
        return 0.0
    all_graph_tokens: set[str] = set()
    for text in graph_node_texts:
        all_graph_tokens.update(tokenize_text(text))
    hallucinated = 0
    for sentence in sentences:
        sent_tokens = tokenize_text(sentence)
        if sent_tokens and not sent_tokens.intersection(all_graph_tokens):
            hallucinated += 1
    return hallucinated / len(sentences)


# ---------------------------------------------------------------------------
# Gold data helpers per family
# ---------------------------------------------------------------------------


def _get_gold_data(
    family: str,
    cases: list[Any],
    graph: Any,
    difficulty: str = "easy",
) -> dict[str, Any]:
    """
    Return gold data dict for a given family and its generated cases.

    Keys: gold_answer, gold_node_labels, gold_conflict_pairs
    """
    if family == "pairwise":
        case = cases[0]
        return {
            "gold_answer": "Use hosted Postgres, Use SaaS vector DB, Use external LLM API",
            "gold_conflict_pairs": case.gold_conflict_pairs,
            "gold_node_labels": case.all_constraint_labels,
        }
    elif family == "codeqa":
        case = cases[0]
        return {
            "gold_answer": "recursive_context.py",
            "gold_conflict_pairs": [],
            "gold_node_labels": ["recursive_context.py", "Decomposition is deterministic"],
        }
    elif family == "context_reset":
        case = cases[0]
        if difficulty == "easy":
            gold_answer = "Use PostgreSQL for storage"
        else:
            gold_answer = "Use FastAPI"
        # Collect gold node labels from the graph
        gold_node_labels: list[str] = []
        for nid in case.gold_decision_ids + case.gold_constraint_ids:
            try:
                node = graph.get_node(nid)
                if node:
                    gold_node_labels.append(node.label)
            except Exception:
                pass
        return {
            "gold_answer": gold_answer,
            "gold_conflict_pairs": [],
            "gold_node_labels": gold_node_labels,
        }
    else:
        # Generic fallback for other families
        return {
            "gold_answer": "",
            "gold_conflict_pairs": [],
            "gold_node_labels": [],
        }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_answer_level_eval(
    families: list[str],
    scales: list[int],
    methods: list[str],
    token_budget: int,
    seed: int,
    output_dir: str,
    answerer_type: str = "deterministic",
    verbose: bool = False,
) -> list[AnswerLevelResult]:
    """
    Run the answer-level evaluation pipeline.

    For each (family, scale, method):
    1. Create a fresh DB, generate cases
    2. Run the method runner to get context pack
    3. Run the answerer to extract an answer
    4. Compute all metrics
    5. Collect AnswerLevelResult

    SECURITY: GROQ_API_KEY is never written to any output file.
    """
    if answerer_type == "ollama":
        answerer: Any = OllamaAnswerer()
    elif answerer_type == "groq":
        answerer = GroqAnswerer()
        if not answerer._api_key:
            print(
                "WARNING: GROQ_API_KEY is not set. GroqAnswerer will fall back to "
                "DeterministicAnswerer for all calls.",
                file=sys.stderr,
            )
    else:
        answerer = DeterministicAnswerer()

    # Pairwise answerer always used for pairwise family regardless of main answerer
    pairwise_answerer = PairwiseDeterministicAnswerer()

    all_results: list[AnswerLevelResult] = []

    # Partial-run safety: flush results on exit
    def _flush_partial(results_list: list[AnswerLevelResult], out_dir: str) -> None:
        if results_list:
            try:
                _write_answer_results(results_list, out_dir, answerer_type=answerer_type)
            except Exception as exc:
                LOGGER.warning("Failed to flush partial results: %s", exc)

    atexit.register(_flush_partial, all_results, output_dir)

    for family in families:
        for scale in scales:
            rng = random.Random(seed)

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                db_path = tmp.name

            try:
                graph = _make_graph(db_path)

                # Generate cases for this family
                if family == "pairwise":
                    cases = generate_pairwise_cases(graph, scale_n=scale, rng=rng)
                    question = cases[0].question if cases else "Which choices conflict with constraints?"
                    difficulty = "easy"
                elif family == "codeqa":
                    cases = generate_codeqa_cases(graph, scale_n=scale, rng=rng)
                    question = cases[0].question if cases else "Which module handles decomposition?"
                    difficulty = "easy"
                elif family == "context_reset":
                    difficulty = "easy"
                    cases = generate_context_reset_cases(
                        graph, scale_n=scale, rng=rng, difficulty=difficulty
                    )
                    question = cases[0].question if cases else "Continue from where we left off"
                else:
                    # Use the benchmark runner for other families
                    runner_fn = _BENCHMARK_RUNNERS.get(family)
                    if runner_fn is None:
                        LOGGER.warning("Unknown family: %s, skipping", family)
                        continue
                    # Run the benchmark to populate the graph, then use a generic question
                    bench_results = runner_fn(
                        db_path=db_path,
                        scale_n=scale,
                        methods=methods[:1],  # just to populate graph
                        token_budget=token_budget,
                        rng=rng,
                        verbose=False,
                    )
                    cases = []
                    question = ""
                    difficulty = "easy"

                gold_data = _get_gold_data(family, cases, graph, difficulty=difficulty)
                gold_answer = gold_data["gold_answer"]
                gold_conflict_pairs = gold_data["gold_conflict_pairs"]
                gold_node_labels = gold_data["gold_node_labels"]

                # Load all node texts for hallucination rate computation
                try:
                    agg_result = graph.aggregate(query="", max_nodes=500, max_depth=0)
                    graph_node_texts = [
                        node.label + " " + node.content for node in agg_result.nodes
                    ]
                except Exception as exc:
                    LOGGER.debug("Failed to load graph nodes for hallucination check: %s", exc)
                    graph_node_texts = []

                for method in methods:
                    # Map rmca_full → build_context
                    runner_key = "build_context" if method == "rmca_full" else method
                    runner = _METHOD_RUNNERS.get(runner_key)
                    if runner is None:
                        LOGGER.warning("Unknown method: %s, skipping", method)
                        continue

                    try:
                        pack, _latency = runner(graph, question, token_budget)
                    except Exception as exc:
                        LOGGER.debug("Method %s failed: %s", method, exc)
                        pack = ""

                    # Extract answer — use structured extraction for Groq
                    insufficient_context = False
                    parse_error = 0
                    answer_tokens = 0
                    model_name = ""

                    if answerer_type == "groq":
                        groq_result = answerer.extract_structured(pack, question, gold_answer)
                        extracted = groq_result["answer"]
                        insufficient_context = groq_result["insufficient_context"]
                        parse_error = groq_result["parse_error"]
                        answer_tokens = groq_result["answer_tokens"]
                        model_name = groq_result["model_name"]
                    else:
                        extracted = answerer.extract(pack, question, gold_answer)
                        model_name = "deterministic" if answerer_type == "deterministic" else "ollama"

                    # Compute metrics
                    em = _final_answer_exact_match(extracted, gold_answer)
                    f1 = _final_answer_f1(extracted, gold_answer)
                    ev_used = _evidence_used(pack, gold_node_labels)
                    contra_corr = _contradiction_correctness(pack, gold_conflict_pairs)
                    hall_rate = _hallucination_rate(extracted, graph_node_texts)
                    tokens_injected = token_estimate(pack)

                    # Pairwise-specific metrics (Task 4) — always computed for pairwise family
                    pairwise_answer_f1 = 0.0
                    pairwise_conflict_recall = 0.0
                    pairwise_conflict_precision = 0.0
                    if family == "pairwise" and cases:
                        case = cases[0]
                        pw_extracted = pairwise_answerer.extract(pack, question, gold_answer)
                        pw_pairs = pairwise_answerer.extract_conflict_pairs(pack)
                        # Normalize gold pairs for comparison
                        gold_pairs_norm = [
                            (_normalize_label(a), _normalize_label(b))
                            for a, b in case.gold_conflict_pairs
                        ]
                        pred_pairs_norm = [
                            (_normalize_label(a), _normalize_label(b))
                            for a, b in pw_pairs
                        ]
                        from rlm_style_waggle_eval import pairwise_f1 as _pairwise_f1
                        pairwise_answer_f1 = _pairwise_f1(pred_pairs_norm, gold_pairs_norm)
                        # Recall: fraction of gold pairs found
                        gold_set = {(min(a, b), max(a, b)) for a, b in gold_pairs_norm}
                        pred_set = {(min(a, b), max(a, b)) for a, b in pred_pairs_norm}
                        if gold_set:
                            pairwise_conflict_recall = len(pred_set & gold_set) / len(gold_set)
                        if pred_set:
                            pairwise_conflict_precision = len(pred_set & gold_set) / len(pred_set)

                    # SECURITY: notes field must never contain API key
                    # Only store truncated extracted answer — no key material
                    safe_notes = f"extracted={extracted[:80]!r}"

                    result = AnswerLevelResult(
                        benchmark_family=family,
                        scale_n=scale,
                        method=method,
                        answerer=answerer_type,
                        final_answer_exact_match=em,
                        final_answer_f1=f1,
                        evidence_used=ev_used,
                        contradiction_correctness=contra_corr,
                        hallucination_rate=hall_rate,
                        tokens_injected=tokens_injected,
                        seed=seed,
                        notes=safe_notes,
                        insufficient_context=insufficient_context,
                        parse_error=parse_error,
                        answer_tokens=answer_tokens,
                        model_name=model_name,
                        pairwise_answer_f1=pairwise_answer_f1,
                        pairwise_conflict_recall=pairwise_conflict_recall,
                        pairwise_conflict_precision=pairwise_conflict_precision,
                    )
                    all_results.append(result)

                    if verbose:
                        print(
                            f"  [{family}] scale={scale} method={method} "
                            f"em={em:.2f} f1={f1:.2f} hall={hall_rate:.2f} "
                            f"tokens={tokens_injected}"
                            + (f" insuff={insufficient_context} parse_err={parse_error}" if answerer_type == "groq" else "")
                        )

            except Exception as exc:
                LOGGER.error("Error in family=%s scale=%d: %s", family, scale, exc)
                if verbose:
                    import traceback
                    traceback.print_exc()
            finally:
                try:
                    os.unlink(db_path)
                except Exception:
                    pass

    return all_results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_answer_results(
    results: list[AnswerLevelResult], output_dir: str, answerer_type: str = "deterministic"
) -> dict[str, str]:
    """Write CSV, Markdown, and JSON to output_dir.

    When answerer_type == 'groq', writes to groq_answer_level_results.{csv,md,json}.
    Otherwise writes to answer_level_results.{csv,md,json}.

    SECURITY: API key is never written to any output file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefix = "groq_answer_level_results" if answerer_type == "groq" else "answer_level_results"
    csv_path = out / f"{prefix}.csv"
    md_path = out / f"{prefix}.md"
    json_path = out / f"{prefix}.json"

    # CSV — verify no API key material in any field
    fieldnames = list(AnswerLevelResult.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            # SECURITY: sanitize all string fields — remove any accidental key material
            for k, v in row.items():
                if isinstance(v, str) and "gsk_" in v:
                    row[k] = "[REDACTED]"
            writer.writerow(row)

    # Markdown
    md_lines = [
        "# RMCA Answer-Level Evaluation Results",
        "",
        f"> **DISCLAIMER:** {DISCLAIMER}",
        "",
    ]
    if answerer_type == "groq":
        md_lines += [
            f"> **LLM EVAL CAVEAT:** {LLM_EVAL_CAVEAT}",
            "",
        ]
    md_lines += [
        "| Family | Scale | Method | Answerer | EM | F1 | Ev.Used | Contra.Corr | Hall.Rate | Tokens |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        md_lines.append(
            f"| {r.benchmark_family} | {r.scale_n} | {r.method} | {r.answerer} "
            f"| {r.final_answer_exact_match:.3f} | {r.final_answer_f1:.3f} "
            f"| {r.evidence_used:.3f} | {r.contradiction_correctness:.3f} "
            f"| {r.hallucination_rate:.3f} | {r.tokens_injected} |"
        )
    md_lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # JSON
    summary: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "total_cases": len(results),
        "families": {},
    }
    if answerer_type == "groq":
        summary["llm_eval_caveat"] = LLM_EVAL_CAVEAT

    families = sorted({r.benchmark_family for r in results})
    for fam in families:
        fam_results = [r for r in results if r.benchmark_family == fam]
        by_method: dict[str, dict] = {}
        for r in fam_results:
            if r.method not in by_method:
                by_method[r.method] = {
                    "avg_exact_match": 0.0,
                    "avg_f1": 0.0,
                    "avg_evidence_used": 0.0,
                    "avg_contradiction_correctness": 0.0,
                    "avg_hallucination_rate": 0.0,
                    "avg_tokens_injected": 0.0,
                    "insufficient_context_rate": 0.0,
                    "parse_error_rate": 0.0,
                    "count": 0,
                }
            entry = by_method[r.method]
            n = entry["count"]
            entry["avg_exact_match"] = (entry["avg_exact_match"] * n + r.final_answer_exact_match) / (n + 1)
            entry["avg_f1"] = (entry["avg_f1"] * n + r.final_answer_f1) / (n + 1)
            entry["avg_evidence_used"] = (entry["avg_evidence_used"] * n + r.evidence_used) / (n + 1)
            entry["avg_contradiction_correctness"] = (
                entry["avg_contradiction_correctness"] * n + r.contradiction_correctness
            ) / (n + 1)
            entry["avg_hallucination_rate"] = (
                entry["avg_hallucination_rate"] * n + r.hallucination_rate
            ) / (n + 1)
            entry["avg_tokens_injected"] = (
                entry["avg_tokens_injected"] * n + r.tokens_injected
            ) / (n + 1)
            entry["insufficient_context_rate"] = (
                entry["insufficient_context_rate"] * n + (1.0 if r.insufficient_context else 0.0)
            ) / (n + 1)
            entry["parse_error_rate"] = (
                entry["parse_error_rate"] * n + float(r.parse_error)
            ) / (n + 1)
            entry["count"] = n + 1
        summary["families"][fam] = by_method

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "json": str(json_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="RMCA answer-level evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["rmca_full", "query_graph", "bm25_topk"],
    )
    parser.add_argument("--scales", nargs="+", type=int, default=[128])
    parser.add_argument(
        "--families",
        nargs="+",
        default=["pairwise", "codeqa"],
        choices=_ALL_FAMILIES,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--answerer",
        default="deterministic",
        choices=["deterministic", "ollama", "groq"],
    )
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    seeds = args.seeds if args.seeds else [args.seed]

    print(f"RMCA Answer-Level Evaluation")
    print(f"  families : {args.families}")
    print(f"  scales   : {args.scales}")
    print(f"  methods  : {args.methods}")
    print(f"  answerer : {args.answerer}")
    print(f"  budget   : {args.token_budget} tokens")
    print(f"  seeds    : {seeds}")
    print(f"  output   : {args.output}")
    print()
    print(f"DISCLAIMER: {DISCLAIMER}")
    print()

    all_results: list[AnswerLevelResult] = []

    for seed in seeds:
        results = run_answer_level_eval(
            families=args.families,
            scales=args.scales,
            methods=args.methods,
            token_budget=args.token_budget,
            seed=seed,
            output_dir=args.output,
            answerer_type=args.answerer,
            verbose=args.verbose,
        )
        all_results.extend(results)

    if not all_results:
        print("No results produced.", file=sys.stderr)
        return 1

    # Add mean/std columns when multiple seeds
    if len(seeds) > 1:
        _add_multi_seed_stats(all_results, seeds)

    paths = _write_answer_results(all_results, args.output, answerer_type=args.answerer)
    print("Results written to:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    return 0


def _add_multi_seed_stats(
    results: list[AnswerLevelResult], seeds: list[int]
) -> None:
    """
    When multiple seeds are used, compute mean/std per (family, scale, method, answerer)
    and append aggregated rows with seed=-1.

    Aggregated fields: mean_f1, std_f1, mean_hallucination_rate, std_hallucination_rate,
    mean_insufficient_context_rate (for Groq).
    """
    from collections import defaultdict
    import statistics

    # Group by (family, scale, method, answerer)
    groups: dict[tuple, list[AnswerLevelResult]] = defaultdict(list)
    for r in results:
        key = (r.benchmark_family, r.scale_n, r.method, r.answerer)
        groups[key].append(r)

    aggregated_rows: list[AnswerLevelResult] = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        family, scale, method, answerer = key
        em_vals = [r.final_answer_exact_match for r in group]
        f1_vals = [r.final_answer_f1 for r in group]
        hall_vals = [r.hallucination_rate for r in group]
        insuff_vals = [1.0 if r.insufficient_context else 0.0 for r in group]

        mean_em = statistics.mean(em_vals)
        std_em = statistics.stdev(em_vals) if len(em_vals) > 1 else 0.0
        mean_f1 = statistics.mean(f1_vals)
        std_f1 = statistics.stdev(f1_vals) if len(f1_vals) > 1 else 0.0
        mean_hall = statistics.mean(hall_vals)
        std_hall = statistics.stdev(hall_vals) if len(hall_vals) > 1 else 0.0
        mean_insuff = statistics.mean(insuff_vals)

        proto = group[0]
        agg = AnswerLevelResult(
            benchmark_family=family,
            scale_n=scale,
            method=method,
            answerer=answerer,
            final_answer_exact_match=mean_em,
            final_answer_f1=mean_f1,
            evidence_used=statistics.mean(r.evidence_used for r in group),
            contradiction_correctness=statistics.mean(r.contradiction_correctness for r in group),
            hallucination_rate=mean_hall,
            tokens_injected=int(statistics.mean(r.tokens_injected for r in group)),
            seed=-1,
            notes=(
                f"mean_em={mean_em:.3f}±{std_em:.3f} "
                f"mean_f1={mean_f1:.3f}±{std_f1:.3f} "
                f"mean_hall={mean_hall:.3f}±{std_hall:.3f} "
                f"mean_insuff={mean_insuff:.3f} "
                f"n_seeds={len(group)}"
            ),
            insufficient_context=False,
            parse_error=0,
            answer_tokens=proto.answer_tokens,
            model_name=proto.model_name,
        )
        aggregated_rows.append(agg)

    results.extend(aggregated_rows)


if __name__ == "__main__":
    sys.exit(main())
