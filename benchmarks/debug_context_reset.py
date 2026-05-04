"""
Debug tool for ContextReset benchmark.

Usage:
  python benchmarks/debug_context_reset.py \\
    --method rmca_full \\
    --scale 128 \\
    --seed 42 \\
    --token-budget 1200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for p in [str(_SRC), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from waggle.recursive_context import RecursiveContextController
from rlm_style_waggle_eval import (
    _make_graph,
    _run_build_context_scoped,
    _run_query_graph,
    _run_prime_context,
    _run_bm25_topk,
    _run_hybrid_rrf,
    _run_raw_context,
    _METHOD_RUNNERS,
    generate_context_reset_cases,
    _score_context_reset,
    token_estimate,
)

_PROJECT = "context_reset"


def _get_node_label(graph, node_id: str) -> str:
    try:
        node = graph.get_node(node_id)
        return node.label if node else f"<missing:{node_id}>"
    except Exception:
        return f"<error:{node_id}>"


def _get_node(graph, node_id: str):
    try:
        return graph.get_node(node_id)
    except Exception:
        return None


def run_debug(
    method: str,
    scale: int,
    seed: int,
    token_budget: int,
    difficulty: str = "easy",
) -> None:
    rng = random.Random(seed)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    graph = _make_graph(db_path)
    cases = generate_context_reset_cases(
        graph, scale_n=scale, rng=rng, difficulty=difficulty, project=_PROJECT
    )
    case = cases[0]

    print("=" * 70)
    print(f"ContextReset Debug — method={method} scale={scale} seed={seed} difficulty={difficulty}")
    print("=" * 70)
    print()

    # --- Gold fields ---
    print("GOLD FIELDS:")
    print(f"  case_id              : {case.case_id}")
    print(f"  question             : {case.question!r}")
    print(f"  gold_decision_ids    : {case.gold_decision_ids}")
    print(f"  gold_constraint_ids  : {case.gold_constraint_ids}")
    print(f"  gold_next_step_id    : {case.gold_next_step_id}")
    print(f"  gold_superseded_id   : {case.gold_superseded_id}")
    print(f"  gold_active_decision_id: {case.gold_active_decision_id}")
    print()

    # --- Gold nodes ---
    all_gold_ids = (
        case.gold_decision_ids
        + case.gold_constraint_ids
        + [case.gold_next_step_id, case.gold_superseded_id]
    )
    print("GOLD NODES:")
    for nid in all_gold_ids:
        node = _get_node(graph, nid)
        if node:
            print(f"  [{node.node_type.value}] {node.label!r}")
            print(f"    content  : {node.content[:120]!r}")
            print(f"    node_id  : {nid}")
        else:
            print(f"  <node not found: {nid}>")
    print()

    # --- Run method ---
    print(f"RUNNING METHOD: {method}")
    t0 = time.perf_counter()
    if method in ("rmca_full", "build_context"):
        pack, latency = _run_build_context_scoped(
            graph, case.question, token_budget, project=_PROJECT
        )
    elif method == "query_graph":
        try:
            result_q = graph.query(
                query=case.question,
                max_nodes=20,
                max_depth=2,
                retrieval_mode="hybrid",
                project=_PROJECT,
            )
            lines = []
            used = 0
            budget = int(token_budget * 1.15)
            for node in result_q.nodes:
                line = f"[{node.node_type.value}] {node.label}: {node.content}"
                cost = token_estimate(line)
                if used + cost > budget:
                    break
                lines.append(line)
                used += cost
            pack = "\n".join(lines)
        except Exception as exc:
            print(f"  query_graph failed: {exc}")
            pack = ""
        latency = (time.perf_counter() - t0) * 1000
    elif method == "no_memory":
        pack = ""
        latency = 0.0
    else:
        runner = _METHOD_RUNNERS.get(method)
        if runner is None:
            print(f"  Unknown method: {method}")
            return
        pack, latency = runner(graph, case.question, token_budget)

    print(f"  latency: {latency:.1f}ms  tokens: {token_estimate(pack)}")
    print()

    # --- Context pack ---
    print("CONTEXT PACK:")
    print("-" * 60)
    print(pack if pack else "<empty>")
    print("-" * 60)
    print()

    # --- Scoring breakdown ---
    scoring = _score_context_reset(pack, case, graph)
    pack_lower = pack.lower()

    print("SCORING BREAKDOWN:")
    print()

    # decision_recall
    decision_labels = [_get_node_label(graph, nid) for nid in case.gold_decision_ids]
    print(f"  decision_recall = {scoring['decision_recall']:.3f}")
    for lbl in decision_labels:
        found = lbl.lower() in pack_lower
        print(f"    {'✅' if found else '❌'} {lbl!r} {'(found)' if found else '(MISSING)'}")
    print()

    # constraint_recall
    constraint_labels = [_get_node_label(graph, nid) for nid in case.gold_constraint_ids]
    print(f"  constraint_recall = {scoring['constraint_recall']:.3f}")
    for lbl in constraint_labels:
        found = lbl.lower() in pack_lower
        print(f"    {'✅' if found else '❌'} {lbl!r} {'(found)' if found else '(MISSING)'}")
    print()

    # next_step_accuracy
    next_step_label = _get_node_label(graph, case.gold_next_step_id)
    found_ns = next_step_label.lower() in pack_lower
    print(f"  next_step_accuracy = {scoring['next_step_accuracy']:.3f}")
    print(f"    {'✅' if found_ns else '❌'} {next_step_label!r} {'(found)' if found_ns else '(MISSING)'}")
    print()

    # superseded_context_handling
    superseded_label = _get_node_label(graph, case.gold_superseded_id)
    sup_in_pack = superseded_label.lower() in pack_lower
    sup_header_idx = pack_lower.find("superseded context")
    sup_label_idx = pack_lower.find(superseded_label.lower()) if sup_in_pack else -1
    print(f"  superseded_context_handling = {scoring['superseded_context_handling']:.3f}")
    print(f"    superseded label: {superseded_label!r}")
    print(f"    in pack: {sup_in_pack}")
    if sup_in_pack:
        print(f"    superseded header idx: {sup_header_idx}")
        print(f"    label idx: {sup_label_idx}")
        if sup_header_idx >= 0 and sup_label_idx > sup_header_idx:
            print("    → appears AFTER 'Superseded context:' header ✅")
        else:
            print("    → appears BEFORE 'Superseded context:' header ❌")
    print()

    # active_decision_preference
    active_label = _get_node_label(graph, case.gold_active_decision_id)
    active_in_pack = active_label.lower() in pack_lower
    if sup_header_idx >= 0:
        active_section = pack_lower[:sup_header_idx]
    else:
        active_section = pack_lower
    active_in_active = active_label.lower() in active_section
    superseded_in_active = superseded_label.lower() in active_section
    print(f"  active_decision_preference = {scoring['active_decision_preference']:.3f}")
    print(f"    active label: {active_label!r}")
    print(f"    active in pack: {active_in_pack}")
    print(f"    active in active section: {active_in_active}")
    print(f"    superseded in active section: {superseded_in_active}")
    print()

    # evidence_coverage
    all_gold_labels = [_get_node_label(graph, nid) for nid in all_gold_ids]
    print(f"  evidence_coverage = {scoring['evidence_coverage']:.3f}")
    for lbl in all_gold_labels:
        found = lbl.lower() in pack_lower
        print(f"    {'✅' if found else '❌'} {lbl!r}")
    print()

    # --- Final scores ---
    composite = (
        scoring["decision_recall"]
        + scoring["constraint_recall"]
        + scoring["next_step_accuracy"]
        + scoring["active_decision_preference"]
    ) / 4.0
    print("FINAL SCORES:")
    for k, v in scoring.items():
        print(f"  {k:<35} = {v:.3f}")
    print(f"  {'composite (mean of 4 primary)':<35} = {composite:.3f}")
    print()

    # --- Subquery debug for rmca_full ---
    if method in ("rmca_full", "build_context"):
        print("SUBQUERY DEBUG (rmca_full):")
        controller = RecursiveContextController(graph=graph)
        subqueries = controller._decompose_query(case.question, max_subqueries=6, mode="balanced")
        for sq in subqueries:
            print(f"  [{sq.purpose}] {sq.query!r} (priority={sq.priority:.2f})")
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Debug ContextReset benchmark scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--method", default="rmca_full", help="Method to debug")
    parser.add_argument("--scale", type=int, default=128, help="Scale (number of nodes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--token-budget", type=int, default=1200, help="Token budget")
    parser.add_argument(
        "--difficulty", default="easy", choices=["easy", "hard"], help="Case difficulty"
    )
    args = parser.parse_args(argv)

    run_debug(
        method=args.method,
        scale=args.scale,
        seed=args.seed,
        token_budget=args.token_budget,
        difficulty=args.difficulty,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
