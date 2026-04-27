from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runner import Config, prepare_fixture_index, run_benchmark_once


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal WTB QA smoke twice and report cache reuse.")
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Workspace directory to use for both passes.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "local"],
        default="openai",
        help="Model provider config to use.",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        help="Retrieval mode passed to the query node.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Benchmark JSONL file.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Corpus JSONL file used for --prepare-index.",
    )
    parser.add_argument(
        "--prepare-index",
        action="store_true",
        help="Populate the workspace with the provided corpus JSONL before the two passes.",
    )
    return parser.parse_args(argv)


def verify_reuse(first_result: Dict[str, Any], second_result: Dict[str, Any]) -> Dict[str, Any]:
    first_results_path = Path(first_result["artifacts"]["results_path"])
    second_results_path = Path(second_result["artifacts"]["results_path"])

    first_rows = [json.loads(line) for line in first_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    second_rows = [json.loads(line) for line in second_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    first_by_question = {row["question_id"]: row for row in first_rows}
    reused_questions = []
    reused_keys = set()

    for row in second_rows:
        question_id = row["question_id"]
        first_row = first_by_question.get(question_id)
        if first_row is None:
            continue
        first_keys = set(first_row.get("materialized_keys", []))
        second_hit_keys = set(row.get("cache_hit_keys", []))
        overlap = first_keys.intersection(second_hit_keys)
        if overlap:
            reused_questions.append(question_id)
            reused_keys.update(overlap)

    failures = []
    first_errors = sorted({
        str(row.get("error"))
        for row in first_rows
        if row.get("status") != "completed" and row.get("error")
    })
    second_errors = sorted({
        str(row.get("error"))
        for row in second_rows
        if row.get("status") != "completed" and row.get("error")
    })
    first_summary = first_result["summary"]
    second_summary = second_result["summary"]
    if first_summary["completed_count"] != first_summary["question_count"]:
        failures.append(
            "first run completed "
            f"{first_summary['completed_count']}/{first_summary['question_count']} questions"
        )
        if first_errors:
            failures.append(f"first run errors: {' | '.join(first_errors[:3])}")
    if second_summary["completed_count"] != second_summary["question_count"]:
        failures.append(
            "second run completed "
            f"{second_summary['completed_count']}/{second_summary['question_count']} questions"
        )
        if second_errors:
            failures.append(f"second run errors: {' | '.join(second_errors[:3])}")
    if not reused_questions:
        failures.append("second run did not reuse any first-run materialized keys")
    if second_result["summary"]["cache_hit_question_count"] == 0:
        failures.append("second run reported zero cache-hit questions")

    return {
        "verified_cache_reuse": bool(reused_questions) and not failures,
        "questions_with_reuse": reused_questions,
        "reused_materialized_key_count": len(reused_keys),
        "second_run_cache_hit_question_count": second_result["summary"]["cache_hit_question_count"],
        "failures": failures,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.prepare_index:
            if args.corpus is None:
                raise ValueError("--prepare-index requires --corpus /path/to/corpus.jsonl")
            import asyncio

            asyncio.run(
                prepare_fixture_index(
                    config=Config(provider=args.provider),
                    workspace_dir=args.workspace,
                    corpus_path=args.corpus,
                )
            )

        first_result = run_benchmark_once(
            workspace_dir=args.workspace,
            benchmark_path=args.benchmark,
            provider=args.provider,
            mode=args.mode,
        )
        second_result = run_benchmark_once(
            workspace_dir=args.workspace,
            benchmark_path=args.benchmark,
            provider=args.provider,
            mode=args.mode,
        )
        reuse_summary = verify_reuse(first_result, second_result)
    except Exception as exc:
        print(f"[wtb-smoke] ERROR: {exc}", file=sys.stderr)
        return 1

    payload = {
        "first_run": first_result,
        "second_run": second_result,
        "reuse_summary": reuse_summary,
    }
    smoke_summary_path = args.workspace / "wtb_benchmark" / "smoke_summary.json"
    smoke_summary_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    failures = reuse_summary.get("failures", [])
    if failures:
        print("[wtb-smoke] FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
