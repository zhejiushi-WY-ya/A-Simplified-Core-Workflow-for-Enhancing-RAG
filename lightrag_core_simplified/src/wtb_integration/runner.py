from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import Config
from ..pipelines import build_index_graph, build_query_state_graph
from ..runtime_paths import ensure_workspace_dir
from .tracing import question_trace


REQUIRED_QUERY_WORKSPACE_FILES = (
    "chunks.json",
    "graph.json",
    "graph_entity_index.json",
    "graph_relation_index.json",
    "kv.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relations.json",
)


@dataclass
class BenchmarkQuestion:
    question_id: str
    question_text: str
    cluster_id: Optional[str] = None
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionCandidateSpec:
    label: str = "baseline"
    variant_config: Dict[str, str] = field(default_factory=dict)
    workflow_variant: Optional[str] = None


@dataclass
class BenchmarkRunArtifacts:
    run_id: str
    run_dir: Path
    mapping_path: Path
    edge_mapping_path: Path
    results_path: Path
    summary_path: Path


@contextlib.contextmanager
def workspace_environment(workspace_dir: Path) -> Iterable[Dict[str, str]]:
    workspace_dir = workspace_dir.resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    wtb_data_dir = workspace_dir / "wtb_data"
    wtb_data_dir.mkdir(parents=True, exist_ok=True)

    env_updates = {
        "RAG_WORKSPACE_DIR": str(workspace_dir),
        "WTB_CHECKPOINT_DB_PATH": str(wtb_data_dir / "wtb_checkpoints.db"),
        "WTB_LLM_CACHE_PATH": str(wtb_data_dir / "llm_response_cache.db"),
        "WTB_LLM_RESPONSE_CACHE_ENABLED": "true",
        "WTB_CACHE_STORAGE_SCOPE": "workflow_local",
        "RAG_USE_WTB_CACHE": "true",
    }

    previous = {key: os.environ.get(key) for key in env_updates}
    os.environ.update(env_updates)
    try:
        yield env_updates
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextlib.contextmanager
def loaded_wtb_context() -> Iterable[Any]:
    try:
        from wtb.infrastructure.llm import reset_service_cache
        from wtb.sdk import WTBTestBench, WorkflowProject
    except ImportError as exc:
        raise RuntimeError(
            "WTB is not available in this Python environment. Install the bundled wheel first."
        ) from exc

    reset_service_cache()
    yield {
        "WTBTestBench": WTBTestBench,
        "WorkflowProject": WorkflowProject,
        "reset_service_cache": reset_service_cache,
    }
    reset_service_cache()


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_benchmark_questions(path: Path) -> List[BenchmarkQuestion]:
    return [
        BenchmarkQuestion(
            question_id=str(record["question_id"]),
            question_text=str(record["question_text"]),
            cluster_id=record.get("cluster_id"),
            ground_truth=record.get("ground_truth"),
            metadata={
                key: value
                for key, value in record.items()
                if key not in {"question_id", "question_text", "cluster_id", "ground_truth"}
            },
        )
        for record in load_jsonl_records(path)
    ]


def build_query_project(
    config: Config,
    workflow_project_cls: Any,
    *,
    project_name: str = "minimal_standardized_rag_qa",
) -> Any:
    return workflow_project_cls(
        name=project_name,
        description="Minimal WTB-wrapped QA path for the simplified modular RAG pipeline.",
        graph_factory=lambda: build_query_state_graph(config),
    )


def ensure_query_workspace_ready(workspace_dir: Path) -> None:
    missing = [name for name in REQUIRED_QUERY_WORKSPACE_FILES if not (workspace_dir / name).exists()]
    if missing:
        joined = "\n".join(f"  - {workspace_dir / name}" for name in missing)
        raise FileNotFoundError(
            "Workspace is missing indexed retrieval artifacts. "
            "Run the runner or smoke script with --prepare-index --corpus /path/to/corpus.jsonl, "
            "or point to an already indexed workspace.\n"
            f"Missing files:\n{joined}"
        )


async def prepare_fixture_index(
    *,
    config: Config,
    workspace_dir: Path,
    corpus_path: Path,
) -> None:
    with workspace_environment(workspace_dir):
        ensure_workspace_dir()
        try:
            index_graph = build_index_graph(config)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Index preparation requires the LightRAG runtime dependencies. "
                "Install the repo's indexing deps before using --prepare-index."
            ) from exc

        for record in load_jsonl_records(corpus_path):
            doc_id = str(record["doc_id"])
            content = str(record["context"])
            await index_graph.ainvoke({
                "doc_id": doc_id,
                "content": content,
            })


def build_run_artifacts(base_dir: Path, run_id: str) -> BenchmarkRunArtifacts:
    benchmark_dir = base_dir / "wtb_benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    run_dir = benchmark_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return BenchmarkRunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        mapping_path=benchmark_dir / "question_node_mapping.jsonl",
        edge_mapping_path=benchmark_dir / "question_materialization_edges.jsonl",
        results_path=run_dir / "question_results.jsonl",
        summary_path=run_dir / "summary.json",
    )


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_request_fingerprint(question_text: str) -> str:
    return hashlib.sha256(question_text.encode("utf-8")).hexdigest()


def node_path_prefixes(node_paths: Iterable[str]) -> List[str]:
    prefixes = set()
    for node_path in node_paths:
        parts = [part for part in node_path.split(".") if part]
        for index in range(1, len(parts) + 1):
            prefixes.add(".".join(parts[:index]))
    return sorted(prefixes)


def unique_materialized_keys(events: List[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for event in events:
        cache_key = event.get("cache_key")
        if not cache_key or cache_key in seen:
            continue
        seen.add(cache_key)
        ordered.append(cache_key)
    return ordered


def cache_hit_keys(events: List[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for event in events:
        cache_key = event.get("cache_key")
        if not cache_key or not event.get("cache_hit") or cache_key in seen:
            continue
        seen.add(cache_key)
        ordered.append(cache_key)
    return ordered


def execution_status_value(execution: Any) -> str:
    status = getattr(execution, "status", None)
    value = getattr(status, "value", None)
    if value:
        return str(value)
    return str(status or "unknown")


def read_execution_metadata(execution: Any) -> Dict[str, Any]:
    metadata = getattr(execution, "metadata", {}) or {}
    if isinstance(metadata, dict):
        return metadata
    return {}


def read_execution_error(execution: Any) -> Optional[str]:
    for attr in ("error_message", "error", "error_node_id"):
        value = getattr(execution, attr, None)
        if value:
            return str(value)
    metadata = read_execution_metadata(execution)
    for key in ("error", "error_message", "failure_reason"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def create_mapping_row(
    *,
    run_id: str,
    question: BenchmarkQuestion,
    trace_snapshot: Dict[str, Any],
    execution: Optional[Any],
    checkpoint_count: int,
    env_paths: Dict[str, str],
    status: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    events = trace_snapshot.get("events", [])
    event_node_paths = [event.get("node_path", "retrieval") for event in events]
    materialized_keys = unique_materialized_keys(events)
    hit_keys = cache_hit_keys(events)
    execution_metadata = read_execution_metadata(execution) if execution is not None else {}

    return {
        "run_id": run_id,
        "question_id": question.question_id,
        "question_text": question.question_text,
        "cluster_id": question.cluster_id,
        "node_path": "retrieval",
        "node_path_prefixes": node_path_prefixes(event_node_paths or ["retrieval"]),
        "materialized_keys": materialized_keys,
        "cache_hit": bool(hit_keys),
        "cache_hit_count": len(hit_keys),
        "materialized_key_count": len(materialized_keys),
        "llm_cache_path": execution_metadata.get("llm_cache_path") or env_paths["WTB_LLM_CACHE_PATH"],
        "checkpoint_db_path": execution_metadata.get("checkpoint_db_path") or env_paths["WTB_CHECKPOINT_DB_PATH"],
        "cache_storage_scope": execution_metadata.get("cache_storage_scope") or env_paths["WTB_CACHE_STORAGE_SCOPE"],
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_id": getattr(execution, "id", None),
        "checkpoint_count": checkpoint_count,
        "cache_hit_keys": hit_keys,
        "question_fingerprint": stable_request_fingerprint(question.question_text),
        "error": error,
    }


def build_edge_rows(
    *,
    mapping_row: Dict[str, Any],
    trace_snapshot: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for event in trace_snapshot.get("events", []):
        cache_key = event.get("cache_key")
        if not cache_key:
            continue
        rows.append({
            "run_id": mapping_row["run_id"],
            "question_id": mapping_row["question_id"],
            "question_text": mapping_row["question_text"],
            "cluster_id": mapping_row["cluster_id"],
            "node_path": event.get("node_path", mapping_row["node_path"]),
            "materialized_key": cache_key,
            "cache_hit": bool(event.get("cache_hit")),
            "llm_cache_path": mapping_row["llm_cache_path"],
            "checkpoint_db_path": mapping_row["checkpoint_db_path"],
            "status": mapping_row["status"],
            "timestamp": mapping_row["timestamp"],
        })
    return rows


def build_result_record(
    *,
    mapping_row: Dict[str, Any],
    trace_snapshot: Dict[str, Any],
    result_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        **mapping_row,
        "trace": trace_snapshot,
        "result": result_payload,
    }


def build_summary(
    *,
    run_id: str,
    mapping_rows: List[Dict[str, Any]],
    candidate: ExecutionCandidateSpec,
    workspace_dir: Path,
    benchmark_path: Path,
) -> Dict[str, Any]:
    completed = [row for row in mapping_rows if row["status"] == "completed"]
    cache_hit_rows = [row for row in completed if row["cache_hit"]]
    return {
        "run_id": run_id,
        "candidate_label": candidate.label,
        "workspace_dir": str(workspace_dir),
        "benchmark_path": str(benchmark_path),
        "question_count": len(mapping_rows),
        "completed_count": len(completed),
        "cache_hit_question_count": len(cache_hit_rows),
        "materialized_key_count": sum(row["materialized_key_count"] for row in mapping_rows),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_benchmark_once(
    *,
    workspace_dir: Path,
    benchmark_path: Path,
    provider: str,
    mode: str,
    candidate: Optional[ExecutionCandidateSpec] = None,
) -> Dict[str, Any]:
    candidate = candidate or ExecutionCandidateSpec()
    config = Config(provider=provider)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    workspace_dir = workspace_dir.resolve()
    artifacts = build_run_artifacts(workspace_dir, run_id)
    questions = load_benchmark_questions(benchmark_path)

    with workspace_environment(workspace_dir) as env_paths:
        ensure_query_workspace_ready(workspace_dir)
        with loaded_wtb_context() as wtb_ctx:
            WTBTestBench = wtb_ctx["WTBTestBench"]
            WorkflowProject = wtb_ctx["WorkflowProject"]
            bench = WTBTestBench.create(mode="development", data_dir=str(workspace_dir / "wtb_data"))
            project_name = f"minimal_standardized_rag_qa_{run_id.replace('-', '_')}"
            bench.register_project(
                build_query_project(
                    config,
                    WorkflowProject,
                    project_name=project_name,
                )
            )

            mapping_rows: List[Dict[str, Any]] = []
            try:
                for question in questions:
                    execution = None
                    checkpoint_count = 0
                    result_payload = None
                    status = "completed"
                    error = None
                    with question_trace(
                        question_id=question.question_id,
                        question_text=question.question_text,
                        cluster_id=question.cluster_id,
                    ) as collector:
                        try:
                            execution = bench.run(
                                project=project_name,
                                initial_state={
                                    "question_id": question.question_id,
                                    "query": question.question_text,
                                    "mode": mode,
                                    "cluster_id": question.cluster_id,
                                    "ground_truth": question.ground_truth,
                                    "candidate_label": candidate.label,
                                },
                                variant_config=candidate.variant_config,
                                workflow_variant=candidate.workflow_variant,
                            )
                            execution = bench.get_execution(execution.id)
                            checkpoint_count = len(bench.get_checkpoints(execution.id))
                            workflow_variables = getattr(getattr(execution, "state", None), "workflow_variables", {}) or {}
                            result_payload = workflow_variables.get("result")
                            status = execution_status_value(execution)
                            if status != "completed":
                                error = read_execution_error(execution) or f"execution status={status}"
                        except Exception as exc:
                            status = "failed"
                            error = str(exc)
                        trace_snapshot = collector.snapshot()

                    mapping_row = create_mapping_row(
                        run_id=run_id,
                        question=question,
                        trace_snapshot=trace_snapshot,
                        execution=execution,
                        checkpoint_count=checkpoint_count,
                        env_paths=env_paths,
                        status=status,
                        error=error,
                    )
                    append_jsonl(artifacts.mapping_path, mapping_row)
                    for edge_row in build_edge_rows(mapping_row=mapping_row, trace_snapshot=trace_snapshot):
                        append_jsonl(artifacts.edge_mapping_path, edge_row)
                    append_jsonl(
                        artifacts.results_path,
                        build_result_record(
                            mapping_row=mapping_row,
                            trace_snapshot=trace_snapshot,
                            result_payload=result_payload,
                        ),
                    )
                    mapping_rows.append(mapping_row)
            finally:
                bench.close()

    summary = build_summary(
        run_id=run_id,
        mapping_rows=mapping_rows,
        candidate=candidate,
        workspace_dir=workspace_dir,
        benchmark_path=benchmark_path,
    )
    artifacts.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "run_id": run_id,
        "artifacts": {
            "run_dir": str(artifacts.run_dir),
            "mapping_path": str(artifacts.mapping_path),
            "edge_mapping_path": str(artifacts.edge_mapping_path),
            "results_path": str(artifacts.results_path),
            "summary_path": str(artifacts.summary_path),
        },
        "summary": summary,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal WTB-backed QA benchmark once.")
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Workspace directory containing the query artifacts or where they should be created.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Benchmark JSONL file.",
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
        help="Retrieval mode passed into the existing retrieval node.",
    )
    parser.add_argument(
        "--prepare-index",
        action="store_true",
        help="Populate the workspace using the provided corpus JSONL before running the benchmark.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Corpus JSONL used with --prepare-index.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.prepare_index:
            if args.corpus is None:
                raise ValueError("--prepare-index requires --corpus /path/to/corpus.jsonl")
            asyncio.run(
                prepare_fixture_index(
                    config=Config(provider=args.provider),
                    workspace_dir=args.workspace,
                    corpus_path=args.corpus,
                )
            )
        result = run_benchmark_once(
            workspace_dir=args.workspace,
            benchmark_path=args.benchmark,
            provider=args.provider,
            mode=args.mode,
        )
    except Exception as exc:
        print(f"[wtb-runner] ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    summary = result["summary"]
    if summary["completed_count"] != summary["question_count"]:
        print(
            "[wtb-runner] FAIL: completed "
            f"{summary['completed_count']}/{summary['question_count']} questions",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
