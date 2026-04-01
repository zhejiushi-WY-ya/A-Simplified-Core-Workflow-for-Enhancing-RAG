import asyncio
import json
import os
from pathlib import Path

from langgraph.graph import END, StateGraph

from .config import Config
from .nodes.retrieval_node import build_node

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATHS = [
    BASE_DIR.parent / "raw_data" / "agriculture.jsonl",
    BASE_DIR.parent / "raw_data" / "cs.jsonl",
    BASE_DIR.parent / "raw_data" / "legal.jsonl",
    BASE_DIR.parent / "raw_data" / "mix.jsonl",
]

OUTPUT_DIR = BASE_DIR / "results"
MAX_RECORDS = 1
DEFAULT_MODE = "hybrid"


def build_graph(config):
    g = StateGraph(dict)

    g.add_node("retrieval", build_node(config))

    g.set_entry_point("retrieval")
    g.add_edge("retrieval", END)

    return g.compile()


def load_queries(data_path: Path, max_records=20):
    queries = []

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ Skipped invalid JSON at {data_path}:{i + 1}")
                continue

            query = str(data.get("input", "")).strip()
            if not query:
                continue

            answers = data.get("answers") or [""]
            ground_truth = answers[0] if isinstance(answers, list) and answers else ""
            if not isinstance(ground_truth, str):
                ground_truth = str(ground_truth)

            queries.append({
                "query": query,
                "ground_truth": ground_truth,
                "meta": {
                    "record_index": i,
                    "source_file": data_path.name,
                    "context": data.get("context", ""),
                },
            })

    return queries


async def run_experiment_for_file(query_graph, data_path: Path, mode: str, output_path: Path, max_records=20):
    queries = load_queries(data_path, max_records=max_records)
    print(f"\n📂 {data_path.name} loaded queries: {len(queries)}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(queries):
            query = item["query"]
            print("\n==============================")
            print(f"{data_path.name} | QUERY {idx + 1}: {query}")
            print("==============================")

            result = await query_graph.ainvoke({
                "query": query,
                "mode": mode,
            })

            answer = result["result"]["answer"]
            print("\n--- RAG ANSWER ---")
            print(answer)

            record = {
                "query": query,
                "ground_truth": item["ground_truth"],
                "rag_answer": answer,
                "mode": mode,
                "retrieval": {
                    "chunks": result["result"]["chunks"],
                    "relations": result["result"]["relations"],
                    "subgraph": result["result"]["subgraph"],
                    "facts": result["result"]["facts"],
                    "context": result["result"]["context"],
                },
                "meta": item["meta"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Results saved to {output_path}")


async def run():
    missing_files = [p for p in DATA_PATHS if not p.exists()]
    if missing_files:
        missing_text = "\n".join(str(p) for p in missing_files)
        raise FileNotFoundError(f"Missing dataset files:\n{missing_text}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = Config(provider='local')

    query_graph = build_graph(config)

    for data_path in DATA_PATHS:
        output_path = OUTPUT_DIR / f"results_{data_path.stem}.jsonl"
        await run_experiment_for_file(
            query_graph=query_graph,
            data_path=data_path,
            mode=DEFAULT_MODE,
            output_path=output_path,
            max_records=MAX_RECORDS,
        )


if __name__ == "__main__":
    asyncio.run(run())
