import json
import asyncio
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

BASE_DIR = Path(__file__).resolve().parent.parent
WORKING_DIR = BASE_DIR / "exp_data"

DATA_PATHS = [
    BASE_DIR.parent / "raw_data" / "agriculture.jsonl",
    BASE_DIR.parent / "raw_data" / "cs.jsonl",
    BASE_DIR.parent / "raw_data" / "legal.jsonl",
    BASE_DIR.parent / "raw_data" / "mix.jsonl",
]

OUTPUT_PATH = BASE_DIR / "results.jsonl"


def load_queries(max_queries=None):
    """
    从多个 jsonl 文件中按顺序读取 query，不去重。
    每条记录保留来源文件名，便于后续分析。
    """
    queries = []
    gts = []
    sources = []

    for file_path in DATA_PATHS:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ query 文件不存在，跳过: {file_path}")
            continue

        print(f"\n📂 开始读取 query 文件: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if max_queries is not None and len(queries) >= max_queries:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ query 文件中存在非法 JSON，跳过: {file_path}:{i}")
                    continue

                q = data.get("input", "")
                if not isinstance(q, str):
                    q = str(q)
                q = q.strip()

                if not q:
                    continue

                queries.append(q)

                answers = data.get("answers", [""])
                gt = answers[0] if isinstance(answers, list) and answers else ""
                if not isinstance(gt, str):
                    gt = str(gt)
                gts.append(gt)

                sources.append(path.name)

        if max_queries is not None and len(queries) >= max_queries:
            break

    print(f"\n✅ Queries loaded: {len(queries)}（来自 4 个文件，未去重）")
    return queries, gts, sources


async def run_experiment(rag, mode="hybrid", max_queries=None):
    queries, gts, sources = load_queries(max_queries=max_queries)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for i, q in enumerate(queries):
            print("\n==============================")
            print(f"QUERY {i+1} [{sources[i]}]: {q}")
            print("==============================")

            ans = await rag.aquery(q, param=QueryParam(mode=mode))

            print("\n--- RAG ANSWER ---")
            print(ans)

            record = {
                "source_file": sources[i],
                "query": q,
                "ground_truth": gts[i],
                "rag_answer": ans,
                "mode": mode,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Results saved to {OUTPUT_PATH}")


async def main():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()
    await run_experiment(rag, mode="hybrid", max_queries=None)
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())