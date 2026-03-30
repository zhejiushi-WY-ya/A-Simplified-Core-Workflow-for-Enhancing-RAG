import json
import asyncio
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

BASE_DIR = Path(__file__).resolve().parent
WORKING_DIR = BASE_DIR / "exp_data"
RESULT_DIR = BASE_DIR / "results"

QUERY_PATHS = [
    BASE_DIR / "raw_data" / "agriculture.jsonl",
    BASE_DIR / "raw_data" / "cs.jsonl",
    BASE_DIR / "raw_data" / "legal.jsonl",
    BASE_DIR / "raw_data" / "mix.jsonl",
]


def load_queries_from_file(query_path: Path, max_queries=None):
    """
    从单个 jsonl 文件中读取 query，不去重，保持原顺序。
    """
    queries = []
    gts = []

    if not query_path.exists():
        raise FileNotFoundError(f"QUERY_PATH 不存在: {query_path}")

    with open(query_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if max_queries is not None and len(queries) >= max_queries:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ query 文件中存在非法 JSON，跳过: {query_path}:{i}")
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

    print(f"✅ {query_path.name} 加载 query 数量: {len(queries)}")
    return queries, gts


async def run_queries_for_file(rag, query_path: Path, mode="hybrid", max_queries=None):
    queries, gts = load_queries_from_file(query_path, max_queries=max_queries)

    output_path = RESULT_DIR / f"results_{query_path.stem}.jsonl"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, q in enumerate(queries):
            print("\n==============================")
            print(f"{query_path.name} | QUERY {i+1}: {q}")
            print("==============================")

            ans = await rag.aquery(q, param=QueryParam(mode=mode))

            print("\n--- RAG ANSWER ---")
            print(ans)

            record = {
                "source_file": query_path.name,
                "query": q,
                "ground_truth": gts[i],
                "rag_answer": ans,
                "mode": mode,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ {query_path.name} 结果已保存到: {output_path}")


async def main():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()

    total_files = len(QUERY_PATHS)

    for idx, query_path in enumerate(QUERY_PATHS, start=1):
        remaining = total_files - idx

        print("\n====================================")
        print(f"文件进度: {idx}/{total_files}")
        print(f"当前文件: {query_path.name}")
        print(f"剩余文件: {remaining}")
        print("====================================")

        if not query_path.exists():
            print(f"⚠️ 文件不存在，跳过: {query_path}")
            continue

        await run_queries_for_file(
            rag,
            query_path=query_path,
            mode="hybrid",
            max_queries=None,
        )

    await rag.finalize_storages()



if __name__ == "__main__":
    asyncio.run(main())
