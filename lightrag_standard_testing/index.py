import os
import json
import asyncio
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from dotenv import load_dotenv

load_dotenv()

WORKING_DIR = "./exp_data"
DATA_PATH = "./raw_data/mix.jsonl"
OUTPUT_PATH = "./results.jsonl"   # ✅ 输出文件

client = AsyncOpenAI()


# ====== 加载数据 ======
async def load_data(rag):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 20:
                break
            data = json.loads(line)
            text = data.get("context", "")
            await rag.ainsert(text)

    print("✅ Data loading done\n")


# ====== 取 query + ground truth ======
def load_queries():
    queries = []
    gts = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 20:
                break
            data = json.loads(line)
            queries.append(data.get("input", ""))
            gts.append(data.get("answers", [""])[0])

    return queries, gts


# ====== 主实验（只保存结果） ======
async def run_experiment(rag):
    queries, gts = load_queries()

    mode = "hybrid"   # ✅ 你当前用的模式

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for i, q in enumerate(queries):
            print("\n==============================")
            print(f"QUERY: {q}")
            print("==============================")

            # 👉 RAG生成
            ans = await rag.aquery(q, param=QueryParam(mode=mode))

            print("\n--- RAG ANSWER ---")
            print(ans)

            # 👉 保存结果
            record = {
                "query": q,
                "ground_truth": gts[i],
                "rag_answer": ans,
                "mode": mode
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Results saved to {OUTPUT_PATH}")


# ====== 主函数 ======
async def main():
    os.makedirs(WORKING_DIR, exist_ok=True)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()

    await load_data(rag)

    await run_experiment(rag)

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
