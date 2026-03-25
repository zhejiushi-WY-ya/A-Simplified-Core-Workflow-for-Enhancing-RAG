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

client = AsyncOpenAI()

async def load_data(rag):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i==10:
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
            if i==10:
                break
            data = json.loads(line)
            queries.append(data.get("input", ""))
            gts.append(data.get("answers", [""])[0])
    return queries, gts


# ====== LLM评估（核心） ======
async def judge_answer(query, gt, ans1, ans2):
    prompt = f"""
        You are an evaluator.
        
        Question:
        {query}
        
        Ground Truth:
        {gt}
        
        Answer A:
        {ans1}
        
        Answer B:
        {ans2}
        
        Which answer is better? Reply with:
        - "A"
        - "B"
        - "Tie"
        
        Also give a short reason.
    """

    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return resp.choices[0].message.content


# ====== 主实验 ======
async def run_experiment(rag):
    queries, gts = load_queries()

    modes = ["naive", "hybrid"]

    for i, q in enumerate(queries):
        print("\n==============================")
        print(f"QUERY: {q}")
        print("==============================")

        results = {}

        for mode in modes:
            ans = await rag.aquery(q, param=QueryParam(mode=mode))
            results[mode] = ans

        print("\n--- naive ---")
        print(results["naive"])

        print("\n--- hybrid ---")
        print(results["hybrid"])

        # ===== LLM评估 =====
        judge = await judge_answer(
            q,
            gts[i],
            results["naive"],
            results["hybrid"],
        )

        print("\n🔥 JUDGE RESULT:")
        print(judge)


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
