import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from json_repair import repair_json
from openai import AsyncOpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SIMPLIFIED_RESULTS = BASE_DIR / "lightrag_core_simplified" / "results.jsonl"
STANDARD_RESULTS = BASE_DIR / "lightrag_standard_testing" / "results.jsonl"
OUTPUT_PATH = BASE_DIR / "evaluation_results.jsonl"
SUMMARY_PATH = BASE_DIR / "evaluation_summary.json"
MODEL_NAME = os.getenv("EVAL_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("LIGHTRAG_BASE_URL")
API_KEY = os.getenv("LIGHTRAG_API_KEY") or os.getenv("OPENAI_API_KEY")


def load_results(path):
    records = []

    if not Path(path).exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def index_by_query(records):
    indexed = {}

    for record in records:
        query = record.get("query", "").strip()
        if query:
            indexed[query] = record

    return indexed


def strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json(text):
    text = strip_code_fence(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            candidate = text
        else:
            candidate = text[start : end + 1]

        repaired = repair_json(candidate, skip_json_loads=True)
        return json.loads(repaired)


def normalize_winner(label):
    label = str(label or "").strip().lower()
    if label in {"answer 1", "1", "a1"}:
        return "Answer 1"
    if label in {"answer 2", "2", "a2"}:
        return "Answer 2"
    return "Unknown"


def build_prompt(query, answer_1, answer_2):
    return f"""
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: Comprehensiveness, Diversity, and Empowerment.

---Goal---
You will evaluate two answers to the same question based on three criteria: Comprehensiveness, Diversity, and Empowerment.

- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question: {query}

Here are the two answers:
Answer 1: {answer_1}

Answer 2: {answer_2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:
{{
  "Comprehensiveness": {{
    "Winner": "Answer 1 or Answer 2",
    "Explanation": "Provide explanation here"
  }},
  "Diversity": {{
    "Winner": "Answer 1 or Answer 2",
    "Explanation": "Provide explanation here"
  }},
  "Empowerment": {{
    "Winner": "Answer 1 or Answer 2",
    "Explanation": "Provide explanation here"
  }},
  "Overall Winner": {{
    "Winner": "Answer 1 or Answer 2",
    "Explanation": "Summarize why this answer is the overall winner based on the three criteria"
  }}
}}

Only output valid JSON.
""".strip()


async def judge_pair(client, query, simplified_record, standard_record, swap_order):
    simplified_answer = simplified_record.get("rag_answer", "")
    standard_answer = standard_record.get("rag_answer", "")

    if swap_order:
        answer_1 = standard_answer
        answer_2 = simplified_answer
        answer_map = {
            "Answer 1": "standard",
            "Answer 2": "simplified",
        }
    else:
        answer_1 = simplified_answer
        answer_2 = standard_answer
        answer_map = {
            "Answer 1": "simplified",
            "Answer 2": "standard",
        }

    prompt = build_prompt(query, answer_1, answer_2)
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.choices[0].message.content
    try:
        parsed = extract_json(raw_text)
    except Exception:
        repair_prompt = f"""
Convert the following content into valid JSON.
Do not change the meaning.
Return only valid JSON.

{raw_text}
""".strip()
        repair_response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": repair_prompt}],
        )
        raw_text = repair_response.choices[0].message.content
        parsed = extract_json(raw_text)

    normalized = {}
    for key in [
        "Comprehensiveness",
        "Diversity",
        "Empowerment",
        "Overall Winner",
    ]:
        value = parsed.get(key, {})
        winner = normalize_winner(value.get("Winner"))
        normalized[key] = {
            "Winner": winner,
            "Mapped Winner": answer_map.get(winner, "unknown"),
            "Explanation": value.get("Explanation", ""),
        }

    return {
        "query": query,
        "ground_truth": simplified_record.get("ground_truth")
        or standard_record.get("ground_truth", ""),
        "answer_order": {
            "Answer 1": answer_map["Answer 1"],
            "Answer 2": answer_map["Answer 2"],
        },
        "answers": {
            "simplified": simplified_answer,
            "standard": standard_answer,
        },
        "evaluation": normalized,
        "raw_judge_output": raw_text,
    }


def build_summary(records):
    criteria = [
        "Comprehensiveness",
        "Diversity",
        "Empowerment",
        "Overall Winner",
    ]
    summary = {
        "total_compared": len(records),
        "wins": {
            criterion: {
                "simplified": 0,
                "standard": 0,
                "unknown": 0,
            }
            for criterion in criteria
        },
    }

    for record in records:
        for criterion in criteria:
            winner = record["evaluation"][criterion]["Mapped Winner"]
            if winner not in summary["wins"][criterion]:
                winner = "unknown"
            summary["wins"][criterion][winner] += 1

    return summary


async def main():
    if not API_KEY:
        raise ValueError("Missing LIGHTRAG_API_KEY or OPENAI_API_KEY.")

    simplified_records = load_results(SIMPLIFIED_RESULTS)
    standard_records = load_results(STANDARD_RESULTS)

    simplified_by_query = index_by_query(simplified_records)
    standard_by_query = index_by_query(standard_records)

    shared_queries = [
        query for query in simplified_by_query
        if query in standard_by_query
    ]

    client_kwargs = {"api_key": API_KEY}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    client = AsyncOpenAI(**client_kwargs)

    evaluations = load_results(OUTPUT_PATH)
    completed_queries = {record.get("query", "") for record in evaluations}

    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
        for idx, query in enumerate(shared_queries):
            if query in completed_queries:
                continue

            print("\n==============================")
            print(f"EVALUATING {idx}: {query}")
            print("==============================")

            result = await judge_pair(
                client,
                query,
                simplified_by_query[query],
                standard_by_query[query],
                swap_order=(idx % 2 == 1),
            )
            evaluations.append(result)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            overall = result["evaluation"]["Overall Winner"]["Mapped Winner"]
            print(f"Overall winner: {overall}")

    summary = build_summary(evaluations)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed evaluations to {OUTPUT_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
