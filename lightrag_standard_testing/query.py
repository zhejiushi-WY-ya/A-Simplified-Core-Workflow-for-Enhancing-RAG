import os
import json
import asyncio
import hashlib
from pathlib import Path
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

BASE_DIR = Path(__file__).resolve().parent
WORKING_DIR = BASE_DIR / "exp_data"

CORPUS_PATHS = [
    BASE_DIR / "raw_data" / "agriculture.jsonl",
    BASE_DIR / "raw_data" / "cs.jsonl",
    BASE_DIR / "raw_data" / "legal.jsonl",
    BASE_DIR / "raw_data" / "mix.jsonl",
]


def normalize_text(text: str) -> str:
    """轻量归一化，避免同一文本因空格/换行差异被当成不同文本。"""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.strip().split())


def text_hash(text: str) -> str:
    """对归一化后的文本做 hash，用于跨文件去重。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


async def build_corpus_db(rag, max_records=None):
    """
    对多个语料文件做去重并写入 RAG 数据库。
    - 多个文件里重复的 context 只插入一次
    - max_records=None 表示不设上限
    """
    seen_hashes = set()
    loaded_count = 0
    skipped_duplicate = 0
    skipped_empty = 0
    skipped_invalid = 0

    for file_path in CORPUS_PATHS:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ 文件不存在，跳过: {file_path}")
            continue

        print(f"\n📂 开始读取语料文件: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if max_records is not None and loaded_count >= max_records:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    skipped_invalid += 1
                    print(f"⚠️ 非法 JSON，跳过: {file_path}:{line_no}")
                    continue

                text = data.get("context", "")
                text = normalize_text(text)

                if not text:
                    skipped_empty += 1
                    continue

                h = text_hash(text)
                if h in seen_hashes:
                    skipped_duplicate += 1
                    continue

                seen_hashes.add(h)
                await rag.ainsert(text)
                loaded_count += 1

        if max_records is not None and loaded_count >= max_records:
            break

    print("\n✅ 数据库构建完成")
    print(f"已插入唯一语料: {loaded_count}")
    print(f"重复语料跳过: {skipped_duplicate}")
    print(f"空文本跳过: {skipped_empty}")
    print(f"非法 JSON 跳过: {skipped_invalid}")


async def main():
    os.makedirs(WORKING_DIR, exist_ok=True)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()
    await build_corpus_db(rag, max_records=None)
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
