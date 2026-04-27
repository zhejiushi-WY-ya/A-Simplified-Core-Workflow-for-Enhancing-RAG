import asyncio
import hashlib
import json
from pathlib import Path

from .config import Config
from .pipelines import build_index_graph
from .runtime_paths import ensure_workspace_dir, get_workspace_dir


BASE_DIR = Path(__file__).resolve().parent.parent
WORKING_DIR = get_workspace_dir()

DATA_PATHS = [
    BASE_DIR.parent / "raw_data" / "agriculture.jsonl",
    BASE_DIR.parent / "raw_data" / "cs.jsonl",
    BASE_DIR.parent / "raw_data" / "legal.jsonl",
    BASE_DIR.parent / "raw_data" / "mix.jsonl",
]


def _content_fingerprint(text: str) -> str:
    normalized = " ".join(str(text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _dedupe_documents(documents):
    seen = set()
    unique_documents = []
    duplicate_count = 0
    empty_count = 0

    for document in documents:
        content = str(document.get("content", ""))
        if not content.strip():
            empty_count += 1
            continue

        fingerprint = _content_fingerprint(content)
        if fingerprint in seen:
            duplicate_count += 1
            continue

        seen.add(fingerprint)
        unique_documents.append(document)

    return unique_documents, duplicate_count, empty_count


def _print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _print_subsection(title: str):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


async def load_data(index_graph, max_records=None):
    from .store.graph_store import GraphStore

    _print_section("Indexing Started")
    print(f"Current working dir : {Path.cwd()}")
    print(f"Script base dir     : {BASE_DIR}")
    print(f"Working/output dir  : {WORKING_DIR}")
    print("DATA_PATHS:")
    for p in DATA_PATHS:
        print(f"  - {p}")

    existing_data_paths = [p for p in DATA_PATHS if p.exists()]

    if not existing_data_paths:
        raise FileNotFoundError(
            "No raw data files found.\n"
            f"Expected paths:\n" + "\n".join(f"  - {p}" for p in DATA_PATHS)
        )

    raw_documents = []
    invalid_count = 0
    total_read_count = 0

    total_files = len(existing_data_paths)

    _print_section(f"Found {total_files} raw_data files, preparing to load")

    for file_idx, data_path in enumerate(existing_data_paths, start=1):
        remaining_files = total_files - file_idx
        _print_subsection(
            f"[File {file_idx}/{total_files}] Reading {data_path.name} | Remaining files: {remaining_files}"
        )

        file_total_lines = 0
        file_valid_docs = 0
        file_invalid = 0

        with data_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if max_records is not None and total_read_count >= max_records:
                    break

                file_total_lines += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    invalid_count += 1
                    file_invalid += 1
                    print(f"  ⚠ Skipped invalid JSON at {data_path.name}:{i}")
                    continue

                text = data.get("context", "")
                raw_documents.append({
                    "doc_id": f"{data_path.stem}_doc_{i}",
                    "content": text,
                    "source_file": data_path.name,
                })
                total_read_count += 1
                file_valid_docs += 1

                if file_valid_docs % 50 == 0:
                    print(
                        f"  Progress: loaded {file_valid_docs} valid records from {data_path.name} "
                        f"(global loaded: {total_read_count})"
                    )

        print(f"  Finished file      : {data_path.name}")
        print(f"  Total lines read   : {file_total_lines}")
        print(f"  Valid docs loaded  : {file_valid_docs}")
        print(f"  Invalid JSON lines : {file_invalid}")

        if max_records is not None and total_read_count >= max_records:
            print(f"\nReached max_records={max_records}, stop reading more files.")
            break

    _print_section("Deduplication Summary")

    documents, duplicate_count, empty_count = _dedupe_documents(raw_documents)

    print(f"Raw documents collected : {len(raw_documents)}")
    print(f"Unique documents        : {len(documents)}")
    print(f"Duplicate documents     : {duplicate_count}")
    print(f"Empty documents         : {empty_count}")
    print(f"Invalid JSON lines      : {invalid_count}")

    total_unique_docs = len(documents)

    _print_section(f"Start indexing {total_unique_docs} unique documents")

    for doc_idx, document in enumerate(documents, start=1):
        remaining_docs = total_unique_docs - doc_idx

        print(
            f"\n[Doc {doc_idx}/{total_unique_docs}] "
            f"{document['doc_id']} ({document['source_file']}) | Remaining docs: {remaining_docs}"
        )

        state = await index_graph.ainvoke({
            "doc_id": document["doc_id"],
            "content": document["content"]
        })

        chunks = state.get("chunks", {})
        graph_delta = state.get("graph_delta", {})

        chunk_count = len(chunks)
        entity_count = len(graph_delta.get("nodes", []))
        relation_count = len(graph_delta.get("edges", []))

        global_graph = GraphStore().load()
        total_entities = len(global_graph.get("nodes", []))
        total_relations = len(global_graph.get("edges", []))

        print("  Done")
        print(f"  Chunks created            : {chunk_count}")
        print(f"  Entities (new or updated) : {entity_count}")
        print(f"  Relations (new or updated): {relation_count}")
        print(f"  Total entities in graph   : {total_entities}")
        print(f"  Total relations in graph  : {total_relations}")

    _print_section("Data loading completed")
    print(f"Output directory      : {WORKING_DIR}")
    print(f"Total raw documents   : {len(raw_documents)}")
    print(f"Total unique indexed  : {len(documents)}")
    print(f"Total duplicates      : {duplicate_count}")
    print(f"Total empty documents : {empty_count}")
    print(f"Total invalid JSON    : {invalid_count}")

async def run():
    ensure_workspace_dir()

    config = Config(provider="openai")

    _print_section("Build graph")
    print(f"Working/output dir: {WORKING_DIR}")

    graph = build_index_graph(config)
    await load_data(graph, max_records=None)

    _print_section("Done")


if __name__ == "__main__":
    asyncio.run(run())
