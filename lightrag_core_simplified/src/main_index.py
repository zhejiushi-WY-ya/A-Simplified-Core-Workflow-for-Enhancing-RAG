import asyncio
import hashlib
import json
from pathlib import Path

from langgraph.graph import END, StateGraph

from .config import Config
from .nodes.chunk_node import build_node as chunk
from .nodes.embedding_node import build_node as embed
from .nodes.graph_node import build_node as graph


BASE_DIR = Path(__file__).resolve().parent
DATA_PATHS = [
    BASE_DIR / "raw_data" / "agriculture.jsonl",
    BASE_DIR / "raw_data" / "cs.jsonl",
    BASE_DIR / "raw_data" / "legal.jsonl",
    BASE_DIR / "raw_data" / "mix.jsonl",
]
INPUT_DIR = BASE_DIR / "input"


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

    input_dir = INPUT_DIR

    _print_section("Indexing Started")
    print(f"Current working dir : {Path.cwd()}")
    print(f"Script base dir     : {BASE_DIR}")
    print("DATA_PATHS:")
    for p in DATA_PATHS:
        print(f"  - {p}")
    print(f"INPUT_DIR           : {input_dir}")

    # 优先走 raw_data 下的四个 jsonl
    existing_data_paths = [p for p in DATA_PATHS if p.exists()]

    if existing_data_paths:
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
        print(f"Total raw documents   : {len(raw_documents)}")
        print(f"Total unique indexed  : {len(documents)}")
        print(f"Total duplicates      : {duplicate_count}")
        print(f"Total empty documents : {empty_count}")
        print(f"Total invalid JSON    : {invalid_count}")
        return

    # raw_data 不存在时，退回 input/
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Neither raw data files nor input dir exists.\n"
            f"DATA_PATHS: {DATA_PATHS}\n"
            f"INPUT_DIR: {input_dir}"
        )

    _print_section("raw_data not found, fallback to input/")

    doc_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )

    raw_documents = []

    for idx, path in enumerate(doc_paths, start=1):
        print(f"[Input File {idx}/{len(doc_paths)}] Loading {path.name}")
        raw_documents.append({
            "doc_id": path.stem,
            "content": path.read_text(encoding="utf-8"),
            "path": path,
        })

    documents, duplicate_count, empty_count = _dedupe_documents(raw_documents)

    _print_section("Input/ Deduplication Summary")
    print(f"Raw documents collected : {len(raw_documents)}")
    print(f"Unique documents        : {len(documents)}")
    print(f"Duplicate documents     : {duplicate_count}")
    print(f"Empty documents         : {empty_count}")

    total_unique_docs = len(documents)

    _print_section(f"Start indexing {total_unique_docs} unique input documents")

    for doc_idx, document in enumerate(documents, start=1):
        remaining_docs = total_unique_docs - doc_idx

        path = document.get("path")
        label = path.name if path is not None else document["doc_id"]

        print(f"\n[Doc {doc_idx}/{total_unique_docs}] {label} | Remaining docs: {remaining_docs}")

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
    print(f"Total raw documents   : {len(raw_documents)}")
    print(f"Total unique indexed  : {len(documents)}")
    print(f"Total duplicates      : {duplicate_count}")
    print(f"Total empty documents : {empty_count}")


def build_graph(config):
    g = StateGraph(dict)

    g.add_node("chunk", chunk(config))
    g.add_node("graph", graph(config))
    g.add_node("embedding", embed(config))

    g.set_entry_point("chunk")

    g.add_edge("chunk", "graph")
    g.add_edge("graph", "embedding")
    g.add_edge("embedding", END)

    return g.compile()


async def run():
    config = Config(
        base_url="https://rtekkxiz.bja.sealos.run/v1",
        api_key="sk-eGYT382xngt2u4kGGnxInmjYvqloG8ltr07UbSKvo7w2uBI7"
    )

    _print_section("Build graph")
    graph = build_graph(config)

    await load_data(graph, max_records=None)

    _print_section("Done")


if __name__ == "__main__":
    asyncio.run(run())
