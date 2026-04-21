import json
import math

from ..llm import get_llm_client

from ..prompt_templates import (
    ANSWER_PROMPT,
    CONTEXT_COMPRESSION_PROMPT,
    build_keywords_extraction_prompt,
)
from ..store.chunk_store import ChunkStore
from ..store.graph_store import GraphStore
from ..store.kv_store import KVStore
from ..store.vector_store import VectorStore
from ..utils.json_file import save
from ..utils.json_parser import safe_json


def cos(a, b):
    if len(a) != len(b):
        return None
    denom = np_l2(a) * np_l2(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def np_l2(vec):
    return math.sqrt(sum(x * x for x in vec))


def score_vectors(query_vec, vectors):
    query_dim = len(query_vec)
    scores = []
    for key, value in vectors.items():
        if not value:
            continue
        if len(value) != query_dim:
            # Skip corrupted/incompatible vectors instead of silently truncating via zip().
            continue
        sim = cos(query_vec, value)
        if sim is None:
            continue
        scores.append((key, sim))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def dedupe_scored_items(items, top_k):
    deduped = []
    seen = set()

    for item in items:
        item_id = item[0]
        if item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    return deduped


def extract_keywords(config, client, query):
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[{
            "role": "user",
            "content": build_keywords_extraction_prompt(
                query=query,
                language="the same language as the user query",
            ),
        }],
    )

    parsed = safe_json(response.choices[0].message.content)
    high_level = parsed.get("high_level_keywords", []) or []
    low_level = parsed.get("low_level_keywords", []) or []

    return {
        "high_level_keywords": [str(item).strip() for item in high_level if str(item).strip()],
        "low_level_keywords": [str(item).strip() for item in low_level if str(item).strip()],
    }


def get_seed_nodes(top_relations):
    nodes = set()

    for rel_id, _ in top_relations:
        if "->" not in rel_id:
            continue
        source, target = rel_id.split("->", 1)
        nodes.add(source)
        nodes.add(target)

    return nodes


def get_seed_nodes_from_chunks(graph, chunk_store, top_chunks):
    doc_ids = set()

    for item in top_chunks:
        chunk_id = item[0]
        chunk = chunk_store.get(chunk_id) or {}
        doc_ids.update(chunk.get("doc_ids", []))

    nodes = set()
    for node in graph.get("nodes", []):
        if doc_ids.intersection(node.get("source_doc_ids", [])):
            nodes.add(node["name"])

    return nodes


def get_seed_nodes_from_keywords(graph, keywords):
    nodes = set()
    keyword_set = {keyword.lower() for keyword in keywords}
    if not keyword_set:
        return nodes

    for node in graph.get("nodes", []):
        name = node.get("name", "")
        name_lower = name.lower()
        description = str(node.get("description", "")).lower()
        if any(keyword in name_lower or keyword in description for keyword in keyword_set):
            nodes.add(name)

    return nodes


def multi_hop_expand(graph, seed_nodes, hops=2, max_nodes=30):
    nodes = set(seed_nodes)
    if not nodes:
        return {"nodes": [], "edges": []}

    for _ in range(hops):
        expanded = set(nodes)

        for edge in graph.get("edges", []):
            if edge["source"] in nodes or edge["target"] in nodes:
                expanded.add(edge["source"])
                expanded.add(edge["target"])

        nodes = expanded
        if len(nodes) >= max_nodes:
            break

    ordered_nodes = []
    selected = set()
    for node in graph.get("nodes", []):
        name = node.get("name")
        if name in nodes and name not in selected:
            ordered_nodes.append(node)
            selected.add(name)
        if len(ordered_nodes) >= max_nodes:
            break

    selected_names = {node["name"] for node in ordered_nodes}
    edges = [
        edge
        for edge in graph.get("edges", [])
        if edge["source"] in selected_names or edge["target"] in selected_names
    ]

    return {"nodes": ordered_nodes, "edges": edges}


def filter_subgraph(subgraph, top_relations, max_edges=20):
    if not subgraph["edges"]:
        return subgraph

    important = set()
    for rel_id, _ in top_relations:
        if "->" not in rel_id:
            continue
        source, target = rel_id.split("->", 1)
        important.add(source)
        important.add(target)

    if not important:
        return {
            "nodes": subgraph["nodes"],
            "edges": subgraph["edges"][:max_edges],
        }

    edges = []
    for edge in subgraph["edges"]:
        if edge["source"] in important or edge["target"] in important:
            edges.append(edge)
        if len(edges) >= max_edges:
            break

    return {
        "nodes": subgraph["nodes"],
        "edges": edges,
    }


def select_kv_items(kv_items, subgraph, top_chunks, chunk_store, max_facts):
    node_names = {node["name"] for node in subgraph.get("nodes", [])}
    chunk_ids = {item[0] for item in top_chunks}
    doc_ids = set()

    for chunk_id in chunk_ids:
        chunk = chunk_store.get(chunk_id) or {}
        doc_ids.update(chunk.get("doc_ids", []))

    selected = []
    for item in kv_items:
        keys = item.get("key", [])
        item_chunk_ids = set(item.get("source_chunk_ids", []))
        item_doc_ids = set(item.get("source_doc_ids", []))

        if item_chunk_ids.intersection(chunk_ids):
            selected.append(item)
        elif item_doc_ids.intersection(doc_ids):
            selected.append(item)
        elif any(key in node_names for key in keys):
            selected.append(item)

        if len(selected) >= max_facts:
            break

    return selected


def collect_related_doc_ids(subgraph, top_entities, top_relations, chunk_store, top_chunks):
    doc_ids = set()

    for node in subgraph.get("nodes", []):
        doc_ids.update(node.get("source_doc_ids", []))

    for edge in subgraph.get("edges", []):
        doc_ids.update(edge.get("source_doc_ids", []))

    for entity_name, _ in top_entities:
        for node in subgraph.get("nodes", []):
            if node.get("name") == entity_name:
                doc_ids.update(node.get("source_doc_ids", []))

    relation_ids = {relation_id for relation_id, _ in top_relations}
    for edge in subgraph.get("edges", []):
        edge_id = f"{edge.get('source', '')}->{edge.get('target', '')}"
        if edge_id in relation_ids:
            doc_ids.update(edge.get("source_doc_ids", []))

    for chunk_id, _ in top_chunks:
        chunk = chunk_store.get(chunk_id) or {}
        doc_ids.update(chunk.get("doc_ids", []))

    return doc_ids


def expand_chunks_with_graph_context(chunk_scores, chunk_store, doc_ids, top_k):
    if not doc_ids:
        return []

    expanded = []
    for chunk_id, score in chunk_scores:
        chunk = chunk_store.get(chunk_id) or {}
        chunk_doc_ids = set(chunk.get("doc_ids", []))
        if chunk_doc_ids.intersection(doc_ids):
            expanded.append((chunk_id, score))
        if len(expanded) >= top_k:
            break

    return expanded


def build_structured_context(config, subgraph, kv_items, top_chunks, chunk_store):
    entities = []
    for node in subgraph.get("nodes", [])[:config.max_context_entities]:
        entities.append({
            "entity_name": node["name"],
            "entity_type": node.get("type", "Other"),
            "description": node.get("description", ""),
            "source_doc_ids": node.get("source_doc_ids", []),
        })

    relations = []
    for edge in subgraph.get("edges", [])[:config.max_context_relations]:
        relations.append({
            "source": edge["source"],
            "target": edge["target"],
            "keywords": edge.get("keywords", []),
            "description": edge.get("description", ""),
            "source_doc_ids": edge.get("source_doc_ids", []),
        })

    facts = []
    for item in kv_items[:config.max_context_facts]:
        facts.append({
            "key": item.get("key", []),
            "value": item.get("value", ""),
            "source_doc_ids": item.get("source_doc_ids", []),
        })

    chunks = []
    references = []
    for idx, chunk_item in enumerate(top_chunks[:config.max_context_chunks], start=1):
        chunk_id, score = chunk_item[:2]
        chunk = chunk_store.get(chunk_id) or {}
        reference_id = f"ref_{idx}"
        doc_ids = chunk.get("doc_ids", [])
        title = doc_ids[0] if doc_ids else chunk_id

        chunks.append({
            "reference_id": reference_id,
            "chunk_id": chunk_id,
            "score": round(score, 4),
            "doc_ids": doc_ids,
            "content": chunk.get("content", ""),
        })
        references.append(f"[{reference_id}] {title}")

    context_data = f"""
Knowledge Graph Data (Entity):

```json
{json.dumps(entities, ensure_ascii=False, indent=2)}
```

Knowledge Graph Data (Relationship):

```json
{json.dumps(relations, ensure_ascii=False, indent=2)}
```

Supporting Facts:

```json
{json.dumps(facts, ensure_ascii=False, indent=2)}
```

Document Chunks (Each entry has a reference_id refer to the Reference Document List):

```json
{json.dumps(chunks, ensure_ascii=False, indent=2)}
```

Reference Document List:

```
{chr(10).join(references)}
```
""".strip()

    return {
        "entities": entities,
        "relations": relations,
        "facts": facts,
        "chunks": chunks,
        "references": references,
        "context_data": context_data,
    }


def compress_context(config, client, query, context):
    prompt = CONTEXT_COMPRESSION_PROMPT.format(
        query=query,
        context=context,
    )

    res = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content


def generate_answer(config, client, query, context_data):
    prompt = ANSWER_PROMPT.format(
        context_data=context_data,
        query=query,
    )

    res = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content


def retrieve(config, query, mode=None):
    retrieval_mode = mode or config.retrieval_mode
    client = get_llm_client(config)

    keywords = extract_keywords(config, client, query)
    retrieval_query = " | ".join(
        [query] + keywords["high_level_keywords"] + keywords["low_level_keywords"]
    )

    query_vec = client.embeddings.create(
        model=config.embedding_model,
        input=retrieval_query
    ).data[0].embedding

    chunk_vector_store = VectorStore("chunks")
    entity_vector_store = VectorStore("entities")
    relation_vector_store = VectorStore("relations")
    chunk_store = ChunkStore()
    kv_items = KVStore().data
    graph = GraphStore().load()

    chunk_scores = score_vectors(query_vec, chunk_vector_store.vectors)
    entity_scores = score_vectors(query_vec, entity_vector_store.vectors)
    relation_scores = score_vectors(query_vec, relation_vector_store.vectors)

    top_chunks = chunk_scores[:config.top_k_chunks]
    top_entities = entity_scores[:config.top_k_entities]
    top_relations = relation_scores[:config.top_k_relations]

    keyword_seed_nodes = get_seed_nodes_from_keywords(
        graph,
        keywords["low_level_keywords"] + keywords["high_level_keywords"],
    )

    if retrieval_mode == "chunk":
        seed_nodes = get_seed_nodes_from_chunks(graph, chunk_store, top_chunks)
    else:
        seed_nodes = get_seed_nodes(top_relations)
        seed_nodes.update(entity_id for entity_id, _ in top_entities)
        seed_nodes.update(keyword_seed_nodes)
        if not seed_nodes:
            seed_nodes = get_seed_nodes_from_chunks(graph, chunk_store, top_chunks)

    subgraph = multi_hop_expand(
        graph,
        seed_nodes,
        hops=config.subgraph_hops,
        max_nodes=config.max_subgraph_nodes,
    )

    if retrieval_mode == "graph":
        top_chunks = []
    elif retrieval_mode == "chunk":
        top_relations = []

    subgraph = filter_subgraph(
        subgraph,
        top_relations,
        max_edges=config.max_subgraph_edges,
    )
    related_doc_ids = collect_related_doc_ids(
        subgraph,
        top_entities,
        top_relations,
        chunk_store,
        top_chunks,
    )
    graph_context_chunks = expand_chunks_with_graph_context(
        chunk_scores,
        chunk_store,
        related_doc_ids,
        config.max_context_chunks + 2,
    )
    top_chunks = dedupe_scored_items(
        top_chunks + graph_context_chunks,
        config.top_k_chunks,
    )
    selected_kv = select_kv_items(
        kv_items,
        subgraph,
        top_chunks,
        chunk_store,
        config.max_context_facts,
    )
    structured_context = build_structured_context(
        config,
        subgraph,
        selected_kv,
        top_chunks,
        chunk_store,
    )
    compressed_context = compress_context(
        config,
        client,
        query,
        structured_context["context_data"],
    )

    return {
        "mode": retrieval_mode,
        "query": query,
        "keywords": keywords,
        "entities": top_entities,
        "chunks": top_chunks,
        "relations": top_relations,
        "subgraph": subgraph,
        "facts": selected_kv,
        "raw_context": structured_context["context_data"],
        "context": compressed_context,
        "references": structured_context["references"],
        "context_chunks": structured_context["chunks"],
    }


def run(config, query, mode=None):
    client = get_llm_client(config)
    retrieval = retrieve(config, query, mode=mode)
    answer_context = f"{retrieval['raw_context']}\n\nCompressed Reasoning Notes:\n{retrieval['context']}"
    answer = generate_answer(config, client, query, answer_context)

    result = {
        **retrieval,
        "answer": answer,
    }

    save("./out/retrieval.json", result)
    return result
