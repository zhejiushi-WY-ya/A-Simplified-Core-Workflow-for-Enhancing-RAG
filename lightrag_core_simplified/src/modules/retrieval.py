import numpy as np
from openai import OpenAI
from ..store.vector_store import VectorStore
from ..store.kv_store import KVStore
from ..store.graph_store import GraphStore
from ..io.file_io import save


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =========================
# 1️⃣ seed nodes
# =========================
def get_seed_nodes(top_relations):
    nodes = set()

    for rel_id, _ in top_relations:
        if "->" in rel_id:
            s, t = rel_id.split("->")
            nodes.add(s)
            nodes.add(t)

    return nodes


# =========================
# 2️⃣ multi-hop expansion（核心）
# =========================
def multi_hop_expand(graph, seed_nodes, hops=2, max_nodes=30):
    nodes = set(seed_nodes)

    for _ in range(hops):
        new_nodes = set(nodes)

        for e in graph["edges"]:
            if e["source"] in nodes or e["target"] in nodes:
                new_nodes.add(e["source"])
                new_nodes.add(e["target"])

        nodes = new_nodes

        # 🔥 控制规模（防爆）
        if len(nodes) > max_nodes:
            break

    sub_nodes = [n for n in graph["nodes"] if n["name"] in nodes]
    sub_edges = [
        e for e in graph["edges"]
        if e["source"] in nodes or e["target"] in nodes
    ]

    return {"nodes": sub_nodes, "edges": sub_edges}


# =========================
# 3️⃣ path filtering（去噪）
# =========================
def filter_subgraph(subgraph, top_relations):
    important = set()

    for rel_id, _ in top_relations:
        if "->" in rel_id:
            s, t = rel_id.split("->")
            important.add(s)
            important.add(t)

    edges = []
    for e in subgraph["edges"]:
        if e["source"] in important or e["target"] in important:
            edges.append(e)

    return {
        "nodes": subgraph["nodes"],
        "edges": edges[:20]  # 控制边数量
    }


# =========================
# 4️⃣ context build（结构化）
# =========================
def build_context(subgraph, kv, chunks):
    parts = []

    parts.append("Entities:")
    for n in subgraph["nodes"][:20]:
        parts.append(f"- {n['name']}: {n.get('description','')}")

    parts.append("\nRelations:")
    for e in subgraph["edges"][:20]:
        parts.append(f"- {e['source']} -> {e['target']}: {e.get('description','')}")

    parts.append("\nFacts:")
    for k in kv[:5]:
        parts.append(f"- {k['value']}")

    parts.append("\nChunks:")
    for cid, _ in chunks[:3]:
        parts.append(f"- {cid}")

    return "\n".join(parts)


# =========================
# 5️⃣ context compression（🔥关键）
# =========================
def compress_context(config, context):
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    prompt = f"""
Compress the following knowledge into concise reasoning facts.

Keep only:
- key entities
- key relations
- causal links

{context}
"""

    res = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content


# =========================
# 6️⃣ answer
# =========================
def generate_answer(config, query, context):
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    prompt = f"""
Answer based on knowledge:

{context}

Question:
{query}
"""

    res = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content


# =========================
# 🔥 主流程
# =========================
def run(config, query):
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    # embedding
    q_vec = client.embeddings.create(
        model=config.embedding_model,
        input=query
    ).data[0].embedding

    chunk_store = VectorStore("chunks")
    rel_store = VectorStore("relations")

    # dual retrieval
    chunk_scores = [(k, cos(q_vec, v)) for k, v in chunk_store.vectors.items()]
    rel_scores = [(k, cos(q_vec, v)) for k, v in rel_store.vectors.items()]

    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    rel_scores.sort(key=lambda x: x[1], reverse=True)

    kv = KVStore().data
    graph = GraphStore().load()

    # 🔥 seed
    seed_nodes = get_seed_nodes(rel_scores[:5])

    # 🔥 multi-hop
    subgraph = multi_hop_expand(graph, seed_nodes, hops=2)

    # 🔥 filtering
    subgraph = filter_subgraph(subgraph, rel_scores[:5])

    # 🔥 context
    context = build_context(subgraph, kv, chunk_scores)

    # 🔥 compression（关键）
    compressed = compress_context(config, context)

    # 🔥 answer
    answer = generate_answer(config, query, compressed)

    result = {
        "chunks": chunk_scores[:5],
        "relations": rel_scores[:5],
        "subgraph": subgraph,
        "context": compressed,
        "answer": answer
    }

    save("./out/retrieval.json", result)

    return result
