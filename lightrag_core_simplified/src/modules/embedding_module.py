from ..llm import get_embedding_client
from ..store.vector_store import VectorStore


def _clean_texts(texts, name="unknown"):
    if texts is None:
        return []

    texts = list(texts)

    cleaned = []
    for i, t in enumerate(texts):
        if t is None:
            continue

        t = str(t).strip()
        if not t:
            continue

        cleaned.append(t)

    if not cleaned:
        print(f"[EMBED][WARN] {name} is empty after cleaning")

    else:
        print(f"[EMBED] {name}: count={len(cleaned)}")
        print(f"[EMBED] {name}: sample='{cleaned[0][:100]}'")

    return cleaned


def run(config, chunks, graph):
    client = get_embedding_client(config)

    def embed(texts, name="unknown"):
        texts = _clean_texts(texts, name)

        if not texts:
            return []

        try:
            res = client.embeddings.create(
                model=config.embedding_model,
                input=texts
            )
            return [x.embedding for x in res.data]

        except Exception as e:
            print(f"[EMBED][ERROR] batch failed: {repr(e)}")
            print("[EMBED] fallback to single mode...")

            vectors = []
            for i, text in enumerate(texts, start=1):
                print(f"[EMBED] single {i}/{len(texts)} len={len(text)}")

                res = client.embeddings.create(
                    model=config.embedding_model,
                    input=text,   # 单条
                )
                vectors.append(res.data[0].embedding)

            return vectors

    chunk_store = VectorStore("chunks")
    entity_store = VectorStore("entities")
    relation_store = VectorStore("relations")

    # ------------------ chunks ------------------
    chunk_ids = []
    chunk_texts = []

    for cid, c in chunks.items():
        text = c.get("content")
        text = str(text).strip() if text else ""
        if text:
            chunk_ids.append(cid)
            chunk_texts.append(text)

    if chunk_ids:
        chunk_vec = embed(chunk_texts, "chunks")
        chunk_store.upsert(chunk_ids, chunk_vec)

    # ------------------ entities ------------------
    entity_names = [n.get("name") for n in graph.get("nodes", [])]
    entity_names = _clean_texts(entity_names, "entities")

    if entity_names:
        entity_vec = embed(entity_names, "entities")
        entity_store.upsert(entity_names, entity_vec)

    # ------------------ relations ------------------
    rel_ids = []
    rel_texts = []

    for r in graph.get("edges", []):
        source = str(r.get("source", "")).strip()
        target = str(r.get("target", "")).strip()

        if not source or not target:
            continue

        rel_ids.append(f"{source}->{target}")
        rel_texts.append(f"{source} {target}")

    if rel_ids:
        rel_vec = embed(rel_texts, "relations")
        relation_store.upsert(rel_ids, rel_vec)

    chunk_store.save()
    entity_store.save()
    relation_store.save()
