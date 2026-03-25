from openai import OpenAI
from ..store.vector_store import VectorStore


def run(config, chunks, graph):
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def embed(texts):
        res = client.embeddings.create(
            model=config.embedding_model,
            input=texts
        )
        return [x.embedding for x in res.data]

    # 初始化 store（关键）
    chunk_store = VectorStore("chunks")
    entity_store = VectorStore("entities")
    relation_store = VectorStore("relations")

    # chunk
    chunk_ids = list(chunks.keys())
    chunk_vec = embed([c["content"] for c in chunks.values()])
    chunk_store.add(chunk_ids, chunk_vec)

    # entity
    entity_names = [n["name"] for n in graph["nodes"]]
    entity_vec = embed(entity_names)
    entity_store.add(entity_names, entity_vec)

    # relation
    rel_ids = [f"{r['source']}->{r['target']}" for r in graph["edges"]]
    rel_text = [r["source"] + " " + r["target"] for r in graph["edges"]]
    rel_vec = embed(rel_text)
    relation_store.add(rel_ids, rel_vec)

    chunk_store.save()
    entity_store.save()
    relation_store.save()
