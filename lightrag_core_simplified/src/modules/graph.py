from openai import OpenAI
from ..utils.json_parser import safe_json
from ..store.graph_store import GraphStore
from ..store.kv_store import KVStore
from ..io.file_io import save


def run(config, chunks):
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    entities, relations = [], []

    for c in chunks.values():
        prompt = f"""
Extract entities and relations.

Return JSON:
{{
 "entities":[{{"name":"...","description":"..."}}],
 "relations":[{{"source":"...","target":"...","description":"..."}}]
}}

ONLY JSON.

Text:
{c["content"]}
"""

        res = client.chat.completions.create(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )

        data = safe_json(res.choices[0].message.content)

        entities += data.get("entities", [])
        relations += data.get("relations", [])

    # 去重（LightRAG关键）
    uniq = {}
    for e in entities:
        uniq[e["name"].lower()] = e

    entities = list(uniq.values())

    graph = {"nodes": entities, "edges": relations}

    GraphStore().save(graph)

    save("./out/entities.json", entities)
    save("./out/relations.json", relations)

    # 🔥 profiling（KV）
    kv = []

    for e in entities:
        kv.append({
            "key": [e["name"]],
            "value": e.get("description", "")
        })

    for r in relations:
        kv.append({
            "key": [r["source"], r["target"]],
            "value": r.get("description", "")
        })

    kv_store = KVStore()
    kv_store.add(kv)
    kv_store.save()

    return graph
