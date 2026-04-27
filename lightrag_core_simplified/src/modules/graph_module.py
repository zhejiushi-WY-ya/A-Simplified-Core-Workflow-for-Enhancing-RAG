import json

from ..llm import get_llm_client
from lightrag.utils import normalize_extracted_info

from ..prompt_templates import (
    GRAPH_ENTITY_TYPES,
    INDEX_EXTRACTION_PROMPT,
    build_description_summary_prompt,
)
from ..utils.json_file import save
from ..store.graph_store import GraphStore
from ..store.kv_store import KVStore
from ..utils.json_parser import safe_json


def _ordered_union(left, right):
    merged = []
    seen = set()

    for value in list(left or []) + list(right or []):
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)

    return merged


def _merge_text(current, incoming):
    parts = []
    seen = set()

    for value in [current, incoming]:
        if not value:
            continue
        for part in str(value).split("\n"):
            text = part.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            parts.append(text)

    return "\n".join(parts)


def _normalize_name(value):
    text = str(value or "").strip()
    if not text:
        return ""
    return normalize_extracted_info(text, remove_inner_quotes=True).strip()


def _normalize_description(value):
    return str(value or "").strip()


def _unique_lines(text):
    lines = []
    seen = set()

    for part in str(text or "").split("\n"):
        item = part.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        lines.append(item)

    return lines


def _entity_key(name):
    normalized = _normalize_name(name)
    return normalized.lower()


def _relation_key(source, target):
    return (_normalize_name(source).lower(), _normalize_name(target).lower())


def _merge_entity(current, incoming):
    current_desc = _unique_lines(current.get("description", ""))
    incoming_desc = _unique_lines(incoming.get("description", ""))
    return {
        **current,
        **incoming,
        "name": incoming.get("name") or current.get("name", ""),
        "type": incoming.get("type") or current.get("type", "Other"),
        "description": _merge_text(current.get("description", ""), incoming.get("description", "")),
        "description_list": _ordered_union(
            current.get("description_list", current_desc),
            incoming.get("description_list", incoming_desc),
        ),
        "source_chunk_ids": _ordered_union(
            current.get("source_chunk_ids", []),
            incoming.get("source_chunk_ids", []),
        ),
        "source_doc_ids": _ordered_union(
            current.get("source_doc_ids", []),
            incoming.get("source_doc_ids", []),
        ),
    }


def _merge_relation(current, incoming):
    current_desc = _unique_lines(current.get("description", ""))
    incoming_desc = _unique_lines(incoming.get("description", ""))
    return {
        **current,
        **incoming,
        "source": incoming.get("source") or current.get("source", ""),
        "target": incoming.get("target") or current.get("target", ""),
        "keywords": _ordered_union(
            current.get("keywords", []),
            incoming.get("keywords", []),
        ),
        "description": _merge_text(current.get("description", ""), incoming.get("description", "")),
        "description_list": _ordered_union(
            current.get("description_list", current_desc),
            incoming.get("description_list", incoming_desc),
        ),
        "source_chunk_ids": _ordered_union(
            current.get("source_chunk_ids", []),
            incoming.get("source_chunk_ids", []),
        ),
        "source_doc_ids": _ordered_union(
            current.get("source_doc_ids", []),
            incoming.get("source_doc_ids", []),
        ),
    }


def _description_list_changed(current, merged):
    return current.get("description_list", []) != merged.get("description_list", [])


def _apply_description_if_no_summary_needed(current, merged):
    descriptions = merged.get("description_list", [])
    if len(descriptions) <= 1:
        merged["description"] = descriptions[0] if descriptions else ""
        return merged

    if not _description_list_changed(current, merged) and current.get("description"):
        merged["description"] = current.get("description", "")

    return merged


def _summarize_description(client, config, description_type, description_name, descriptions):
    cleaned = [item.strip() for item in descriptions if str(item).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]

    description_list = "\n".join(
        json.dumps({"description": item}, ensure_ascii=False)
        for item in cleaned
    )

    prompt = build_description_summary_prompt(
        description_type=description_type,
        description_name=description_name,
        description_list=description_list,
    )
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip() or cleaned[0]


def run(config, chunks):
    graph_store = GraphStore()
    changed_entities = set()
    changed_relations = set()
    summary_entities = set()
    summary_relations = set()

    if chunks:
        client = get_llm_client(config)

        for chunk_id, chunk in chunks.items():
            res = client.chat.completions.create(
                model=config.llm_model,
                messages=[{
                    "role": "user",
                    "content": INDEX_EXTRACTION_PROMPT.format(
                        entity_types=", ".join(GRAPH_ENTITY_TYPES),
                        input_text=chunk["content"],
                    ),
                }]
            )

            data = safe_json(res.choices[0].message.content)

            for entity in data.get("entities", []):
                name = _normalize_name(entity.get("name", ""))
                if not name:
                    continue

                key = name.lower()
                incoming = {
                    "name": name,
                    "type": entity.get("type", "Other"),
                    "description": _normalize_description(
                        entity.get("description", "")
                    ),
                    "source_chunk_ids": [chunk_id],
                    "source_doc_ids": list(chunk.get("doc_ids", [])),
                }
                current_entity = graph_store.get_node(key) or {}
                merged_entity = _merge_entity(current_entity, incoming)
                merged_entity = _apply_description_if_no_summary_needed(
                    current_entity,
                    merged_entity,
                )
                graph_store.upsert_node(key, merged_entity)
                changed_entities.add(key)
                if (
                    _description_list_changed(current_entity, merged_entity)
                    and len(merged_entity.get("description_list", [])) > 1
                ):
                    summary_entities.add(key)

            for relation in data.get("relations", []):
                source = _normalize_name(relation.get("source", ""))
                target = _normalize_name(relation.get("target", ""))
                if not source or not target:
                    continue

                key = (source.lower(), target.lower())
                incoming = {
                    "source": source,
                    "target": target,
                    "keywords": [
                        _normalize_name(keyword)
                        for keyword in relation.get("keywords", [])
                        if _normalize_name(keyword)
                    ],
                    "description": _normalize_description(
                        relation.get("description", "")
                    ),
                    "source_chunk_ids": [chunk_id],
                    "source_doc_ids": list(chunk.get("doc_ids", [])),
                }
                current_relation = graph_store.get_edge(key[0], key[1]) or {}
                merged_relation = _merge_relation(current_relation, incoming)
                merged_relation = _apply_description_if_no_summary_needed(
                    current_relation,
                    merged_relation,
                )
                graph_store.upsert_edge(key[0], key[1], merged_relation)
                changed_relations.add(key)
                if (
                    _description_list_changed(current_relation, merged_relation)
                    and len(merged_relation.get("description_list", [])) > 1
                ):
                    summary_relations.add(key)

        for key in summary_entities:
            entity = graph_store.get_node(key) or {}
            entity["description"] = _summarize_description(
                client,
                config,
                "Entity",
                entity["name"],
                entity.get("description_list", []),
            )
            graph_store.upsert_node(key, entity)

        for key in summary_relations:
            relation = graph_store.get_edge(key[0], key[1]) or {}
            relation_name = f"{relation['source']} -> {relation['target']}"
            relation["description"] = _summarize_description(
                client,
                config,
                "Relation",
                relation_name,
                relation.get("description_list", []),
            )
            graph_store.upsert_edge(key[0], key[1], relation)

    merged_graph = graph_store.save()

    save("./exp_data/entities.json", merged_graph["nodes"])
    save("./exp_data/relations.json", merged_graph["edges"])

    kv = []

    for key in changed_entities:
        entity = graph_store.get_node(key) or {}
        kv.append({
            "key": [entity["name"]],
            "value": entity.get("description", ""),
            "source_chunk_ids": entity.get("source_chunk_ids", []),
            "source_doc_ids": entity.get("source_doc_ids", []),
        })

    for key in changed_relations:
        relation = graph_store.get_edge(key[0], key[1]) or {}
        kv.append({
            "key": [relation["source"], relation["target"]],
            "value": relation.get("description", ""),
            "source_chunk_ids": relation.get("source_chunk_ids", []),
            "source_doc_ids": relation.get("source_doc_ids", []),
        })

    kv_store = KVStore()
    kv_store.upsert(kv)
    kv_store.save()

    return {
        "full": merged_graph,
        "delta": {
            "nodes": [graph_store.get_node(key) for key in changed_entities],
            "edges": [graph_store.get_edge(key[0], key[1]) for key in changed_relations],
        },
    }
