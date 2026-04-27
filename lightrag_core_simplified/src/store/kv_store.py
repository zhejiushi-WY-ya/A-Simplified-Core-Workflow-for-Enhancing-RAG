from ..utils.json_file import save, load
from ..runtime_paths import workspace_file


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


class KVStore:
    def __init__(self):
        self.path = workspace_file("kv.json")
        self.data = load(self.path) or []
        self.index = {}

        for idx, item in enumerate(self.data):
            key = tuple(str(part).strip().lower() for part in item.get("key", []))
            if key:
                self.index[key] = idx

    def upsert(self, items):
        for item in items:
            key = tuple(str(part).strip().lower() for part in item.get("key", []))
            if not key:
                continue

            current_idx = self.index.get(key)
            if current_idx is None:
                self.data.append(item)
                self.index[key] = len(self.data) - 1
                continue

            current = self.data[current_idx]
            self.data[current_idx] = {
                **current,
                **item,
                "value": _merge_text(current.get("value", ""), item.get("value", "")),
                "source_chunk_ids": _ordered_union(
                    current.get("source_chunk_ids", []),
                    item.get("source_chunk_ids", []),
                ),
                "source_doc_ids": _ordered_union(
                    current.get("source_doc_ids", []),
                    item.get("source_doc_ids", []),
                ),
            }

    def save(self):
        save(self.path, self.data)
