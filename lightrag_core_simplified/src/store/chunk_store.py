from ..utils.json_file import save, load


def _ordered_union(left, right):
    merged = []
    seen = set()

    for value in list(left or []) + list(right or []):
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)

    return merged


class ChunkStore:
    def __init__(self):
        self.path = "./exp_data/chunks.json"
        self.data = load(self.path) or {}

    def get(self, chunk_id):
        return self.data.get(chunk_id)

    def upsert(self, chunk_id, chunk):
        current = self.data.get(chunk_id)
        if current is None:
            merged = {
                **chunk,
                "doc_ids": list(chunk.get("doc_ids", [])),
            }
        else:
            merged = {
                **current,
                **chunk,
                "doc_ids": _ordered_union(
                    current.get("doc_ids", []),
                    chunk.get("doc_ids", []),
                ),
            }

        self.data[chunk_id] = merged
        return merged

    def save(self):
        save(self.path, self.data)
