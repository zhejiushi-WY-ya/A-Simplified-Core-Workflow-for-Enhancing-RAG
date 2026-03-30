from ..utils.json_file import save, load


class VectorStore:
    def __init__(self, name):
        self.path = f"./out/vdb_{name}.json"
        self.vectors = load(self.path) or {}

    def upsert(self, ids, vecs):
        for i, v in zip(ids, vecs):
            self.vectors[i] = v

    def save(self):
        save(self.path, self.vectors)
