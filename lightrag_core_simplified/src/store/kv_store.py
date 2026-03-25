from ..io.file_io import save, load


class KVStore:
    def __init__(self):
        self.path = "./out/kv.json"
        self.data = load(self.path) or []

    def add(self, items):
        self.data.extend(items)

    def save(self):
        save(self.path, self.data)
