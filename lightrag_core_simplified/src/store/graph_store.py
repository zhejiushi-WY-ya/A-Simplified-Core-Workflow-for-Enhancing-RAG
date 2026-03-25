from ..io.file_io import save, load


class GraphStore:
    def __init__(self):
        self.path = "./out/graph.json"

    def save(self, g):
        save(self.path, g)

    def load(self):
        return load(self.path)
