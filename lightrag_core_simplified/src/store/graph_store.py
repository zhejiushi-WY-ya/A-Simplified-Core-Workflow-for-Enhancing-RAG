import json

from ..utils.json_file import save, load
from ..runtime_paths import workspace_file


class GraphStore:
    def __init__(self):
        self.path = workspace_file("graph.json")
        self.entity_index_path = workspace_file("graph_entity_index.json")
        self.relation_index_path = workspace_file("graph_relation_index.json")

        self.data = load(self.path) or {"nodes": [], "edges": []}
        self.entity_index = load(self.entity_index_path)
        self.relation_index = load(self.relation_index_path)

        if self.entity_index is None or self.relation_index is None:
            self.entity_index = {}
            self.relation_index = {}
            self._build_indexes()
            self._save_indexes()

    def _build_indexes(self):
        for idx, node in enumerate(self.data.get("nodes", [])):
            key = str(node.get("name", "")).strip().lower()
            if key:
                self.entity_index[key] = idx

        for idx, edge in enumerate(self.data.get("edges", [])):
            source = str(edge.get("source", "")).strip().lower()
            target = str(edge.get("target", "")).strip().lower()
            if source and target:
                self.relation_index[self._relation_key(source, target)] = idx

    def _relation_key(self, source, target):
        return json.dumps([source, target], ensure_ascii=False)

    def _save_indexes(self):
        save(self.entity_index_path, self.entity_index)
        save(self.relation_index_path, self.relation_index)

    def get_node(self, key):
        idx = self.entity_index.get(key)
        if idx is None:
            return None
        return dict(self.data["nodes"][idx])

    def upsert_node(self, key, node):
        idx = self.entity_index.get(key)
        if idx is None:
            self.data["nodes"].append(node)
            self.entity_index[key] = len(self.data["nodes"]) - 1
            return node

        self.data["nodes"][idx] = node
        return node

    def get_edge(self, source_key, target_key):
        idx = self.relation_index.get(self._relation_key(source_key, target_key))
        if idx is None:
            return None
        return dict(self.data["edges"][idx])

    def upsert_edge(self, source_key, target_key, edge):
        relation_key = self._relation_key(source_key, target_key)
        idx = self.relation_index.get(relation_key)
        if idx is None:
            self.data["edges"].append(edge)
            self.relation_index[relation_key] = len(self.data["edges"]) - 1
            return edge

        self.data["edges"][idx] = edge
        return edge

    def upsert(self, g):
        self.data = g
        self.entity_index = {}
        self.relation_index = {}
        self._build_indexes()
        save(self.path, self.data)
        self._save_indexes()
        return self.data

    def save(self):
        save(self.path, self.data)
        self._save_indexes()
        return self.data

    def load(self):
        return self.data
