import base64
import struct

from ..utils.json_file import save, load


class VectorStore:
    def __init__(self, name):
        self.path = f"./exp_data/vdb_{name}.json"
        self.vectors = self._decode_store(load(self.path) or {})

    def upsert(self, ids, vecs):
        for i, v in zip(ids, vecs):
            self.vectors[i] = [float(x) for x in v]

    def save(self):
        encoded = {key: self._encode_vector(value) for key, value in self.vectors.items()}
        save(self.path, encoded)

    @staticmethod
    def _encode_vector(vector):
        if not vector:
            return ""
        packed = struct.pack(f"<{len(vector)}f", *vector)
        return base64.b64encode(packed).decode("ascii")

    @staticmethod
    def _decode_vector(encoded):
        if not encoded:
            return []
        raw = base64.b64decode(encoded)
        if len(raw) % 4 != 0:
            raise ValueError("Corrupted vector payload: byte length is not divisible by 4.")
        count = len(raw) // 4
        return list(struct.unpack(f"<{count}f", raw))

    def _decode_store(self, raw_store):
        decoded = {}
        for key, value in raw_store.items():
            if isinstance(value, str):
                decoded[key] = self._decode_vector(value)
            elif isinstance(value, list):
                # Backward compatibility for legacy JSON float list storage.
                decoded[key] = [float(x) for x in value]
            else:
                raise TypeError(f"Unsupported vector payload type for key '{key}': {type(value).__name__}")
        return decoded
