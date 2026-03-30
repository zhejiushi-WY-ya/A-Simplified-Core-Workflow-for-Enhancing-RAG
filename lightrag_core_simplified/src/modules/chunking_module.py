from lightrag.operate import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer, compute_mdhash_id
from ..store.chunk_store import ChunkStore


def run(config, text, doc_id):
    tokenizer = TiktokenTokenizer("gpt-4o-mini")
    chunk_store = ChunkStore()

    chunk_list = chunking_by_token_size(
        tokenizer=tokenizer,
        content=text,
        chunk_token_size=config.chunk_size,
        chunk_overlap_token_size=config.chunk_overlap,
    )

    chunks = {}

    for order_index, c in enumerate(chunk_list):
        cid = compute_mdhash_id(c["content"], prefix="chunk-")
        chunk = {
            **c,
            "id": cid,
            "doc_id": doc_id,
            "doc_ids": [doc_id],
            "chunk_order_index": order_index,
        }
        existing = chunk_store.get(cid)
        doc_is_new_for_chunk = existing is None or doc_id not in existing.get("doc_ids", [])
        merged = chunk_store.upsert(cid, chunk)

        if doc_is_new_for_chunk:
            chunks[cid] = merged

    chunk_store.save()

    return chunks
