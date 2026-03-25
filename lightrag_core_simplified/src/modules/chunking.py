from lightrag.operate import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer, compute_mdhash_id
from ..io.file_io import save


def run(config, text, doc_id):
    tokenizer = TiktokenTokenizer("gpt-4o-mini")

    chunk_list = chunking_by_token_size(
        tokenizer=tokenizer,
        content=text,
        chunk_token_size=config.chunk_size,
        chunk_overlap_token_size=config.chunk_overlap,
    )

    chunks = {}
    for c in chunk_list:
        cid = compute_mdhash_id(c["content"], prefix="chunk-")
        chunks[cid] = c

    save("./out/chunks.json", chunks)

    return chunks
