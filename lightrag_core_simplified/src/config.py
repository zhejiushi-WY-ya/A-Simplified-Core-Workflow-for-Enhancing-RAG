from dataclasses import dataclass


@dataclass
class Config:
    base_url: str
    api_key: str

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    chunk_size: int = 1200
    chunk_overlap: int = 200

    retrieval_mode: str = "hybrid"
    top_k_chunks: int = 8
    top_k_entities: int = 10
    top_k_relations: int = 8
    max_subgraph_nodes: int = 40
    max_subgraph_edges: int = 30
    max_context_chunks: int = 5
    max_context_entities: int = 25
    max_context_relations: int = 25
    max_context_facts: int = 8
    subgraph_hops: int = 3
