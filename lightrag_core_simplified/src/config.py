from dataclasses import dataclass


@dataclass
class Config:
    base_url: str
    api_key: str

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    chunk_size: int = 1200
    chunk_overlap: int = 200
