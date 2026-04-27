from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

Provider = Literal["openai", "local"]


@dataclass
class Config:
    provider: Provider = "local"

    llm_base_url: str = field(init=False)
    embedding_base_url: str = field(init=False)

    llm_api_key: str = ''
    embedding_api_key: str = ''

    llm_model: str = field(init=False)
    embedding_model: str = field(init=False)

    chunk_size: int = 1200
    chunk_overlap: int = 100

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

    def __post_init__(self):
        if self.provider == "local":
            self.llm_base_url = os.getenv(
                "LOCAL_LLM_BASE_URL", "http://127.0.0.1:18888/v1"
            )
            self.embedding_base_url = os.getenv(
                "LOCAL_EMBEDDING_BASE_URL", "http://127.0.0.1:18889/v1"
            )

            self.llm_model = os.getenv(
                "LOCAL_LLM_MODEL", "Qwen3-30B-A3B-Instruct-2507"
            )
            self.embedding_model = os.getenv(
                "LOCAL_EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"
            )

        elif self.provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY")

            self.llm_base_url = base_url
            self.embedding_base_url = base_url
            self.llm_api_key = api_key
            self.embedding_api_key = api_key

            self.llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
            self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

if __name__ == "__main__":
    config = Config("openai")
    print(config)
