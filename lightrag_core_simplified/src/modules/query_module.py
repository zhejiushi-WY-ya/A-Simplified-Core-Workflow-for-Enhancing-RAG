"""Query expansion via LLM-based keyword extraction.

Separated from retrieval_module to conform to the rag_contracts Query protocol.
"""

from ..llm import get_llm_client
from ..prompt_templates import build_keywords_extraction_prompt
from ..utils.json_parser import safe_json


def extract_keywords(config, client, query):
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[{
            "role": "user",
            "content": build_keywords_extraction_prompt(
                query=query,
                language="the same language as the user query",
            ),
        }],
    )

    parsed = safe_json(response.choices[0].message.content)
    high_level = parsed.get("high_level_keywords", []) or []
    low_level = parsed.get("low_level_keywords", []) or []

    return {
        "high_level_keywords": [str(item).strip() for item in high_level if str(item).strip()],
        "low_level_keywords": [str(item).strip() for item in low_level if str(item).strip()],
    }


def build_retrieval_query(query, keywords):
    return " | ".join(
        [query] + keywords["high_level_keywords"] + keywords["low_level_keywords"]
    )


def embed_query(config, client, retrieval_query):
    return client.embeddings.create(
        model=config.embedding_model,
        input=retrieval_query,
    ).data[0].embedding


def run(config, query):
    """Full query expansion: keywords + composite query string + query vector."""
    client = get_llm_client(config)
    keywords = extract_keywords(config, client, query)
    retrieval_query = build_retrieval_query(query, keywords)
    query_vec = embed_query(config, client, retrieval_query)
    return {
        "keywords": keywords,
        "retrieval_query": retrieval_query,
        "query_vec": query_vec,
    }
