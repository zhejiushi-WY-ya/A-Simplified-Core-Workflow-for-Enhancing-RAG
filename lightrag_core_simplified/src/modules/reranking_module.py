"""Context compression as reranking.

Separated from retrieval_module to conform to the rag_contracts Reranking protocol.
"""

from ..llm import get_llm_client
from ..prompt_templates import CONTEXT_COMPRESSION_PROMPT


def compress_context(config, client, query, context):
    prompt = CONTEXT_COMPRESSION_PROMPT.format(
        query=query,
        context=context,
    )

    res = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
    )

    return res.choices[0].message.content


def run(config, query, raw_context):
    """Compress structured retrieval context into a focused evidence brief."""
    client = get_llm_client(config)
    return compress_context(config, client, query, raw_context)
