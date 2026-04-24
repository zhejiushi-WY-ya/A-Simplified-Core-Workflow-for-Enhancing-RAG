"""Answer generation from context.

Separated from retrieval_module to conform to the rag_contracts Generation protocol.
"""

from ..llm import get_llm_client
from ..prompt_templates import ANSWER_PROMPT


def generate_answer(config, client, query, context_data):
    prompt = ANSWER_PROMPT.format(
        context_data=context_data,
        query=query,
    )

    res = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
    )

    return res.choices[0].message.content


def run(config, query, raw_context, compressed_context):
    """Generate answer from raw context + compressed reasoning notes."""
    client = get_llm_client(config)
    answer_context = f"{raw_context}\n\nCompressed Reasoning Notes:\n{compressed_context}"
    return generate_answer(config, client, query, answer_context)
