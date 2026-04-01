import os
from functools import partial
from lightrag.llm.openai import(
    openai_complete_if_cache,
    gpt_4o_mini_complete,
    openai_embed,
)
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv
load_dotenv()


def select_model(type):
    if type == 'local':

        async def llm_model_func(
                prompt, system_prompt=None, history_messages=[], **kwargs
        ) -> str:
            return await openai_complete_if_cache(
                model=os.getenv("LLM_MODEL", "Qwen3-30B-A3B-Instruct-2507"),
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=os.getenv("LLM_BINDING_HOST", "http://0.0.0.0:18888/v1"),
                api_key=os.getenv("LLM_BINDING_API_KEY", "not_needed"),
                timeout=600,
                **kwargs,
            )

        vLLM_emb_func = EmbeddingFunc(
            model_name=os.getenv("EMBEDDING_MODEL", "Qwen3-embedding-0.6B"),
            send_dimensions=False,
            embedding_dim=int(os.getenv("EMBEDDING_DIM", 1024)),
            max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", 4096)),
            func=partial(
                openai_embed.func,
                model=os.getenv("EMBEDDING_MODEL", "Qwen3-embedding-0.6B"),
                base_url=os.getenv(
                    "EMBEDDING_BINDING_HOST",
                    "http://0.0.0.0:18889/v1",
                ),
                api_key=os.getenv("EMBEDDING_BINDING_API_KEY", "not_needed"),
            ),
        )

        return vLLM_emb_func,llm_model_func

    elif type == 'openai':
        return openai_embed,gpt_4o_mini_complete
