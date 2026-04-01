from openai import OpenAI

def get_llm_client(config):
    return OpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
    )


def get_embedding_client(config):
    return OpenAI(
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
    )
