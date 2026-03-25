from ..modules import retrieval


def build_node(config):
    async def node(state):
        result = retrieval.run(config, state["query"])
        return {**state, "result": result}

    return node
