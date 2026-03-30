from ..modules import embedding_module


def build_node(config):
    async def node(state):
        embedding_module.run(config, state["chunks"], state["graph_delta"])
        return state

    return node
