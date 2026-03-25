from ..modules import embedding


def build_node(config):
    async def node(state):
        embedding.run(config, state["chunks"], state["graph"])
        return state

    return node
