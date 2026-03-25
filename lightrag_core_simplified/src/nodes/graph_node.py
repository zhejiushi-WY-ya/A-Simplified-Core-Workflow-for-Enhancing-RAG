from ..modules import graph


def build_node(config):
    async def node(state):
        g = graph.run(config, state["chunks"])
        return {**state, "graph": g}

    return node
