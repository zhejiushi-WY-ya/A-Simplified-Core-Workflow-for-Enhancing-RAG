from ..modules import graph_module


def build_node(config):
    async def node(state):
        g = graph_module.run(config, state["chunks"])
        return {
            **state,
            "graph": g["full"],
            "graph_delta": g["delta"],
        }

    return node
