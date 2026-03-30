from ..modules import retrieval_module


def build_node(config):
    async def node(state):
        result = retrieval_module.run(
            config,
            state["query"],
            mode=state.get("mode"),
        )
        return {**state, "result": result}

    return node
