from ..modules import chunking_module


def build_node(config):
    async def node(state):
        chunks = chunking_module.run(
            config,
            state["content"],
            state["doc_id"]
        )
        return {**state, "chunks": chunks}

    return node
