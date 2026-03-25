from ..modules import chunking


def build_node(config):
    async def node(state):
        chunks = chunking.run(
            config,
            state["content"],
            state["doc_id"]
        )
        return {**state, "chunks": chunks}

    return node
