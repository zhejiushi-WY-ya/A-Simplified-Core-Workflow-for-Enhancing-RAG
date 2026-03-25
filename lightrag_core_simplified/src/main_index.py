import asyncio
from langgraph.graph import StateGraph, END

from .config import Config
from .nodes.chunk_node import build_node as chunk
from .nodes.graph_node import build_node as graph
from .nodes.embedding_node import build_node as embed


def build_graph(config):
    g = StateGraph(dict)

    g.add_node("chunk", chunk(config))
    g.add_node("graph", graph(config))
    g.add_node("embedding", embed(config))

    g.set_entry_point("chunk")

    g.add_edge("chunk", "graph")
    g.add_edge("graph", "embedding")
    g.add_edge("embedding", END)

    return g.compile()


async def run():
    config = Config(
        base_url="https://rtekkxiz.bja.sealos.run/v1",
        api_key="sk-eGYT382xngt2u4kGGnxInmjYvqloG8ltr07UbSKvo7w2uBI7"
    )

    graph = build_graph(config)

    with open('./input/book.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    await graph.ainvoke({
        "doc_id": "test_doc",
        "content": content
    })

    print('Done')


if __name__ == "__main__":
    asyncio.run(run())
