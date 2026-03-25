import asyncio
from langgraph.graph import StateGraph, END

from .config import Config
from .nodes.retrieval_node import build_node


def build_graph(config):
    g = StateGraph(dict)

    g.add_node("retrieval", build_node(config))

    g.set_entry_point("retrieval")
    g.add_edge("retrieval", END)

    return g.compile()


async def run():
    config = Config(
        base_url="https://rtekkxiz.bja.sealos.run/v1",
        api_key="sk-eGYT382xngt2u4kGGnxInmjYvqloG8ltr07UbSKvo7w2uBI7"
    )

    graph = build_graph(config)

    result = await graph.ainvoke({
        "query": "What did Elara find under the loose floorboard in the workshop?"
    })

    '''
    She found a small, iridescent shell called a resonant shell.
    '''

    print(result["result"])

    '''
    "answer": "Elara found the **Resonant Shell** under the loose floorboard in the workshop. This shell is significant as it contains music and the memories of the First People, which deeply connects Elara to her emotions and influences her journey in transforming the town of Dunsmuir."
    '''


if __name__ == "__main__":
    asyncio.run(run())
