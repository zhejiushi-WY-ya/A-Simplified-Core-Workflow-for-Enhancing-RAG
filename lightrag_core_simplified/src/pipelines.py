from __future__ import annotations


def build_index_state_graph(config):
    from langgraph.graph import END, StateGraph

    from .nodes.chunk_node import build_node as chunk
    from .nodes.embedding_node import build_node as embed
    from .nodes.graph_node import build_node as graph

    graph_builder = StateGraph(dict)
    graph_builder.add_node("chunk", chunk(config))
    graph_builder.add_node("graph", graph(config))
    graph_builder.add_node("embedding", embed(config))
    graph_builder.set_entry_point("chunk")
    graph_builder.add_edge("chunk", "graph")
    graph_builder.add_edge("graph", "embedding")
    graph_builder.add_edge("embedding", END)
    return graph_builder


def build_index_graph(config):
    return build_index_state_graph(config).compile()


def build_query_state_graph(config):
    from langgraph.graph import END, StateGraph

    from .nodes.retrieval_node import build_node as retrieval

    graph_builder = StateGraph(dict)
    graph_builder.add_node("retrieval", retrieval(config))
    graph_builder.set_entry_point("retrieval")
    graph_builder.add_edge("retrieval", END)
    return graph_builder


def build_query_graph(config):
    return build_query_state_graph(config).compile()
