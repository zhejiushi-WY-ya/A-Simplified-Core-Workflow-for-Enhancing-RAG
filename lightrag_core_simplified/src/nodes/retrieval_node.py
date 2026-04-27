from ..modules import retrieval_module
from ..wtb_integration.tracing import build_state_patch, trace_node


def build_node(config):
    def node(state):
        with trace_node("retrieval"):
            result = retrieval_module.run(
                config,
                state["query"],
                mode=state.get("mode"),
            )
            state_patch = build_state_patch()
        return {**state, "result": result, **state_patch}

    return node
