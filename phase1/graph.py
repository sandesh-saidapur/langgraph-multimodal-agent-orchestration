"""
Phase 1: a minimal LangGraph agent with a router and two tools.

Flow:
    START -> router -> (calculator | lookup) -> answer -> END

The router inspects the question and sends it to the right tool.
"""

import re
from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. STATE: the shared notebook passed between nodes.
#    Each key is a field any node can read or update.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str            # the user's input
    route: Optional[str]     # which tool the router picked
    tool_result: Optional[str]  # what the tool produced
    answer: Optional[str]    # the final reply


# ---------------------------------------------------------------------------
# 2. NODES: plain functions. They receive the state and return a dict
#    containing only the fields they want to change.
# ---------------------------------------------------------------------------
def router(state: AgentState) -> dict:
    """Decide which tool should handle the question."""
    question = state["question"].lower()
    # A question containing only digits, operators and spaces is maths.
    if re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", question.replace("what is", "").strip()):
        return {"route": "calculator"}
    if any(word in question for word in ["what is", "+", "*", "/", "calculate"]) and re.search(r"\d", question):
        return {"route": "calculator"}
    return {"route": "lookup"}


def calculator(state: AgentState) -> dict:
    """Evaluate a simple arithmetic expression."""
    expression = re.sub(r"[^\d\+\-\*\/\(\)\.\s]", "", state["question"].lower())
    try:
        # eval is acceptable only because we stripped everything except math characters.
        result = eval(expression, {"__builtins__": {}}, {})
        return {"tool_result": f"{expression.strip()} = {result}"}
    except Exception as exc:
        return {"tool_result": f"Could not calculate: {exc}"}


# A tiny stand-in for a knowledge base. Later phases will use real documents.
KNOWLEDGE_BASE = {
    "porsche": "Porsche AG is a German sports car manufacturer headquartered in Stuttgart.",
    "bmw": "BMW is a German manufacturer of cars and motorcycles headquartered in Munich.",
    "langgraph": "LangGraph is a framework for building stateful, long-running agents.",
}


def lookup(state: AgentState) -> dict:
    """Search the tiny knowledge base for a keyword match."""
    question = state["question"].lower()
    for key, fact in KNOWLEDGE_BASE.items():
        if key in question:
            return {"tool_result": fact}
    return {"tool_result": "No matching entry found."}


def answer(state: AgentState) -> dict:
    """Turn the tool output into the final reply."""
    return {"answer": f"[{state['route']}] {state['tool_result']}"}


# ---------------------------------------------------------------------------
# 3. EDGES: the routing function. It reads the state and returns the NAME
#    of the next node. The dictionary maps each returned name to a node.
# ---------------------------------------------------------------------------
def pick_tool(state: AgentState) -> Literal["calculator", "lookup"]:
    return state["route"]


# ---------------------------------------------------------------------------
# 4. BUILD the graph: add nodes, connect them, compile.
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router", router)
    builder.add_node("calculator", calculator)
    builder.add_node("lookup", lookup)
    builder.add_node("answer", answer)

    builder.add_edge(START, "router")                 # always start at the router
    builder.add_conditional_edges("router", pick_tool)  # router decides the branch
    builder.add_edge("calculator", "answer")          # both tools lead to answer
    builder.add_edge("lookup", "answer")
    builder.add_edge("answer", END)                   # finish

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    for question in [
        "what is 12 * (3 + 4)",
        "tell me about Porsche",
        "what is LangGraph",
        "calculate 100 / 4",
    ]:
        result = graph.invoke({"question": question, "route": None,
                               "tool_result": None, "answer": None})
        print(f"Q: {question}\nA: {result['answer']}\n")
