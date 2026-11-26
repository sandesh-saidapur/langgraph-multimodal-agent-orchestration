"""
Phase 2: multimodal input (text + image) and a model-based router.

Flow:
    START -> (image given?) -> describe_image -> router -> (calculator | lookup) -> answer -> END
                  |                                 ^
                  +-------- no image ---------------+

Models are behind small classes so they can be swapped:
  - MockVisionModel / MockRouterModel run offline (no API key needed).
  - Replace them with real API calls when you have a key (see models.py).
"""

import os
import re
from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, START, END

from models import MockVisionModel, MockRouterModel


# ---------------------------------------------------------------------------
# STATE: everything a node may read or write.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    image_path: Optional[str]          # NEW: path to an image the user attached
    image_description: Optional[str]   # NEW: what the vision model saw
    route: Optional[str]
    tool_result: Optional[str]
    answer: Optional[str]


# Models are created once and shared by the nodes.
vision_model = MockVisionModel()
router_model = MockRouterModel()


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def describe_image(state: AgentState) -> dict:
    """Turn the attached image into text the rest of the graph can use."""
    description = vision_model.describe(state["image_path"])
    return {"image_description": description}


def router(state: AgentState) -> dict:
    """Decide the tool using the question AND any image description."""
    route = router_model.choose(
        question=state["question"],
        image_description=state.get("image_description"),
    )
    return {"route": route}


def calculator(state: AgentState) -> dict:
    expression = re.sub(r"[^\d\+\-\*\/\(\)\.\s]", "", state["question"].lower())
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # teaching shortcut, see Phase 4
        return {"tool_result": f"{expression.strip()} = {result}"}
    except Exception as exc:
        return {"tool_result": f"Could not calculate: {exc}"}


KNOWLEDGE_BASE = {
    "porsche": "Porsche AG is a German sports car manufacturer headquartered in Stuttgart.",
    "bmw": "BMW is a German manufacturer of cars and motorcycles headquartered in Munich.",
    "langgraph": "LangGraph is a framework for building stateful, long-running agents.",
    "audi": "Audi is a German car manufacturer headquartered in Ingolstadt.",
    # Automotive facts used by the image + text demo:
    "engine warning": "An orange engine warning light usually means the engine management "
                      "system has logged a fault. Read the fault code with an OBD-II scanner "
                      "and have it checked by a workshop.",
}


def lookup(state: AgentState) -> dict:
    # Search the question AND the image description, so a photo can trigger a lookup.
    haystack = f"{state['question']} {state.get('image_description') or ''}".lower()
    for key, fact in KNOWLEDGE_BASE.items():
        if key in haystack:
            return {"tool_result": fact}
    return {"tool_result": "No matching entry found."}


def answer(state: AgentState) -> dict:
    parts = []
    if state.get("image_description"):
        parts.append(f"Image: {state['image_description']}")
    parts.append(f"[{state['route']}] {state['tool_result']}")
    return {"answer": "\n".join(parts)}


# ---------------------------------------------------------------------------
# EDGES: routing functions that pick the next node.
# ---------------------------------------------------------------------------
def start_route(state: AgentState) -> Literal["describe_image", "router"]:
    return "describe_image" if state.get("image_path") else "router"


def pick_tool(state: AgentState) -> Literal["calculator", "lookup"]:
    return state["route"]


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("describe_image", describe_image)
    builder.add_node("router", router)
    builder.add_node("calculator", calculator)
    builder.add_node("lookup", lookup)
    builder.add_node("answer", answer)

    builder.add_conditional_edges(START, start_route)   # image? -> vision first
    builder.add_edge("describe_image", "router")         # vision output feeds the router
    builder.add_conditional_edges("router", pick_tool)
    builder.add_edge("calculator", "answer")
    builder.add_edge("lookup", "answer")
    builder.add_edge("answer", END)

    return builder.compile()


def run(question: str, image_path: Optional[str] = None) -> str:
    graph = build_graph()
    result = graph.invoke({
        "question": question,
        "image_path": image_path,
        "image_description": None,
        "route": None,
        "tool_result": None,
        "answer": None,
    })
    return result["answer"]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    warning_photo = os.path.join(here, "sample_images", "dashboard_warning.jpg")

    cases = [
        ("what is 12 * (3 + 4)", None),
        ("tell me about Porsche", None),
        ("what does this warning mean?", warning_photo),
    ]
    for question, image in cases:
        print(f"Q: {question}")
        if image:
            print(f"Image: {os.path.basename(image)}")
        print(f"A:\n{run(question, image)}\n")
