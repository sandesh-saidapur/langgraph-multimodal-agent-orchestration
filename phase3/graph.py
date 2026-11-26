"""
Phase 3: a supervisor that coordinates specialists in parallel.

Main graph:

    START -> supervisor -+-> image_branch (SUBGRAPH) --+-> synthesize -> END
                         +-> calculator ---------------+
                         +-> retrieval ----------------+

- supervisor: reads the input and writes a plan (which specialists are needed).
- image_branch: a subgraph with its own steps (describe the image, then look
  up facts about what it shows). It is a complete graph nested inside this one.
- calculator and retrieval: run in parallel with the image branch.
- synthesize: waits for all three, then combines their notes into one answer.

Design note: all three specialists always run. Each one checks the plan and
does nothing if it is not needed. This keeps the join simple: synthesize waits
for every branch to finish, so it never waits forever on a skipped branch.
"""

import operator
import re
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from models import MockVisionModel


# ---------------------------------------------------------------------------
# Knowledge base (same idea as earlier phases)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "porsche": "Porsche AG is a German sports car manufacturer headquartered in Stuttgart.",
    "bmw": "BMW is a German manufacturer of cars and motorcycles headquartered in Munich.",
    "audi": "Audi is a German car manufacturer headquartered in Ingolstadt.",
    "langgraph": "LangGraph is a framework for building stateful, long-running agents.",
    "engine warning": "An orange engine warning light usually means the engine management "
                      "system has logged a fault. Read the fault code with an OBD-II scanner "
                      "and have it checked by a workshop.",
}

vision_model = MockVisionModel()


def search_kb(text: str) -> Optional[str]:
    text = text.lower()
    for key, fact in KNOWLEDGE_BASE.items():
        if key in text:
            return fact
    return None


# ---------------------------------------------------------------------------
# MAIN STATE
# `notes` uses a reducer (operator.add): parallel branches each append,
# and the appends are combined instead of overwriting each other.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    image_path: Optional[str]
    plan: list[str]
    notes: Annotated[list[str], operator.add]
    answer: Optional[str]


# ---------------------------------------------------------------------------
# SUBGRAPH: the image branch has its own small state and steps.
# ---------------------------------------------------------------------------
class ImageState(TypedDict):
    question: str
    image_path: Optional[str]
    plan: list[str]
    image_description: Optional[str]
    notes: Annotated[list[str], operator.add]


def describe(state: ImageState) -> dict:
    description = vision_model.describe(state["image_path"])
    return {"image_description": description}


def image_lookup(state: ImageState) -> dict:
    fact = search_kb(state["image_description"] or "")
    notes = [f"[image] {state['image_description']}"]
    if fact:
        notes.append(f"[image facts] {fact}")
    return {"notes": notes}


def image_entry(state: ImageState) -> str:
    # Only run the image steps when the supervisor planned them.
    return "describe" if "image" in state["plan"] else END


def build_image_subgraph():
    b = StateGraph(ImageState)
    b.add_node("describe", describe)
    b.add_node("image_lookup", image_lookup)
    b.add_conditional_edges(START, image_entry)
    b.add_edge("describe", "image_lookup")
    b.add_edge("image_lookup", END)
    return b.compile()


# ---------------------------------------------------------------------------
# MAIN NODES
# ---------------------------------------------------------------------------
def supervisor(state: AgentState) -> dict:
    """Write the plan: which specialists the input needs."""
    plan = []
    if state.get("image_path"):
        plan.append("image")
    if re.search(r"\d", state["question"]) and re.search(r"[\+\-\*\/]|calculate|what is", state["question"].lower()):
        plan.append("calculator")
    else:
        plan.append("retrieval")
    return {"plan": plan}


def calculator(state: AgentState) -> dict:
    if "calculator" not in state["plan"]:
        return {}
    expression = re.sub(r"[^\d\+\-\*\/\(\)\.\s]", "", state["question"].lower())
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # teaching shortcut, see Phase 4
        return {"notes": [f"[calculator] {expression.strip()} = {result}"]}
    except Exception as exc:
        return {"notes": [f"[calculator] could not calculate: {exc}"]}


def retrieval(state: AgentState) -> dict:
    if "retrieval" not in state["plan"]:
        return {}
    fact = search_kb(state["question"])
    return {"notes": [f"[retrieval] {fact}"]} if fact else {}


def synthesize(state: AgentState) -> dict:
    """Combine every specialist's note into the final answer."""
    if not state["notes"]:
        return {"answer": "I could not find an answer to that."}
    return {"answer": "\n".join(state["notes"])}


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("image_branch", build_image_subgraph())  # subgraph as a node
    builder.add_node("calculator", calculator)
    builder.add_node("retrieval", retrieval)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "supervisor")
    # Fan out: all three specialists start in the same step, in parallel.
    builder.add_conditional_edges(
        "supervisor",
        lambda state: ["image_branch", "calculator", "retrieval"],
    )
    # Join: synthesize runs only after all three branches have finished.
    builder.add_edge(["image_branch", "calculator", "retrieval"], "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


def run(question: str, image_path: Optional[str] = None) -> dict:
    graph = build_graph()
    return graph.invoke({
        "question": question,
        "image_path": image_path,
        "plan": [],
        "notes": [],
        "answer": None,
    })


if __name__ == "__main__":
    cases = [
        ("what is 12 * (3 + 4)", None),
        ("tell me about Porsche", None),
        ("what does this warning mean?", "sample_images/dashboard_warning.jpg"),
    ]
    for question, image in cases:
        result = run(question, image)
        print(f"Q: {question}")
        print(f"Plan: {result['plan']}")
        print(f"A:\n{result['answer']}\n")
