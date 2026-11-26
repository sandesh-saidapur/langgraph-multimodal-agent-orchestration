"""
Phase 4: reliability.

Main graph:

    START -> supervisor -> (image_branch | calculator | retrieval) -> synthesize
          -> review -> finalize -> END

- review: pauses the graph (interrupt) when an image-based answer needs a human
  to approve it. Needs a checkpointer to save the paused state.
- finalize: shares the draft only if it was approved (or no review was needed).
- The calculator now uses safe_eval instead of eval.
"""

import operator
import re
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from models import MockVisionModel
from safe_math import safe_eval


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


class AgentState(TypedDict):
    question: str
    image_path: Optional[str]
    plan: list[str]
    notes: Annotated[list[str], operator.add]
    draft: Optional[str]
    needs_review: bool
    approved: Optional[bool]
    final: Optional[str]


class ImageState(TypedDict):
    question: str
    image_path: Optional[str]
    plan: list[str]
    image_description: Optional[str]
    notes: Annotated[list[str], operator.add]


def describe(state: ImageState) -> dict:
    return {"image_description": vision_model.describe(state["image_path"])}


def image_lookup(state: ImageState) -> dict:
    fact = search_kb(state["image_description"] or "")
    notes = [f"[image] {state['image_description']}"]
    if fact:
        notes.append(f"[image facts] {fact}")
    return {"notes": notes}


def image_entry(state: ImageState) -> str:
    return "describe" if "image" in state["plan"] else END


def build_image_subgraph():
    b = StateGraph(ImageState)
    b.add_node("describe", describe)
    b.add_node("image_lookup", image_lookup)
    b.add_conditional_edges(START, image_entry)
    b.add_edge("describe", "image_lookup")
    b.add_edge("image_lookup", END)
    return b.compile()


def supervisor(state: AgentState) -> dict:
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
    expression = re.sub(r"^(what is|calculate)\s*", "", state["question"].strip().lower())
    try:
        return {"notes": [f"[calculator] {expression} = {safe_eval(expression)}"]}
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return {"notes": [f"[calculator] could not calculate: {exc}"]}


def retrieval(state: AgentState) -> dict:
    if "retrieval" not in state["plan"]:
        return {}
    fact = search_kb(state["question"])
    return {"notes": [f"[retrieval] {fact}"]} if fact else {}


def synthesize(state: AgentState) -> dict:
    if not state["notes"]:
        draft = "I could not find an answer to that."
    else:
        draft = "\n".join(state["notes"])
    return {"draft": draft, "needs_review": "image" in state["plan"]}


def review(state: AgentState) -> dict:
    """Pause here until a human decides. Resuming reruns this node from the top,
    so keep side effects out of it and put them in a later node."""
    if not state["needs_review"]:
        return {}
    decision = interrupt({"action": "approve_image_answer", "draft": state["draft"]})
    return {"approved": decision == "approve"}


def finalize(state: AgentState) -> dict:
    if state["needs_review"] and not state.get("approved"):
        return {"final": "Escalated: a human technician must check this before it is shared."}
    return {"final": state["draft"]}


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("image_branch", build_image_subgraph())
    builder.add_node("calculator", calculator)
    builder.add_node("retrieval", retrieval)
    builder.add_node("synthesize", synthesize)
    builder.add_node("review", review)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: ["image_branch", "calculator", "retrieval"],
    )
    builder.add_edge(["image_branch", "calculator", "retrieval"], "synthesize")
    builder.add_edge("synthesize", "review")
    builder.add_edge("review", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
