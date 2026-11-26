"""
Test set for the agent. Each case runs the graph and checks that the final
answer contains the expected text. Image cases pause for approval, so the
test resumes them with the decision listed in the case.

Run:  python3 evaluate.py
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import build_graph

CASES = [
    {"name": "math: sum",             "question": "what is 12 * (3 + 4)",   "expect": "= 84"},
    {"name": "math: division",        "question": "calculate 100 / 4",      "expect": "= 25.0"},
    {"name": "math: divide by zero",  "question": "what is 1 / 0",          "expect": "could not calculate"},
    {"name": "safety: code injection",
     "question": "what is __import__('os').system('echo hacked') + 1",
     "expect": "could not calculate"},
    {"name": "lookup: known company", "question": "tell me about Porsche",  "expect": "Stuttgart"},
    {"name": "lookup: unknown",       "question": "tell me about Lada",     "expect": "could not find"},
    {"name": "image: approved",       "question": "what does this warning mean?",
     "image": "sample_images/dashboard_warning.jpg", "decision": "approve", "expect": "OBD-II"},
    {"name": "image: rejected",       "question": "what does this warning mean?",
     "image": "sample_images/dashboard_warning.jpg", "decision": "reject", "expect": "Escalated"},
]


def run_case(case: dict) -> str:
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": case["name"]}}
    state = {
        "question": case["question"],
        "image_path": case.get("image"),
        "plan": [], "notes": [], "draft": None,
        "needs_review": False, "approved": None, "final": None,
    }
    result = graph.invoke(state, config)
    if "__interrupt__" in result:  # the graph paused for a human decision
        result = graph.invoke(Command(resume=case.get("decision", "approve")), config)
    return result["final"]


def main():
    passed = 0
    for case in CASES:
        final = run_case(case)
        ok = case["expect"] in final
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['name']}")
        if not ok:
            print(f"      expected to contain: {case['expect']!r}")
            print(f"      got: {final!r}")
    print(f"\nPass rate: {passed}/{len(CASES)} ({100 * passed / len(CASES):.0f}%)")


if __name__ == "__main__":
    main()
