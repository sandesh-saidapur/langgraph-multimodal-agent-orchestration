"""
Demo of a pause that survives a restart.

Step 1:  python3 run_demo.py start
         Runs the graph until it needs approval, saves the state to
         checkpoints.db, and exits.

Step 2:  python3 run_demo.py resume approve     (or: resume reject)
         A brand-new program run reloads the saved state and finishes.
"""

import sqlite3
import sys

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from graph import build_graph

DB_FILE = "checkpoints.db"
CONFIG = {"configurable": {"thread_id": "warning-review-1"}}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "start"
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(conn))

    if mode == "start":
        state = {
            "question": "what does this warning mean?",
            "image_path": "sample_images/dashboard_warning.jpg",
            "plan": [], "notes": [], "draft": None,
            "needs_review": False, "approved": None, "final": None,
        }
        result = graph.invoke(state, CONFIG)
        print("PAUSED. Waiting for a human decision.")
        print("Draft for review:")
        print(result["__interrupt__"][0].value["draft"])
        print("\nNext: python3 run_demo.py resume approve")

    elif mode == "resume":
        decision = sys.argv[2] if len(sys.argv) > 2 else "approve"
        result = graph.invoke(Command(resume=decision), CONFIG)
        print(f"Decision: {decision}")
        print("FINAL ANSWER:")
        print(result["final"])

    conn.close()


if __name__ == "__main__":
    main()
