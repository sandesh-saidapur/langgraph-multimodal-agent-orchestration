"""
Phase 5: the agent as a web API.

Endpoints:
  GET  /health                 -> is the service up?
  POST /ask                    -> ask a question, optionally with an image
  POST /review/{thread_id}     -> approve or reject a paused answer

Run locally:
  uvicorn api:app --reload        (from the app/ folder)
Then open http://127.0.0.1:8000/docs for the interactive page.
"""

import os
import shutil
import sqlite3
import uuid
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from graph import build_graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
DB_PATH = os.environ.get("CHECKPOINT_DB", os.path.join(BASE_DIR, "checkpoints.db"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
graph = build_graph(checkpointer=SqliteSaver(conn))

app = FastAPI(
    title="LangGraph multimodal agent",
    version="0.5.0",
    description="A supervisor-driven agent that answers questions about text and images, "
                "with a human approval step for image-based answers.",
)


class Decision(BaseModel):
    decision: Literal["approve", "reject"]


class AskResponse(BaseModel):
    thread_id: str
    status: Literal["complete", "awaiting_review"]
    answer: Optional[str] = None
    draft_for_review: Optional[str] = None


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(question: str = Form(..., min_length=1, max_length=500),
        image: Optional[UploadFile] = File(None)) -> AskResponse:
    thread_id = str(uuid.uuid4())

    image_path = None
    if image is not None:
        # Keep only the file name, never a path the client chose.
        safe_name = os.path.basename(image.filename or "upload.jpg")
        image_path = os.path.join(UPLOAD_DIR, f"{thread_id}_{safe_name}")
        with open(image_path, "wb") as out:
            shutil.copyfileobj(image.file, out)

    state = {
        "question": question,
        "image_path": image_path,
        "plan": [], "notes": [], "draft": None,
        "needs_review": False, "approved": None, "final": None,
    }
    result = graph.invoke(state, _config(thread_id))

    if "__interrupt__" in result:
        draft = result["__interrupt__"][0].value["draft"]
        return AskResponse(thread_id=thread_id, status="awaiting_review",
                           draft_for_review=draft)
    return AskResponse(thread_id=thread_id, status="complete", answer=result["final"])


@app.post("/review/{thread_id}", response_model=AskResponse)
def review(thread_id: str, body: Decision) -> AskResponse:
    snapshot = graph.get_state(_config(thread_id))
    if not snapshot.next:
        raise HTTPException(status_code=404,
                            detail="No paused answer found for this thread_id.")
    result = graph.invoke(Command(resume=body.decision), _config(thread_id))
    return AskResponse(thread_id=thread_id, status="complete", answer=result["final"])
