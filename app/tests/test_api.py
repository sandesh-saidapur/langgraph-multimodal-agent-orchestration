"""
API tests. Run from the app/ folder:  pytest -q
"""

import io
import os
import tempfile

# Point the app at a throwaway database and upload folder before it is imported.
_TMP = tempfile.mkdtemp()
os.environ["CHECKPOINT_DB"] = os.path.join(_TMP, "test_checkpoints.db")
os.environ["UPLOAD_DIR"] = os.path.join(_TMP, "uploads")

from fastapi.testclient import TestClient  # noqa: E402
from api import app  # noqa: E402

client = TestClient(app)


def _warning_photo():
    # The stand-in vision model reads the file name, so the bytes can be anything.
    return {"image": ("dashboard_warning.jpg", io.BytesIO(b"fake image bytes"), "image/jpeg")}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_math_question_completes_immediately():
    response = client.post("/ask", data={"question": "what is 12 * (3 + 4)"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "complete"
    assert "= 84" in body["answer"]


def test_injection_is_refused():
    response = client.post("/ask", data={"question": "what is __import__('os').system('ls') + 1"})
    assert "could not calculate" in response.json()["answer"]


def test_image_answer_waits_then_is_approved():
    first = client.post("/ask", data={"question": "what does this warning mean?"},
                        files=_warning_photo()).json()
    assert first["status"] == "awaiting_review"
    assert "warning light" in first["draft_for_review"]

    second = client.post(f"/review/{first['thread_id']}", json={"decision": "approve"}).json()
    assert second["status"] == "complete"
    assert "OBD-II" in second["answer"]


def test_image_answer_can_be_rejected():
    first = client.post("/ask", data={"question": "what does this warning mean?"},
                        files=_warning_photo()).json()
    second = client.post(f"/review/{first['thread_id']}", json={"decision": "reject"}).json()
    assert "Escalated" in second["answer"]


def test_review_unknown_thread_returns_404():
    response = client.post("/review/does-not-exist", json={"decision": "approve"})
    assert response.status_code == 404


def test_empty_question_is_rejected():
    response = client.post("/ask", data={"question": ""})
    assert response.status_code == 422
