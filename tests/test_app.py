import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from starlette.websockets import WebSocketDisconnect

from app import main, voice
from app.agent import FALLBACK
from app.config import DATABASE_URL, EMBED_DIM
from app.retrieval import embeddings, ingest_folder

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
SEAL = {"equipment_tag": "P-101", "priority": "high", "summary": "Mechanical seal is leaking at the drive end"}


class HashEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * EMBED_DIM
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBED_DIM] += 1.0
        return vector


class ScriptedModel(BaseChatModel):
    replies: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def tool_call(name, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": uuid.uuid4().hex, "type": "tool_call"}])


def work_orders(thread):
    with psycopg.connect(DATABASE_URL) as conn:
        return conn.execute("SELECT equipment_tag, priority, summary FROM work_orders WHERE thread_id = %s",
                            (thread,)).fetchall()


def receive_turn(ws):
    events, audio = [], b""
    while True:
        message = ws.receive()
        if message.get("bytes"):
            audio += message["bytes"]
            continue
        events.append(json.loads(message["text"]))
        if events[-1]["type"] in ("turn.done", "error"):
            return events, audio


def text_of(events, kind):
    return "".join(e["text"] for e in events if e["type"] == kind)


@pytest.fixture
def app(monkeypatch):
    model, spoken, heard = ScriptedModel(), [], []

    async def transcribe(client, audio):
        for word in heard.pop(0).split(" "):
            yield word + " "

    async def speak(client, text):
        spoken.append(text)
        yield b"\x00\x00" * 480

    monkeypatch.setattr(main, "chat_model", lambda: model)
    monkeypatch.setattr(main, "embeddings", HashEmbeddings)
    monkeypatch.setattr(main, "audio_client", lambda: None)
    monkeypatch.setattr(voice, "transcribe", transcribe)
    monkeypatch.setattr(voice, "speak", speak)
    asyncio.run(ingest_folder(ROOT / "docs", HashEmbeddings()))
    with TestClient(main.app) as client:
        yield SimpleNamespace(client=client, model=model, spoken=spoken, heard=heard, thread=uuid.uuid4().hex)


def ask(app, text):
    return app.client.post(f"/threads/{app.thread}/messages", json={"text": text})


def test_sentences_keep_citations_together():
    done, rest = voice.pop_sentences("Check the seal. Torque the nut to spec (hx200_pump_manual.pdf, p. 4). Then")
    assert done == ["Check the seal. Torque the nut to spec (hx200_pump_manual.pdf, p. 4)."]
    assert rest == "Then"
    assert voice.for_speech(done[0]) == "Check the seal. Torque the nut to spec."
    assert voice.for_speech("**Stop** the pump (a.pdf, p. 1; b.pdf, p. 2).") == "Stop the pump."


def test_voice_helpers():
    assert all(map(voice.is_confirmation, ["Yes, go ahead.", "confirm", "Approved"]))
    assert not any(map(voice.is_confirmation, ["No, cancel it", "I'm not sure, yes", ""]))
    assert voice.audio_filename(b"\x1aE\xdf\xa3....") == "speech.webm"
    assert voice.audio_filename(b"\x00\x00\x00\x20ftypM4A ") == "speech.mp4"
    assert voice.approval_prompt(SEAL).startswith("Shall I create a high priority work order for P-101")


def test_hybrid_search_finds_pages(app):
    top = app.client.get("/search", params={"q": "What torque do I use for the HX-200 impeller nut?"}).json()[0]
    assert (top["source"], top["page"]) == ("hx200_pump_manual.pdf", 4)
    hits = [any((h["source"], h["page"]) == (q["source"], q["page"])
                for h in app.client.get("/search", params={"q": q["question"]}).json()) for q in QUESTIONS]
    assert mean(hits) >= 0.9


def test_agent_answers_with_citations(app):
    app.model.replies += [tool_call("search_manuals", query="HX-200 impeller nut torque"),
                          AIMessage("Use the torque in the table (hx200_pump_manual.pdf, p. 4).")]
    body = ask(app, "What torque for the HX-200 impeller nut?").json()
    assert body["answer"] == "Use the torque in the table (hx200_pump_manual.pdf, p. 4)."
    assert {"source": "hx200_pump_manual.pdf", "page": 4} in body["sources"]
    excerpts = app.model.calls[1][-1]
    assert isinstance(excerpts, ToolMessage) and "[hx200_pump_manual.pdf, p. 4]" in excerpts.content


def test_conversation_memory(app):
    app.model.replies += [AIMessage("P-101 is a Cairo pump."), AIMessage("It is running.")]
    ask(app, "Tell me about P-101")
    assert ask(app, "Is it running?").json()["answer"] == "It is running."
    second = app.model.calls[1]
    assert [m.content for m in second if isinstance(m, HumanMessage)] == ["Tell me about P-101", "Is it running?"]


def test_sql_tool_is_read_only(app):
    expected = {
        "SELECT count(*) AS n FROM equipment": '[{"n": 13}]',
        "UPDATE equipment SET status = 'running'": "read-only",
        "INSERT INTO work_orders (equipment_tag, priority, summary, thread_id) VALUES ('P-101', 'low', 'xxxxx', 't')":
            "read-only",
        "SELECT * FROM chunks": "permission denied",
        "SELECT pg_read_file('/etc/passwd')": "permission denied",
        "SET statement_timeout = 0; SELECT pg_sleep(10)": "multiple commands",
        "SELECT pg_sleep(10)": "statement timeout",
    }
    calls = [{"name": "query_equipment", "args": {"sql": sql}, "id": f"q{i}", "type": "tool_call"}
             for i, sql in enumerate(expected)]
    app.model.replies += [AIMessage("", tool_calls=calls), AIMessage("Done.")]
    body = ask(app, "Run these").json()
    assert body["queries"] == list(expected)
    results = {m.tool_call_id: m.content for m in app.model.calls[1] if isinstance(m, ToolMessage)}
    for i, needle in enumerate(expected.values()):
        assert needle in results[f"q{i}"]


def test_work_order_waits_for_approval(app):
    app.model.replies.append(tool_call("create_work_order", **SEAL))
    body = ask(app, "Log a high priority job, P-101 seal is leaking").json()
    assert body["pending_approval"] == SEAL and body["answer"] is None
    assert work_orders(app.thread) == []
    assert ask(app, "Anything else?").status_code == 409

    app.model.replies.append(AIMessage("Work order created."))
    body = app.client.post(f"/threads/{app.thread}/approval", json={"approved": True}).json()
    assert body["answer"] == "Work order created." and body["pending_approval"] is None
    assert work_orders(app.thread) == [tuple(SEAL.values())]
    assert "Created work order WO-" in app.model.calls[-1][-1].content
    assert app.client.post(f"/threads/{app.thread}/approval", json={"approved": True}).status_code == 409


def test_cancelled_or_unknown_work_order_writes_nothing(app):
    app.model.replies += [tool_call("create_work_order", **SEAL), AIMessage("Cancelled.")]
    ask(app, "Log a job for P-101")
    app.client.post(f"/threads/{app.thread}/approval", json={"approved": False})
    assert "cancelled" in app.model.calls[-1][-1].content

    app.model.replies += [tool_call("create_work_order", **{**SEAL, "equipment_tag": "X-999"}),
                          AIMessage("X-999 does not exist.")]
    body = ask(app, "Log a job for X-999").json()
    assert body["pending_approval"] is None
    assert "no equipment with tag X-999" in app.model.calls[-1][-1].content
    assert work_orders(app.thread) == []


def test_step_limit_forces_an_answer(app):
    app.model.replies += [tool_call("search_manuals", query=f"pump {i}") for i in range(10)]
    assert ask(app, "Loop forever").json()["answer"] == FALLBACK


def test_voice_turn_streams_transcript_text_and_audio(app):
    app.heard.append("What torque for the HX-200 impeller nut?")
    app.model.replies += [tool_call("search_manuals", query="HX-200 impeller nut torque"),
                          AIMessage("Use the torque in the table (hx200_pump_manual.pdf, p. 4). Tighten in two steps.")]
    with app.client.websocket_connect(f"/threads/{app.thread}/voice") as ws:
        ws.send_bytes(b"\x1aE\xdf\xa3 recording")
        events, audio = receive_turn(ws)
    assert text_of(events, "transcript.delta").strip() == "What torque for the HX-200 impeller nut?"
    assert text_of(events, "answer.delta") == events[-1]["answer"]
    assert events[-1]["answer"].endswith("Tighten in two steps.")
    assert app.spoken == ["Use the torque in the table.", "Tighten in two steps."]
    assert len(audio) == 2 * 960
    assert {"source": "hx200_pump_manual.pdf", "page": 4} in events[-1]["sources"]


def test_voice_confirmation_creates_work_order(app):
    app.heard += ["Log a work order, P-101 seal is leaking", "Confirm."]
    app.model.replies += [tool_call("create_work_order", **SEAL), AIMessage("Work order created.")]
    with app.client.websocket_connect(f"/threads/{app.thread}/voice") as ws:
        ws.send_bytes(b"\x1aE\xdf\xa3 first")
        events, _ = receive_turn(ws)
        assert [e["request"] for e in events if e["type"] == "approval"] == [SEAL]
        assert app.spoken[-1] == voice.approval_prompt(SEAL)
        assert work_orders(app.thread) == []
        ws.send_bytes(b"\x1aE\xdf\xa3 second")
        events, _ = receive_turn(ws)
    assert events[-1]["answer"] == "Work order created."
    assert work_orders(app.thread) == [tuple(SEAL.values())]


def test_rejects_bad_input(app):
    assert app.client.post("/threads/bad%20id/messages", json={"text": "hi"}).status_code == 422
    assert ask(app, "").status_code == 422
    assert app.client.get("/search", params={"q": ""}).status_code == 422
    with pytest.raises(WebSocketDisconnect), app.client.websocket_connect("/threads/bad%21/voice") as ws:
        ws.receive()
    with app.client.websocket_connect(f"/threads/{app.thread}/voice") as ws:
        ws.send_bytes(b"\x00" * (main.MAX_AUDIO_BYTES + 1))
        assert ws.receive_json() == {"type": "error", "message": "Recording is too long"}


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY for real models")
def test_eval_with_real_models():
    asyncio.run(ingest_folder(ROOT / "docs", embeddings()))
    with TestClient(main.app) as client:
        retrieved = [any((h["source"], h["page"]) == (q["source"], q["page"])
                         for h in client.get("/search", params={"q": q["question"]}).json()) for q in QUESTIONS]
        cited = [f"{q['source']}, p. {q['page']}" in (client.post(f"/threads/eval-{uuid.uuid4().hex}/messages",
                                                                  json={"text": q["question"]}).json()["answer"] or "")
                 for q in QUESTIONS]
    print(f"retrieval recall@5 = {mean(retrieved):.2f}, answer cites expected page = {mean(cited):.2f}")
    assert mean(retrieved) >= 0.9
    assert mean(cited) >= 0.8
