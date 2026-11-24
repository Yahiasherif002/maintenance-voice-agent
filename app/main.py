import asyncio
import logging
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path as FilePath
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from openai import AsyncOpenAI
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from app import voice
from app.agent import build_graph, last_turn, pending_approval, thread_config
from app.config import CHAT_MODEL, DATABASE_URL, READER_DATABASE_URL
from app.retrieval import HybridRetriever, embeddings

logger = logging.getLogger(__name__)
THREAD_ID = r"^[A-Za-z0-9_-]{1,64}$"
MAX_AUDIO_BYTES = 5_000_000
INDEX = FilePath(__file__).resolve().parent.parent / "static" / "index.html"


def chat_model():
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def audio_client():
    return AsyncOpenAI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with (
        AsyncConnectionPool(DATABASE_URL, open=False, kwargs={
            "autocommit": True, "row_factory": dict_row, "prepare_threshold": 0}) as pool,
        AsyncConnectionPool(READER_DATABASE_URL, open=False, kwargs={"row_factory": dict_row}) as reader_pool,
    ):
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        app.state.retriever = HybridRetriever(pool=pool, embedder=embeddings())
        app.state.graph = build_graph(chat_model(), app.state.retriever, pool, reader_pool, checkpointer)
        app.state.audio = audio_client()
        yield


app = FastAPI(title="Maintenance Voice Agent", lifespan=lifespan)
ThreadId = Annotated[str, Path(pattern=THREAD_ID)]


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class Approval(BaseModel):
    approved: bool


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(INDEX)


@app.get("/search")
async def search(q: Annotated[str, Query(min_length=1, max_length=500)]):
    docs = await app.state.retriever.ainvoke(q)
    return [{**d.metadata, "content": d.page_content} for d in docs]


@app.post("/threads/{thread_id}/messages")
async def post_message(thread_id: ThreadId, body: Message):
    graph = app.state.graph
    if await pending_approval(graph, thread_id):
        raise HTTPException(409, "A work order is waiting for approval on this thread")
    await graph.ainvoke({"messages": [HumanMessage(body.text)]}, thread_config(thread_id))
    return await last_turn(graph, thread_id)


@app.post("/threads/{thread_id}/approval")
async def post_approval(thread_id: ThreadId, body: Approval):
    graph = app.state.graph
    if not await pending_approval(graph, thread_id):
        raise HTTPException(409, "Nothing is waiting for approval on this thread")
    await graph.ainvoke(Command(resume={"approved": body.approved}), thread_config(thread_id))
    return await last_turn(graph, thread_id)


async def speaker(ws: WebSocket, sentences: asyncio.Queue, muted: asyncio.Event):
    while (sentence := await sentences.get()) is not None:
        text = voice.for_speech(sentence)
        if muted.is_set() or not text:
            continue
        async for pcm in voice.speak(app.state.audio, text):
            if muted.is_set():
                break
            await ws.send_bytes(pcm)


async def voice_turn(ws: WebSocket, thread_id: str, audio: bytes, muted: asyncio.Event):
    graph = app.state.graph
    transcript = ""
    async for delta in voice.transcribe(app.state.audio, audio):
        transcript += delta
        await ws.send_json({"type": "transcript.delta", "text": delta})
    transcript = transcript.strip()
    if not transcript:
        await ws.send_json({"type": "turn.done", "answer": None, "sources": [], "queries": [],
                            "pending_approval": None})
        return
    if await pending_approval(graph, thread_id):
        run_input = Command(resume={"approved": voice.is_confirmation(transcript)})
    else:
        run_input = {"messages": [HumanMessage(transcript)]}

    sentences = asyncio.Queue()
    speaking = asyncio.create_task(speaker(ws, sentences, muted))
    try:
        buffer = ""
        async for chunk, meta in graph.astream(run_input, thread_config(thread_id), stream_mode="messages"):
            if meta.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessage) or not chunk.text:
                continue
            await ws.send_json({"type": "answer.delta", "text": chunk.text})
            done, buffer = voice.pop_sentences(buffer + chunk.text)
            for sentence in done:
                sentences.put_nowait(sentence)
        if buffer.strip():
            sentences.put_nowait(buffer)
        result = await last_turn(graph, thread_id)
        if result["pending_approval"]:
            prompt = voice.approval_prompt(result["pending_approval"])
            await ws.send_json({"type": "approval", "request": result["pending_approval"], "text": prompt})
            sentences.put_nowait(prompt)
        sentences.put_nowait(None)
        await speaking
        await ws.send_json({"type": "turn.done", **result})
    finally:
        speaking.cancel()


async def guarded_turn(ws: WebSocket, thread_id: str, audio: bytes, muted: asyncio.Event):
    try:
        await voice_turn(ws, thread_id, audio, muted)
    except Exception:
        logger.exception("voice turn failed on thread %s", thread_id)
        with suppress(Exception):
            await ws.send_json({"type": "error", "message": "That request failed. Please try again."})


@app.websocket("/threads/{thread_id}/voice")
async def voice_socket(ws: WebSocket, thread_id: str):
    if not re.fullmatch(THREAD_ID, thread_id):
        await ws.close(code=1008)
        return
    await ws.accept()
    turn, muted = None, asyncio.Event()
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            muted.set()
            audio = message.get("bytes")
            if not audio:
                continue
            if turn:
                await asyncio.gather(turn, return_exceptions=True)
            if len(audio) > MAX_AUDIO_BYTES:
                await ws.send_json({"type": "error", "message": "Recording is too long"})
                continue
            muted = asyncio.Event()
            turn = asyncio.create_task(guarded_turn(ws, thread_id, audio, muted))
    except WebSocketDisconnect:
        pass
    finally:
        if turn:
            await asyncio.gather(turn, return_exceptions=True)
