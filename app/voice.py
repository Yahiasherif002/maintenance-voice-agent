import re
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import STT_MODEL, TTS_MODEL, TTS_VOICE

SAMPLE_RATE = 24_000
MIN_SPOKEN_CHARS = 25
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])|\n+")
CITATION = re.compile(r"\s*\((?:[\w.-]+\.pdf,\s*p\.\s*\d+(?:;\s*)?)+\)")
MARKDOWN = re.compile(r"[*_#`>]+")
CONFIRM = re.compile(r"^\W*(yes|yeah|yep|confirm(ed)?|approve(d)?|go ahead|do it|create it)\b", re.IGNORECASE)
VOCABULARY = "HX-200 pump, TR-500 transformer, AC-75 compressor, tags like P-101, T-201, C-301, work order"
VOICE_STYLE = "Calm, clear, and steady, like an experienced maintenance engineer talking on a noisy plant floor."
AUDIO_TYPES = {b"\x1aE\xdf\xa3": "speech.webm", b"OggS": "speech.ogg", b"RIFF": "speech.wav"}


def audio_filename(audio: bytes) -> str:
    if audio[4:8] == b"ftyp":
        return "speech.mp4"
    return AUDIO_TYPES.get(audio[:4], "speech.webm")


def pop_sentences(buffer: str) -> tuple[list[str], str]:
    parts = SENTENCE_END.split(buffer)
    done, current = [], ""
    for part in parts[:-1]:
        current = f"{current} {part}".strip()
        if len(current) >= MIN_SPOKEN_CHARS:
            done.append(current)
            current = ""
    return done, f"{current} {parts[-1]}".strip() if current else parts[-1]


def for_speech(text: str) -> str:
    return MARKDOWN.sub("", CITATION.sub("", text)).strip()


def is_confirmation(transcript: str) -> bool:
    return bool(CONFIRM.match(transcript))


def approval_prompt(request: dict) -> str:
    return (f"Shall I create a {request['priority']} priority work order for {request['equipment_tag']}: "
            f"{request['summary'].rstrip('.')}? Say confirm to create it, or cancel.")


async def transcribe(client: AsyncOpenAI, audio: bytes) -> AsyncIterator[str]:
    stream = await client.audio.transcriptions.create(model=STT_MODEL, file=(audio_filename(audio), audio),
                                                      prompt=VOCABULARY, stream=True)
    async for event in stream:
        if event.type == "transcript.text.delta":
            yield event.delta


async def speak(client: AsyncOpenAI, text: str) -> AsyncIterator[bytes]:
    async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=TTS_VOICE, input=text, instructions=VOICE_STYLE, response_format="pcm") as response:
        async for chunk in response.iter_bytes(4800):
            yield chunk
