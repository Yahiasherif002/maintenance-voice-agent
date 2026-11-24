import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
READER_DATABASE_URL = os.getenv("READER_DATABASE_URL",
                                "postgresql://agent_reader:agent_reader@localhost:5432/postgres")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536
STT_MODEL = "gpt-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "marin"
