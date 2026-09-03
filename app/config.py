import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", OLLAMA_HOST)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", CHAT_MODEL)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)
LLM_MAX_OUTPUT_TOKENS = _env_int("LLM_MAX_OUTPUT_TOKENS", 256)
LLM_REQUEST_TIMEOUT = _env_float("LLM_REQUEST_TIMEOUT", 120.0)
LLM_STREAMING_ENABLED = _env_bool("LLM_STREAMING_ENABLED", True)
LLM_REASONING_ENABLED = _env_bool("LLM_REASONING_ENABLED", False)

# Cloud fallback (disabled by default)
ALLOW_CLOUD_FALLBACK = _env_bool("ALLOW_CLOUD_FALLBACK", False)
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "")
CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "")

# Future company GPU / OpenAI-compatible endpoint
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "")
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "")
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY")

# PostgreSQL + pgvector
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://raguser:ragpassword@localhost:5433/rag_ai_chatbot",
)

# Knowledge base
DATA_DIR = "data"

# Ingestion
CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 200)
CSV_ROWS_PER_CHUNK = _env_int("CSV_ROWS_PER_CHUNK", 20)
OCR_ENABLED = _env_bool("OCR_ENABLED", True)
OCR_MIN_TEXT_CHARS = _env_int("OCR_MIN_TEXT_CHARS", 50)

# Retrieval (legacy aliases kept for compatibility)
TOP_K = _env_int("TOP_K", 5)
RETRIEVAL_MIN_SIMILARITY = _env_float("RETRIEVAL_MIN_SIMILARITY", 0.55)

# Hybrid retrieval
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", TOP_K)
KEYWORD_TOP_K = _env_int("KEYWORD_TOP_K", TOP_K)
FINAL_TOP_K = _env_int("FINAL_TOP_K", TOP_K)
VECTOR_WEIGHT = _env_float("VECTOR_WEIGHT", 1.0)
KEYWORD_WEIGHT = _env_float("KEYWORD_WEIGHT", 1.0)
RRF_K = _env_int("RRF_K", 60)
RERANKING_ENABLED = _env_bool("RERANKING_ENABLED", True)
HIGH_CONFIDENCE_THRESHOLD = _env_float("HIGH_CONFIDENCE_THRESHOLD", 0.72)
MEDIUM_CONFIDENCE_THRESHOLD = _env_float("MEDIUM_CONFIDENCE_THRESHOLD", 0.55)
MAX_CONTEXT_CHUNKS = _env_int("MAX_CONTEXT_CHUNKS", FINAL_TOP_K)
MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 6000)
NEAR_DUPLICATE_OVERLAP = _env_float("NEAR_DUPLICATE_OVERLAP", 0.85)

# Contact shown when the knowledge base cannot answer
SUPPORT_CONTACT = "PLEASE_CONFIGURE_SUPPORT_NUMBER"
