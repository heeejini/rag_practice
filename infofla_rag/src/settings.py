# src/settings.py
import os
from pathlib import Path

from src.config import ChunkConfig, EmbedConfig, LLMConfig, QdrantConfig

# ==== 공통 상수 ====
MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "2000"))

INDEX_SRC_DIR_DEFAULT: str = os.getenv(
    "INDEX_SRC_DIR",
    "/app/data/news_articles_preprocessing",
)

UPLOAD_DIR: str = os.getenv(
    "UPLOAD_DIR",
    "/app/data/uploads",
)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ==== RAG 관련 Config 인스턴스 ====

chunk_cfg = ChunkConfig(
    chunk_size=1000,      
    overlap=500,
    strip_brackets=True,
)

qdrant_cfg = QdrantConfig(
    host=os.getenv("QDRANT_HOST", "qdrant"),     #
    port=int(os.getenv("QDRANT_PORT", "6333")),
    collection=os.getenv("QDRANT_COLLECTION", "news_chunks"),
    recreate=False,  # API 서버에서는 기본적으로 컬렉션 삭제 X
)

embed_cfg = EmbedConfig(
    model_name=os.getenv(
        "EMBED_MODEL_NAME",
        "Alibaba-NLP/gte-multilingual-base",
    ),
    batch_size=int(os.getenv("EMBED_BATCH_SIZE", "256")),
    trust_remote_code=True,
)

llm_cfg = LLMConfig(
    model_id=os.getenv(
        "LLM_MODEL_ID",
        "K-intelligence/Midm-2.0-Base-Instruct",
    ),
    max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "256")),
    do_sample=False,
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    use_vllm=os.getenv("USE_VLLM", "true").lower() == "true",
    vllm_api_base=os.getenv("VLLM_API_BASE", "http://vllm:8000"),
)
