from pydantic import BaseModel

class EmbedConfig(BaseModel):
    model_name: str = "Alibaba-NLP/gte-multilingual-base"
    batch_size: int = 256
    trust_remote_code: bool = True

class QdrantConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "news_chunks"
    recreate: bool = True  # 필요 시 새로 생성

class LLMConfig(BaseModel):
    model_id: str = "K-intelligence/Midm-2.0-Base-Instruct"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    use_vllm: bool = False
    vllm_api_base: str = "http://localhost:8001"

class ChunkConfig(BaseModel):
    chunk_size: int = 2000
    overlap: int = 500
    strip_brackets: bool = True
