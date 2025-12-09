from pydantic import BaseModel
from typing import Optional

class Chunk(BaseModel):
    source_path: str
    source_name: str
    chunk_index: int
    text: str

class GenerationStats(BaseModel):
    llm_backend: str
    llm_latency: float
    max_new_tokens: int
    do_sample: bool

class RAGResult(BaseModel):
    answer: str
    context: Optional[str] = None
    stats: Optional[GenerationStats] = None
