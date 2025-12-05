# app/api.py
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import time
import os

from src.config import ChunkConfig, QdrantConfig, EmbedConfig, LLMConfig
from src.pipeline import RAGPipeline
from src.qdrant import ensure_or_recreate_collection

import logging
from src.logging_config import setup_logging

from functools import lru_cache

setup_logging()
logger = logging.getLogger("rag.api")


app = FastAPI(title="InfoFla RAG API", version="1.0.0")

templates = Jinja2Templates(directory="app/templates")

chunk_cfg = ChunkConfig(
    chunk_size=2000,
    overlap=500,
    strip_brackets=True,
)

qdrant_cfg = QdrantConfig(
    host=os.getenv("QDRANT_HOST", "qdrant"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
    collection="news_chunks",
    recreate=False,  # API 서버에서는 기본적으로 컬렉션 삭제 X
)

embed_cfg = EmbedConfig(
    model_name="Alibaba-NLP/gte-multilingual-base",
    batch_size=256,
)

llm_cfg = LLMConfig(
    model_id="K-intelligence/Midm-2.0-Base-Instruct",
    max_new_tokens=256,
    do_sample=False,
    temperature=0.2,
    use_vllm=True,
    vllm_api_base=os.getenv("VLLM_API_BASE", "http://vllm:8000"),
)

INDEX_SRC_DIR_DEFAULT = os.getenv(
    "INDEX_SRC_DIR",
    "/app/data/news_articles_preprocessing",
)

@lru_cache
def get_pipeline() -> RAGPipeline:
    logger.info("Initializing RAGPipeline...")
    return RAGPipeline(
        chunk_cfg=chunk_cfg,
        qdrant_cfg=qdrant_cfg,
        embed_cfg=embed_cfg,
        llm_cfg=llm_cfg,
    )

class RAGRequest(BaseModel):
    query: str
    topk: int = 3
    use_rag: bool = True


class RAGResponse(BaseModel):
    answer: str
    context: str | None = None
    backend: str
    llm_latency_ms: float
    total_latency_ms: float


class IndexRequest(BaseModel):
    src_dir: str | None = None
    pattern: str = "*.txt"
    recreate: bool = False


class IndexResponse(BaseModel):
    indexed_chunks: int
    src_dir: str
    recreate: bool


@app.get("/health")
def health():
    backend = "vllm" if llm_cfg.use_vllm else "hf"
    return {"status": "ok", "backend": backend}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    backend = "vllm" if llm_cfg.use_vllm else "hf"
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": "",
            "answer": None,
            "context": None,
            "backend": backend,
            "llm_latency_ms": None,
            "total_latency_ms": None,
            "use_rag": True,
            "topk": 3,
            "error": None,
        },
    )


@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    query: str = Form(...),
    use_rag: bool = Form(False),
    topk: int = Form(3),
    pipe: RAGPipeline = Depends(get_pipeline),  
):
    t0 = time.time()
    answer = ""
    rag_ctx = None
    backend = "vllm" if llm_cfg.use_vllm else "hf"
    llm_latency_ms = 0.0
    total_latency_ms = 0.0
    error = None

    try:
        if use_rag:
            hits = pipe.retrieve(query, topk=topk)
            result = pipe.answer_rag(
                query,
                hits,
                max_chunks=3,
                max_each=800,
            )
            answer = result.answer
            rag_ctx = result.context
            if result.stats:
                backend = result.stats.llm_backend
                llm_latency_ms = result.stats.llm_latency * 1000
        else:
            result = pipe.answer_no_rag(query)
            answer = result.answer
            rag_ctx = result.context
            if result.stats:
                backend = result.stats.llm_backend
                llm_latency_ms = result.stats.llm_latency * 1000

        t1 = time.time()
        total_latency_ms = (t1 - t0) * 1000

    except Exception as e:
        error = str(e)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": query,
            "answer": answer,
            "context": rag_ctx,
            "backend": backend,
            "llm_latency_ms": llm_latency_ms,
            "total_latency_ms": total_latency_ms,
            "use_rag": use_rag,
            "topk": topk,
            "error": error,
        },
    )


@app.post("/rag", response_model=RAGResponse)
def rag_endpoint(
    req: RAGRequest,
    pipe : RAGPipeline = Depends(get_pipeline)
    ):
    t0 = time.time()
    logger.info(
        "[/rag] request received | query=%r | use_rag=%s | topk=%d",
        req.query[:200],  # 너무 길면 잘라서
        req.use_rag,
        req.topk,
    )

    try:
        if req.use_rag:
            hits = pipe.retrieve(req.query, topk=req.topk)
            result = pipe.answer_rag(
                req.query,
                hits,
                max_chunks=3,
                max_each=800,
            )
            answer = result.answer
            rag_ctx = result.context
            if result.stats:
                backend = result.stats.llm_backend
                llm_latency = result.stats.llm_latency
            else:
                backend = "unknown"
                llm_latency = (time.time() - t0)
        else:
            result = pipe.answer_no_rag(req.query)
            answer = result.answer
            rag_ctx = result.context
            if result.stats:
                backend = result.stats.llm_backend
                llm_latency = result.stats.llm_latency
            else:
                backend = "unknown"
                llm_latency = (time.time() - t0)

        t1 = time.time()
        total_latency = t1 - t0

        logger.info(
            "[/rag] response | backend=%s | llm_latency=%.3fs | total_latency=%.3fs",
            backend,
            llm_latency,
            total_latency,
        )

        return RAGResponse(
            answer=answer,
            context=rag_ctx,
            backend=backend,
            llm_latency_ms=llm_latency * 1000,
            total_latency_ms=total_latency * 1000,
        )

    except Exception:
        logger.exception("[/rag] unhandled exception")
        raise

@app.post("/admin/build_index", response_model=IndexResponse)
def build_index_endpoint(
        req: IndexRequest,
        pipe: RAGPipeline = Depends(get_pipeline)
    ):
    src_dir = req.src_dir or INDEX_SRC_DIR_DEFAULT
    logger.info(
        "[/admin/build_index] start | src_dir=%s | pattern=%s | recreate=%s",
        src_dir,
        req.pattern,
        req.recreate,
    )

    if not os.path.isdir(src_dir):
        logger.error(
            "[/admin/build_index] source directory not found: %s",
            src_dir,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Source directory not found: {src_dir}",
        )

    if req.recreate:
        dim = pipe.embedder.get_sentence_embedding_dimension()
        ensure_or_recreate_collection(
            client=pipe.client,
            collection=pipe.qdrant_cfg.collection,
            dim=dim,
            recreate=True,
        )
        logger.info(
            "[/admin/build_index] collection recreated | collection=%s | dim=%d",
            pipe.qdrant_cfg.collection,
            dim,
        )

    chunks = pipe.chunk(src_dir=src_dir, pattern=req.pattern)
    n_chunks = len(chunks)

    if n_chunks == 0:
        logger.warning(
            "[/admin/build_index] no chunks found | src_dir=%s | pattern=%s",
            src_dir,
            req.pattern,
        )
        return IndexResponse(
            indexed_chunks=0,
            src_dir=src_dir,
            recreate=req.recreate,
        )

    pipe.upsert(chunks)
    logger.info(
        "[/admin/build_index] finished | indexed_chunks=%d | src_dir=%s",
        n_chunks,
        src_dir,
    )

    return IndexResponse(
        indexed_chunks=n_chunks,
        src_dir=src_dir,
        recreate=req.recreate,
    )
