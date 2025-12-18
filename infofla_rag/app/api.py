# app/api.py
import hashlib  
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import pdfplumber
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue
from src.text_utils import normalize_paragraphs, split_text_to_chunks

from app.gradio_ui import attach_gradio
from src.logging_config import setup_logging
from src.pipeline import RAGPipeline
from src.qdrant import ensure_or_recreate_collection
from src.schemas import Chunk
from src.settings import (
    INDEX_SRC_DIR_DEFAULT,
    MAX_QUERY_CHARS,
    UPLOAD_DIR,
    chunk_cfg,
    embed_cfg,
    llm_cfg,
    qdrant_cfg,
)

setup_logging()
logger = logging.getLogger("rag.api")


app = FastAPI(title="INFOFLA RAG API", version="1.0.0")

class RAGRequest(BaseModel):
    query: str
    topk: int = 3
    use_rag: bool = True
    score_threshold : float =0.65


class RAGResponse(BaseModel):
    answer: str
    context: str | None = None
    backend: str
    llm_latency_ms: float
    total_latency_ms: float


class IndexRequest(BaseModel):
    src_dir: str | None = None
    pattern: str = "*.jsonl"
    recreate: bool = False


class IndexResponse(BaseModel):
    indexed_chunks: int
    src_dir: str
    recreate: bool
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 항상 컬렉션을 recreate 하고, 기본 jsonl 코퍼스를 인덱싱."""
    try:
        pipe = get_pipeline()

        # 1) 항상 컬렉션 recreate
        try:
            dim = pipe.embedder.get_sentence_embedding_dimension()
            ensure_or_recreate_collection(
                client=pipe.client,
                collection=pipe.qdrant_cfg.collection,
                dim=dim,
                recreate=True,  # ✅ 항상 드랍 후 재생성
            )
            logger.info(
                "[startup] Collection recreated | collection=%s | dim=%d",
                pipe.qdrant_cfg.collection,
                dim,
            )
        except Exception as e:
            logger.error(
                "[startup] Failed to recreate collection: %s",
                str(e),
                exc_info=True,
            )
            return  

        if os.path.isdir(INDEX_SRC_DIR_DEFAULT):
            try:
                chunks = pipe.chunk(
                    src_dir=INDEX_SRC_DIR_DEFAULT,
                    pattern="*.jsonl",
                )
                n_chunks = len(chunks)

                if n_chunks > 0:
                    pipe.upsert(chunks)
                    logger.info(
                        "[startup] Index built successfully "
                        "| indexed_chunks=%d | src_dir=%s",
                        n_chunks,
                        INDEX_SRC_DIR_DEFAULT,
                    )
                else:
                    logger.warning(
                        "[startup] No chunks found in %s",
                        INDEX_SRC_DIR_DEFAULT,
                    )
            except Exception as e:
                logger.error(
                    "[startup] Failed to build index from %s: %s",
                    INDEX_SRC_DIR_DEFAULT,
                    str(e),
                    exc_info=True,
                )
        else:
            logger.warning(
                "[startup] Source directory not found: %s",
                INDEX_SRC_DIR_DEFAULT,
            )

    except Exception as e:
        logger.error(
            "[startup] Failed to initialize pipeline or build index: %s",
            str(e),
            exc_info=True,
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


@app.get("/health")
def health():
    backend = "vllm" if llm_cfg.use_vllm else "hf"
    return {"status": "ok", "backend": backend}


@app.post("/rag", response_model=RAGResponse)
def rag_endpoint(
    req: RAGRequest,
    pipe : RAGPipeline = Depends(get_pipeline)
    ):
    t0 = time.time()
    logger.info(
        "[/rag] request received | query=%r | use_rag=%s | topk=%d | score_threshold=%f",
        req.query[:200],  # 너무 길면 잘라서
        req.use_rag,
        req.topk,
        req.score_threshold
    )
    # 입력 길이 검사
    if len(req.query) > MAX_QUERY_CHARS:
        logger.warning(
            "[/rag] query too long | length=%d | max=%d",
            len(req.query),
            MAX_QUERY_CHARS,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Query too long: {len(req.query)} characters. "
                   f"Maximum allowed is {MAX_QUERY_CHARS}.",
        )
    try:
        if req.use_rag:
            hits = pipe.retrieve(req.query, topk=req.topk)
            result = pipe.answer_rag(
                req.query,
                hits,
                max_chunks=req.topk,
                max_each=800,
                score_threshold=req.score_threshold,
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

@app.post("/build_index", response_model=IndexResponse)
def build_index_endpoint(
        req: IndexRequest,
        pipe: RAGPipeline = Depends(get_pipeline)
    ):
    src_dir = req.src_dir or INDEX_SRC_DIR_DEFAULT
    logger.info(
        "[/build_index] start | src_dir=%s | pattern=%s | recreate=%s",
        src_dir,
        req.pattern,
        req.recreate,
    )

    if not os.path.isdir(src_dir):
        logger.error(
            "[/build_index] source directory not found: %s",
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
            "[/build_index] collection recreated | collection=%s | dim=%d",
            pipe.qdrant_cfg.collection,
            dim,
        )

    chunks = pipe.chunk(src_dir=src_dir, pattern=req.pattern)
    n_chunks = len(chunks)

    if n_chunks == 0:
        logger.warning(
            "[/build_index] no chunks found | src_dir=%s | pattern=%s",
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
        "[/build_index] finished | indexed_chunks=%d | src_dir=%s",
        n_chunks,
        src_dir,
    )

    return IndexResponse(
        indexed_chunks=n_chunks,
        src_dir=src_dir,
        recreate=req.recreate,
    )
class UploadResponse(BaseModel):
    file_name: str
    num_chunks: int
    collection: str


@app.post("/upload_doc", response_model=UploadResponse)
async def upload_doc_endpoint(
    file: UploadFile = File(...),
    pipe: RAGPipeline = Depends(get_pipeline),
):
    """사용자가 업로드한 PDF/TXT 파일을 인덱싱해서 Qdrant 컬렉션에 추가"""
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: {ext} (지원: .pdf, .txt)",
        )

    # 0) 업로드 파일 바이트 읽기 + 해시 계산
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 읽기 중 오류 발생: {e}",
        )

    # 문서 전체 SHA-256 해시
    file_hash = hashlib.sha256(content).hexdigest()

    # 0-1) Qdrant에 동일 hash 가 이미 있는지 확인
    try:
        points, _ = pipe.client.scroll(
            collection_name=pipe.qdrant_cfg.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.file_hash",   
                        match=MatchValue(value=file_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"중복 검사 중 Qdrant 오류 발생: {e}",
        )

    if points:
        # 이미 동일한 문서가 있음 → 409 Conflict
        raise HTTPException(
            status_code=409,
            detail=f"이미 동일한 문서가 업로드되어 있습니다: {filename}",
        )

    # 1) 파일 저장
    save_path = Path(UPLOAD_DIR) / filename
    try:
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 저장 중 오류 발생: {e}",
        )

    # 2) 텍스트 추출
    try:
        if ext == ".pdf":
            texts = []
            with pdfplumber.open(str(save_path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    texts.append(page_text)
            full_text = "\n".join(texts)
        else:  # .txt
            full_text = save_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"텍스트 추출 중 오류 발생: {e}",
        )

    full_text = full_text.strip()
    if not full_text:
        raise HTTPException(
            status_code=400,
            detail="추출된 텍스트가 비어 있습니다.",
        )

    # 🔥 PDF 줄바꿈 전처리 적용 (이미 쓰고 있던 함수)
    full_text = normalize_paragraphs(full_text)

    # 3) 텍스트를 청크로 분할
    text_chunks = split_text_to_chunks(
        full_text,
        chunk_size=pipe.chunk_cfg.chunk_size,
        overlap=pipe.chunk_cfg.overlap,
    )

    chunks: list[Chunk] = []
    for idx, ch in enumerate(text_chunks):
        chunks.append(
            Chunk(
                text=ch,
                source_path=str(save_path),
                source_name=filename,
                chunk_index=idx,
                metadata={
                    "uploaded": True,
                    "file_hash": file_hash,   # ⬅️ 여기서 file_hash를 metadata에 저장
                },
            )
        )

    # 4) Qdrant에 upsert
    try:
        pipe.upsert(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Qdrant upsert 중 오류 발생: {e}",
        )

    logger.info(
        "[/upload_doc] indexed uploaded file | file=%s | chunks=%d | collection=%s | file_hash=%s",
        filename,
        len(chunks),
        pipe.qdrant_cfg.collection,
        file_hash,
    )

    return UploadResponse(
        file_name=filename,
        num_chunks=len(chunks),
        collection=pipe.qdrant_cfg.collection,
    )


app = attach_gradio(app, get_pipeline, MAX_QUERY_CHARS)