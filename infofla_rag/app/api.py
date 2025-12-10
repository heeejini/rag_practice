# app/api.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
import os
import time
import logging
from functools import lru_cache

import gradio as gr
import re

from src.config import ChunkConfig, QdrantConfig, EmbedConfig, LLMConfig
from src.pipeline import RAGPipeline
from src.qdrant import ensure_or_recreate_collection
from src.logging_config import setup_logging

from pathlib import Path
import pdfplumber

from src.schemas import Chunk

setup_logging()
logger = logging.getLogger("rag.api")


app = FastAPI(title="InfoFla RAG API", version="1.0.0")

MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "2000"))

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 자동으로 인덱스 빌드 (컬렉션이 비어있을 경우)"""
    try:
        pipe = get_pipeline()
        
        # 컬렉션 상태 확인
        try:
            collection_info = pipe.client.get_collection(qdrant_cfg.collection)
            points_count = collection_info.points_count
            
            if points_count == 0:
                logger.info(
                    "[startup] Collection is empty, building index automatically | "
                    "collection=%s | src_dir=%s",
                    qdrant_cfg.collection,
                    INDEX_SRC_DIR_DEFAULT,
                )
                
                if os.path.isdir(INDEX_SRC_DIR_DEFAULT):
                    chunks = pipe.chunk(src_dir=INDEX_SRC_DIR_DEFAULT, pattern="*.jsonl")
                    n_chunks = len(chunks)
                    
                    if n_chunks > 0:
                        pipe.upsert(chunks)
                        logger.info(
                            "[startup] Index built successfully | indexed_chunks=%d",
                            n_chunks,
                        )
                    else:
                        logger.warning(
                            "[startup] No chunks found in %s",
                            INDEX_SRC_DIR_DEFAULT,
                        )
                else:
                    logger.warning(
                        "[startup] Source directory not found: %s",
                        INDEX_SRC_DIR_DEFAULT,
                    )
            else:
                logger.info(
                    "[startup] Collection already has data | collection=%s | points=%d",
                    qdrant_cfg.collection,
                    points_count,
                )
        except Exception as e:
            # 컬렉션이 없거나 접근 불가능한 경우
            logger.warning(
                "[startup] Could not check collection status: %s. "
                "Index will need to be built manually.",
                str(e),
            )
    except Exception as e:
        logger.error(
            "[startup] Failed to initialize pipeline or build index: %s",
            str(e),
            exc_info=True,
        )

chunk_cfg = ChunkConfig(
    chunk_size=1000,
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
UPLOAD_DIR = os.getenv(
    "UPLOAD_DIR",
    "/app/data/uploads",   
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
class UploadResponse(BaseModel):
    file_name: str
    num_chunks: int
    collection: str


@app.post("/admin/upload_doc", response_model=UploadResponse)
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

    # 1) 파일 저장
    save_path = Path(UPLOAD_DIR) / filename
    try:
        content = await file.read()
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

    # 🔥 PDF 줄바꿈 전처리 적용
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
                metadata={"uploaded": True},
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
        "[/admin/upload_doc] indexed uploaded file | file=%s | chunks=%d | collection=%s",
        filename,
        len(chunks),
        pipe.qdrant_cfg.collection,
    )

    return UploadResponse(
        file_name=filename,
        num_chunks=len(chunks),
        collection=pipe.qdrant_cfg.collection,
    )


def normalize_paragraphs(raw_text: str) -> str:
    """PDF에서 잘못 분리된 줄바꿈을 고쳐 문장 단위로 합쳐주는 전처리."""
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3줄 이상 개행 → 2줄

    paragraphs = text.split("\n\n")
    normalized = []

    for p in paragraphs:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if not lines:
            continue
        normalized.append(" ".join(lines))  # 문단 안에서 줄바꿈 제거 → 공백으로 연결

    return "\n\n".join(normalized)


def split_text_to_chunks(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 500,
):
    """긴 텍스트를 chunk_size / overlap 기준으로 잘라서 리스트로 반환"""
    chunks = []
    start = 0
    n = len(text)

    if n == 0:
        return []

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)

        if end == n:
            break
        # 다음 시작 위치: overlap만큼 겹치게
        start = end - overlap if end - overlap > 0 else end

    return chunks

def gradio_chat_fn(query: str, use_rag: bool, topk: int):
    if not query.strip():
        return "질문을 입력하세요.", "", ""

    # 🔹 입력 길이 체크 + 잘라쓰기
    notice = ""
    if len(query) > MAX_QUERY_CHARS:
        notice = (
            f"[알림] 입력이 너무 길어서 앞 {MAX_QUERY_CHARS}자만 사용합니다. "
            f"(원래 길이: {len(query)}자)\n\n"
        )
        query = query[:MAX_QUERY_CHARS]

    t0 = time.time()

    pipe = get_pipeline()

    try:
        if use_rag:
            hits = pipe.retrieve(query, topk=topk)
            result = pipe.answer_rag(
                query=query,
                hits=hits,
                max_chunks=topk,
                max_each=800,
                max_context_chars=3000,
            )
            context = result.context or ""
        else:
            result = pipe.answer_no_rag(query)
            context = ""

        answer = result.answer
        llm_latency_ms = result.stats.llm_latency * 1000.0 if result.stats else None
        total_latency_ms = (time.time() - t0) * 1000.0

        stats_text = ""
        if llm_latency_ms is not None and total_latency_ms is not None:
            stats_text = (
                f"LLM latency: {llm_latency_ms:.1f} ms\n\n"
                f"Total latency: {total_latency_ms:.1f} ms"
            )

        # 🔹 너무 길어서 잘랐으면 안내 문구를 답변 앞에 붙여주기
        return notice + answer, context, stats_text

    except Exception as e:
        return f"[에러] {e}", "", ""

# 🔹 여기부터 Gradio UI 정의
# ⚠️ Gradio 6에서는 theme 을 Blocks(...) 에 넣지 않고,
#     mount_gradio_app 에 넘겨야 함
with gr.Blocks(title="InfoFla RAG Demo 🤩") as gradio_demo:
    gr.HTML("""
    <h1>InfoFla RAG 데모</h1>
    <div style="text-align: center; color: #64748b; font-size: 0.95rem; margin-bottom: 1rem;">
      Backend: <strong>vLLM / HF</strong> |
      API Docs: <a href="/docs" target="_blank">/docs</a> |
      Health: <a href="/health" target="_blank">/health</a>
    </div>
    """)

    query = gr.Textbox(
        label="질문",
        placeholder="질문을 입력하세요. (예: infofla 셀토 알려줘)",
        lines=4,
    )

    with gr.Row():
        use_rag = gr.Checkbox(label="RAG 사용", value=True)
        topk = gr.Slider(label="Top-k", minimum=1, maximum=10, step=1, value=3)

    submit_btn = gr.Button("질문 보내기")

    answer_box = gr.Textbox(
        label="답변",
        interactive=False,
        lines=10,
    )

    context_box = gr.Textbox(
        label="RAG 컨텍스트",
        interactive=False,
        lines=12,
    )

    stats_box = gr.Markdown()  # label 없어도 됨

    submit_btn.click(
        fn=gradio_chat_fn,
        inputs=[query, use_rag, topk],
        outputs=[answer_box, context_box, stats_box],
    )

    with gr.Tab("문서 업로드"):
        gr.Markdown("### PDF / TXT 문서를 업로드하여 RAG 인덱스에 추가할 수 있습니다.")

        upload_file = gr.File(
            label="문서 업로드 (PDF 또는 TXT)",
            file_types=[".pdf", ".txt"],
            file_count="single",   
            type="filepath", 
        )

        upload_btn = gr.Button("인덱싱 실행")


        upload_output = gr.Textbox(label="결과", lines=5, interactive=False)
        def gradio_upload_fn(file):
            import requests, mimetypes, os

            if file is None:
                return "⚠️ 파일을 선택하세요."

            # file은 filepath (str)
            filepath = file
            filename = os.path.basename(filepath)

            if not os.path.exists(filepath):
                return f"⚠️ 파일을 찾을 수 없습니다: {filepath}"

            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or "application/octet-stream"

            url = "http://127.0.0.1:9000/admin/upload_doc"

            try:
                with open(filepath, "rb") as f:
                    files = {"file": (filename, f, mime_type)}
                    resp = requests.post(url, files=files)

                if resp.status_code == 200:
                    return f"✅ 업로드 성공!\n{resp.json()}"
                else:
                    return f"❌ 오류 발생 ({resp.status_code})\n{resp.text}"

            except Exception as e:
                return f"[예외 발생] {e}"



    upload_btn.click(
        fn=gradio_upload_fn,
        inputs=[upload_file],
        outputs=[upload_output],
    )

app = gr.mount_gradio_app(
    app,
    gradio_demo,
    path="/",                        # 지금처럼 루트에 두려면 "/"
    theme=gr.themes.Citrus(),         # ✅ Soft 테마 여기서 적용
    footer_links=["api", "gradio", "settings"],  # 필요 없으면 [] 나 None 으로
)