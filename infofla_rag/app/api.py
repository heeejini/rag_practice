# app/api.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
import time
import logging
from functools import lru_cache

import gradio as gr
from gradio.routes import mount_gradio_app

from src.config import ChunkConfig, QdrantConfig, EmbedConfig, LLMConfig
from src.pipeline import RAGPipeline
from src.qdrant import ensure_or_recreate_collection
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("rag.api")


app = FastAPI(title="InfoFla RAG API", version="1.0.0")


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
                    chunks = pipe.chunk(src_dir=INDEX_SRC_DIR_DEFAULT, pattern="*.txt")
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
                max_chunks=req.topk,
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

import time
import gradio as gr
from gradio.routes import mount_gradio_app
def gradio_chat_fn(query: str, use_rag: bool, topk: int):
    if not query.strip():
        return "질문을 입력하세요.", "", ""

    t0 = time.time()

    # 🔥 여기서 파이프라인 인스턴스 가져오기
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

        return answer, context, stats_text

    except Exception as e:
        return f"[에러] {e}", "", ""


css_block = """
<style>
  body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 16px;
    line-height: 1.5;
  }
  h1 {
    margin-bottom: 8px;
  }
  .meta {
    color: #666;
    font-size: 0.9rem;
    margin-bottom: 16px;
  }
  .answer,
  .context,
  .stats {
    margin-top: 16px;
    padding: 12px;
    border-radius: 4px;
    white-space: pre-wrap;
  }
  .answer {
    background: #f1f5f9;
  }
  .context {
    background: #f9fafb;
    font-size: 0.9rem;
    border: 1px dashed #cbd5f5;
  }
  .stats {
    font-size: 0.85rem;
    color: #4b5563;
  }
</style>
"""

gradio_demo = gr.Blocks(title="InfoFla RAG Demo ver.1")

with gradio_demo:
    gr.HTML(css_block)
    gr.HTML("""
    <h1>InfoFla RAG 데모</h1>
    <div class="meta">
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
    use_rag = gr.Checkbox(label="RAG 사용", value=True)
    topk = gr.Slider(label="Top-k", minimum=1, maximum=10, step=1, value=3)

    submit_btn = gr.Button("질문 보내기")

    answer_box = gr.Textbox(label="답변", interactive=False)
    context_box = gr.Textbox(label="RAG 컨텍스트", interactive=False)
    stats_box = gr.Markdown(label="통계")

    submit_btn.click(
        fn=gradio_chat_fn,
        inputs=[query, use_rag, topk],
        outputs=[answer_box, context_box, stats_box],
    )

app = mount_gradio_app(app, gradio_demo, path="/")
