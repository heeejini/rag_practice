# app/api.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import time
from src.config import ChunkConfig, QdrantConfig, EmbedConfig, LLMConfig
from src.pipeline import RAGPipeline
import os 
app = FastAPI(title="InfoFla RAG API", version="1.0.0")

# -------------------------------
# 1) 템플릿 설정
# -------------------------------
templates = Jinja2Templates(directory="app/templates")

# -------------------------------
# 2) 파이프라인 초기화
# -------------------------------
chunk_cfg = ChunkConfig(
    chunk_size=2000,
    overlap=500,
    strip_brackets=True,
)

qdrant_cfg = QdrantConfig(
    host=os.getenv("QDRANT_HOST", "qdrant"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
    collection="news_chunks",
    recreate=False,
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

pipe = RAGPipeline(
    chunk_cfg=chunk_cfg,
    qdrant_cfg=qdrant_cfg,
    embed_cfg=embed_cfg,
    llm_cfg=llm_cfg,
)

# -------------------------------
# 3) JSON용 스키마 (POST /rag)
# -------------------------------
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


# -------------------------------
# 4) 헬스체크
# -------------------------------
@app.get("/health")
def health():
    backend = "vllm" if llm_cfg.use_vllm else "hf"
    return {"status": "ok", "backend": backend}


# -------------------------------
# 5) HTML UI: GET /
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    처음 접속 시 빈 폼 + 기본 상태.
    """
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


# -------------------------------
# 6) HTML UI: 폼 제출 (POST /chat)
# -------------------------------
@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    query: str = Form(...),
    use_rag: bool = Form(False),
    topk: int = Form(3),
):
    """
    index.html 폼에서 넘어온 값으로 RAG/No-RAG 수행.
    """
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
            # 현재 pipeline.answer_rag 가 (answer, rag_ctx, stats) 를 반환한다고 가정
            result = pipe.answer_rag(
                query,
                hits,
                max_chunks=3,
                max_each=800,
            )
            # (answer, rag_ctx, stats) 또는 (answer, rag_ctx) 방어적으로 처리
            if len(result) == 3:
                ans, ctx, stats = result
                backend = stats.get("llm_backend", backend)
                llm_latency_ms = stats.get("llm_latency", 0.0) * 1000
            else:
                ans, ctx = result
            answer = ans
            rag_ctx = ctx
        else:
            result = pipe.answer_no_rag(query)
            if len(result) == 2:
                ans, stats = result
                backend = stats.get("llm_backend", backend)
                llm_latency_ms = stats.get("llm_latency", 0.0) * 1000
            else:
                ans = result
            answer = ans
            rag_ctx = None

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


# -------------------------------
# 7) JSON RAG API (POST /rag)
# -------------------------------
@app.post("/rag", response_model=RAGResponse)
def rag_endpoint(req: RAGRequest):
    t0 = time.time()

    if req.use_rag:
        hits = pipe.retrieve(req.query, topk=req.topk)
        result = pipe.answer_rag(
            req.query,
            hits,
            max_chunks=3,
            max_each=800,
        )
        if len(result) == 3:
            answer, rag_ctx, stats = result
            backend = stats.get("llm_backend", "unknown")
            llm_latency = stats.get("llm_latency", (time.time() - t0))
        else:
            answer, rag_ctx = result
            backend = "unknown"
            llm_latency = (time.time() - t0)
    else:
        result = pipe.answer_no_rag(req.query)
        if len(result) == 2:
            answer, stats = result
            backend = stats.get("llm_backend", "unknown")
            llm_latency = stats.get("llm_latency", (time.time() - t0))
            rag_ctx = None
        else:
            answer = result
            backend = "unknown"
            llm_latency = (time.time() - t0)
            rag_ctx = None

    t1 = time.time()
    return RAGResponse(
        answer=answer,
        context=rag_ctx,
        backend=backend,
        llm_latency_ms=llm_latency * 1000,
        total_latency_ms=(t1 - t0) * 1000,
    )
