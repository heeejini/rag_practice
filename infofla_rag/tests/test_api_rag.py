import os
import requests
import pytest
from fastapi.testclient import TestClient

import warnings

from app.api import app, get_pipeline
from src.schemas import RAGResult, GenerationStats

# Suppress known deprecation warnings from FastAPI / Starlette during tests
warnings.filterwarnings(
    "ignore",
    message=".*on_event is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*first parameter should be the Request instance.*",
    category=DeprecationWarning,
)


class DummyStats:
    def __init__(self):
        self.llm_backend = "vllm"
        self.llm_latency = 0.123


class DummyPipe:
    def retrieve(self, query: str, topk: int = 3):
        return [{"id": "1", "score": 1.0, "text": "dummy context"}]

    def answer_rag(self, query, hits, max_chunks=3, max_each=800, max_context_chars=3000):
        return RAGResult(
            answer=f"dummy answer for: {query}",
            context="dummy context",
            stats=GenerationStats(
                llm_backend="vllm",
                llm_latency=0.123,
                max_new_tokens=256,
                do_sample=False,
            ),
        )

    def answer_no_rag(self, query: str):
        return RAGResult(
            answer=f"no rag answer for: {query}",
            context=None,
            stats=GenerationStats(
                llm_backend="vllm",
                llm_latency=0.123,
                max_new_tokens=256,
                do_sample=False,
            ),
        )


client = TestClient(app)


def override_get_pipeline():
    return DummyPipe()


def test_rag_endpoint_with_rag_enabled():
    app.dependency_overrides[get_pipeline] = override_get_pipeline

    payload = {
        "query": "테스트 질문입니다",
        "topk": 3,
        "use_rag": True,
    }
    resp = client.post("/rag", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert "dummy answer for" in data["answer"]
    assert data["context"] == "dummy context"
    assert data["backend"] == "vllm"
    assert data["llm_latency_ms"] >= 0
    assert data["total_latency_ms"] >= 0
    # RAG 포함 응답 구조가 마는 지, context 포함되는 지, backend 가 정해지는 지, latency 값들이 있는 지 등 확인
    app.dependency_overrides.clear()


def test_rag_endpoint_without_rag():
    app.dependency_overrides[get_pipeline] = override_get_pipeline

    payload = {
        "query": "RAG를 쓰지 않는 질문",
        "topk": 3,
        "use_rag": False,
    }
    resp = client.post("/rag", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert "no rag answer for" in data["answer"]
    assert data["context"] is None or data["context"] == ""

    app.dependency_overrides.clear()


def test_chat_renders_html_with_rag():
    app.dependency_overrides[get_pipeline] = override_get_pipeline

    resp = client.post(
        "/chat",
        data={
            "query": "테스트 질문입니다",
            "use_rag": "on", 
            "topk": "3",
        },
    )

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]

    # 렌더링된 HTML 안에 DummyPipe가 만든 answer/context가 들어갔는지 확인
    html = resp.text
    assert "dummy answer for" in html
    assert "dummy context" in html

    app.dependency_overrides.clear()

def test_health_endpoint():
    # /health 엔드포인트가 정상적으로 응답하는지 확인
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend"] in ("hf", "vllm")


# ------------ Live service checks (optional integration) ------------

def _skip_if_unreachable(url: str, exc: Exception):
    pytest.skip(f"Service unreachable at {url}: {exc}")


def test_live_rag_server_alive():
    """실제 rag 서버(9000) 헬스 체크"""
    url = os.getenv("RAG_HEALTH_URL", "http://localhost:9000/health")
    try:
        resp = requests.get(url, timeout=5)
    except Exception as exc:
        _skip_if_unreachable(url, exc)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


def test_live_vllm_server_alive():
    """실제 vLLM 서버(8001) 헬스 체크"""
    url = os.getenv("VLLM_HEALTH_URL", "http://localhost:8001/health")
    try:
        resp = requests.get(url, timeout=5)
    except Exception as exc:
        _skip_if_unreachable(url, exc)
    assert resp.status_code == 200
    # vLLM의 /health는 단순 문자열 또는 JSON을 반환할 수 있음
    try:
        data = resp.json()
        assert data is not None
    except Exception:
        # JSON이 아니더라도 200 OK면 통과
        assert resp.text is not None


def test_live_qdrant_server_alive():
    """실제 Qdrant 서버(6335) 헬스 체크"""
    url = os.getenv("QDRANT_HEALTH_URL", "http://localhost:6335/collections")
    try:
        resp = requests.get(url, timeout=5)
    except Exception as exc:
        _skip_if_unreachable(url, exc)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
