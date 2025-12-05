from fastapi.testclient import TestClient

from app.api import app, get_pipeline


class DummyStats:
    def __init__(self):
        self.llm_backend = "vllm"
        self.llm_latency = 0.123


class DummyPipe:
    def retrieve(self, query: str, topk: int = 3):
        return [{"id": "1", "score": 1.0, "text": "dummy context"}]

    def answer_rag(self, query, hits, max_chunks=3, max_each=800, max_context_chars=3000):
        answer = f"dummy answer for: {query}"
        rag_ctx = "dummy context"
        stats = DummyStats()
        return answer, rag_ctx, stats

    def answer_no_rag(self, query: str):
        answer = f"no rag answer for: {query}"
        return answer


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
