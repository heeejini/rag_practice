# app/gradio_ui.py
import os
import time
import mimetypes

import gradio as gr
import requests
from fastapi import FastAPI

from src.pipeline import RAGPipeline


def create_gradio_demo(
    get_pipeline,
    max_query_chars: int,
) -> gr.Blocks:
    """Gradio Blocks UI 생성."""

    # ------------------------
    # 1) 질문 처리 함수
    # ------------------------
    def gradio_chat_fn(query: str, use_rag: bool, topk: int):
        if not query.strip():
            return "질문을 입력하세요.", "", ""

        notice = ""
        if len(query) > max_query_chars:
            notice = (
                f"[알림] 입력이 너무 길어서 앞 {max_query_chars}자만 사용합니다. "
                f"(원래 길이: {len(query)}자)\n\n"
            )
            query_trimmed = query[:max_query_chars]
        else:
            query_trimmed = query

        t0 = time.time()
        pipe: RAGPipeline = get_pipeline()

        try:
            if use_rag:
                hits = pipe.retrieve(query_trimmed, topk=topk)
                result = pipe.answer_rag(
                    query=query_trimmed,
                    hits=hits,
                    max_chunks=topk,
                    max_each=800,
                    max_context_chars=3000,
                )
                context = result.context or ""
            else:
                result = pipe.answer_no_rag(query_trimmed)
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

            return notice + answer, context, stats_text

        except Exception as e:
            return f"[에러] {e}", "", ""

    # ------------------------
    # 2) 업로드 처리 함수
    # ------------------------
    def gradio_upload_fn(file_path: str | None):
        if file_path is None:
            return "⚠️ 파일을 선택하세요."

        if not os.path.exists(file_path):
            return f"⚠️ 파일을 찾을 수 없습니다: {file_path}"

        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or "application/octet-stream"

        # FastAPI 업로드 엔드포인트
        url = "http://127.0.0.1:9000/admin/upload_doc"

        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, mime_type)}
                resp = requests.post(url, files=files)

            if resp.status_code == 200:
                return f"✅ 업로드 성공!\n{resp.json()}"
            if resp.status_code == 409:
                return f"⚠️ 이미 동일한 문서가 존재합니다.\n{resp.text}"
            return f"❌ 오류 발생 ({resp.status_code})\n{resp.text}"

        except Exception as e:
            return f"[예외 발생] {e}"

    # ------------------------
    # 3) Gradio Blocks + 탭 구성
    # ------------------------
    with gr.Blocks(title="InfoFla RAG Demo 🤩") as demo:
        gr.HTML(
            """
        <h1>InfoFla RAG 데모</h1>
        <div style="text-align: center; color: #64748b; font-size: 0.95rem; margin-bottom: 1rem;">
          Backend: <strong>vLLM / HF</strong> |
          API Docs: <a href="/docs" target="_blank">/docs</a> |
          Health: <a href="/health" target="_blank">/health</a>
        </div>
        """
        )

        # 🔹 탭 1: 문서 업로드 (왼쪽 / 첫 번째 탭)
        with gr.Tab("문서 업로드 📄"):
            gr.Markdown("### 1️⃣ PDF / TXT 문서를 업로드하여 RAG 인덱스에 추가합니다.")

            upload_file = gr.File(
                label="문서 업로드 (PDF 또는 TXT)",
                file_types=[".pdf", ".txt"],
                file_count="single",
                type="filepath",
            )
            upload_btn = gr.Button("인덱싱 실행")
            upload_output = gr.Textbox(
                label="결과",
                lines=5,
                interactive=False,
            )

            upload_btn.click(
                fn=gradio_upload_fn,
                inputs=[upload_file],
                outputs=[upload_output],
            )

        # 🔹 탭 2: 질문하기
        with gr.Tab("질문하기 💬"):
            gr.Markdown("### 2️⃣ 인덱싱된 문서 기반으로 질문을 해보세요.")

            query = gr.Textbox(
                label="질문",
                placeholder="질문을 입력하세요. (예: infofla 셀토 알려줘)",
                lines=4,
            )

            with gr.Row():
                use_rag = gr.Checkbox(label="RAG 사용", value=True)
                topk = gr.Slider(
                    label="Top-k",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=3,
                )

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

            stats_box = gr.Markdown()

            submit_btn.click(
                fn=gradio_chat_fn,
                inputs=[query, use_rag, topk],
                outputs=[answer_box, context_box, stats_box],
            )

    return demo


def attach_gradio(app: FastAPI, get_pipeline, max_query_chars: int) -> FastAPI:
    demo = create_gradio_demo(get_pipeline, max_query_chars)
    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",  
        theme=gr.themes.Citrus(),
        footer_links=["api", "gradio", "settings"],
    )
    return app
