#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

from src.config import ChunkConfig, EmbedConfig, LLMConfig, QdrantConfig
from src.pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser(description="Chunking → Qdrant → RAG/No-RAG LLM")
    ap.add_argument("--src", default="/home/mlbox-a6000x2/works/ict/infofla_rag/data/news_articles_preprocessing")
    ap.add_argument("--pattern", default="*.txt")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--query", required=True)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--llm", default="K-intelligence/Midm-2.0-Base-Instruct")
    ap.add_argument("--embedding_model", default="Alibaba-NLP/gte-multilingual-base")
    ap.add_argument("--chunk_size", type=int, default=2000)
    ap.add_argument("--overlap", type=int, default=500)

    # ✅ vLLM 사용 여부 플래그
    ap.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM OpenAI server instead of local HF model",
    )

    args = ap.parse_args()

    chunk_cfg = ChunkConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strip_brackets=True,
    )
    qdrant_cfg = QdrantConfig(
        host="127.0.0.1",
        port=6333,
        collection="news_chunks",
        recreate=args.recreate,
    )
    embed_cfg = EmbedConfig(
        model_name=args.embedding_model,
        batch_size=256,
    )

    # ⚠️ 이 코드가 동작하려면 LLMConfig에
    # use_vllm: bool, vllm_api_base: str 필드가 정의돼 있어야 한다.
    llm_cfg = LLMConfig(
        model_id=args.llm,
        max_new_tokens=256,
        do_sample=False,
        use_vllm=args.use_vllm,
        vllm_api_base=os.getenv("VLLM_API_BASE", "http://localhost:8000"),
    )

    pipe = RAGPipeline(chunk_cfg, qdrant_cfg, embed_cfg, llm_cfg)

    # 1) 청킹 (메모리)
    chunks = pipe.chunk(args.src, pattern=args.pattern)
    print(f"[INFO] 청크 {len(chunks)}개")

    # 2) 업서트
    pipe.upsert(chunks)
    print("[INFO] Qdrant 업서트 완료")

    # 3) 검색 + 생성 (RAG)
    t0 = time.time()
    hits = pipe.retrieve(args.query, topk=args.topk)
    rag_answer, rag_ctx, stats = pipe.answer_rag(
        args.query,
        hits,
        max_chunks=3,
        max_each=800,
    )
    t1 = time.time()
    end_to_end_latency = t1 - t0

    print("\n=== RAG 컨텍스트 일부 ===")
    print(rag_ctx[:800])

    print("\n=== RAG 답변 ===")
    print(rag_answer)

    # 🔎 Latency/성능 로그
    print("\n=== Latency / 성능 정보 ===")
    print(f"LLM backend       : {stats['llm_backend']}")         # 'hf' or 'vllm'
    print(f"LLM latency       : {stats['llm_latency']:.3f} s")
    print(f"End-to-end latency: {end_to_end_latency:.3f} s")
    print(f"max_new_tokens    : {stats['max_new_tokens']}")
    print(f"do_sample         : {stats['do_sample']}")

    # 4) No-RAG 비교
    no_rag = pipe.answer_no_rag(args.query)
    print("\n=== No-RAG 답변 ===")
    print(no_rag)


if __name__ == "__main__":
    main()
