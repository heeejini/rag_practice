from typing import List, Dict
import time

from sentence_transformers import SentenceTransformer

from .config import ChunkConfig, QdrantConfig, EmbedConfig, LLMConfig
from .chunking import chunk_dir_to_list
from .qdrant import get_client, ensure_or_recreate_collection, upsert_chunks, search
from .prompt import build_rag_context, make_prompt_chat
from .llm import load_llm, generate_answer
from .vLLM import VLLMClient   # ✅ vLLM 클라이언트


class RAGPipeline:
    def __init__(
        self,
        chunk_cfg: ChunkConfig = ChunkConfig(),
        qdrant_cfg: QdrantConfig = QdrantConfig(),
        embed_cfg: EmbedConfig = EmbedConfig(),
        llm_cfg: LLMConfig = LLMConfig(),
    ):
        self.chunk_cfg = chunk_cfg
        self.qdrant_cfg = qdrant_cfg
        self.embed_cfg = embed_cfg
        self.llm_cfg = llm_cfg

        # -----------------------------
        # 1) Embedder / Qdrant 초기화
        # -----------------------------
        self.embedder = SentenceTransformer(
            embed_cfg.model_name,
            trust_remote_code=embed_cfg.trust_remote_code,
        )
        self.client = get_client(qdrant_cfg.host, qdrant_cfg.port)
        dim = self.embedder.get_sentence_embedding_dimension()
        ensure_or_recreate_collection(
            self.client,
            qdrant_cfg.collection,
            dim,
            qdrant_cfg.recreate,
        )

        # -----------------------------
        # 2) LLM 백엔드 선택 (HF vs vLLM)
        # -----------------------------
        if self.llm_cfg.use_vllm:
            # vLLM(OpenAI 호환) 서버 호출용 클라이언트
            self.vllm_client = VLLMClient(self.llm_cfg)
            self.model = None
            self.tokenizer = None
            self.gen_cfg = None
        else:
            # 기존 HF Transformers 로컬 로딩
            self.model, self.tokenizer, self.gen_cfg = load_llm(llm_cfg.model_id)
            self.vllm_client = None

    # 1) chunking → memory
    def chunk(self, src_dir: str, pattern: str = "*.txt") -> List[Dict]:
        return chunk_dir_to_list(
            src_dir=src_dir,
            chunk_size=self.chunk_cfg.chunk_size,
            chunk_overlap=self.chunk_cfg.overlap,
            strip_brackets=self.chunk_cfg.strip_brackets,
            pattern=pattern,
        )

    # 2) upsert
    def upsert(self, chunks: List[Dict]):
        upsert_chunks(
            client=self.client,
            collection=self.qdrant_cfg.collection,
            embedder=self.embedder,
            chunks=chunks,
            batch_size=self.embed_cfg.batch_size,
        )

    # 3) search
    def retrieve(self, query: str, topk: int = 3):
        return search(
            client=self.client,
            collection=self.qdrant_cfg.collection,
            embedder=self.embedder,
            query=query,
            topk=topk,
        )

    # 4) build context + LLM generate (RAG)
    #    → (answer, rag_ctx, stats) 반환
    def answer_rag(
        self,
        query: str,
        hits,
        max_chunks: int = 3,
        max_each: int = 800,
        max_context_chars: int = 3000,
    ):
        rag_ctx = build_rag_context(
            hits,
            max_chunks=max_chunks,
            max_each=max_each,
        )
        messages = make_prompt_chat(
            query=query,
            rag=rag_ctx,
            max_context_chars=max_context_chars,
        )

        # LLM 부분만 latency 측정
        t0 = time.time()
        if self.llm_cfg.use_vllm:
            llm_backend = "vllm"
            answer = self.vllm_client.chat(
                messages,
                max_new_tokens=self.llm_cfg.max_new_tokens,
                do_sample=self.llm_cfg.do_sample,
            )
        else:
            llm_backend = "hf"
            answer = generate_answer(
                self.model,
                self.tokenizer,
                self.gen_cfg,
                messages,
                max_new_tokens=self.llm_cfg.max_new_tokens,
                do_sample=self.llm_cfg.do_sample,
            )
        t1 = time.time()
        llm_latency = t1 - t0

        stats = {
            "llm_backend": llm_backend,                     # "hf" or "vllm"
            "llm_latency": llm_latency,                     # 초 단위
            "max_new_tokens": self.llm_cfg.max_new_tokens,
            "do_sample": self.llm_cfg.do_sample,
        }

        return answer, rag_ctx, stats

    # 5) No-RAG (context 없이 질문만)
    #    → 여기서는 일단 answer만 리턴 유지 (필요하면 stats도 추가 가능)
    def answer_no_rag(self, query: str):
        messages = make_prompt_chat(query=query, rag=None)

        if self.llm_cfg.use_vllm:
            return self.vllm_client.chat(
                messages,
                max_new_tokens=self.llm_cfg.max_new_tokens,
                do_sample=self.llm_cfg.do_sample,
            )
        else:
            return generate_answer(
                self.model,
                self.tokenizer,
                self.gen_cfg,
                messages,
                max_new_tokens=self.llm_cfg.max_new_tokens,
                do_sample=self.llm_cfg.do_sample,
            )
