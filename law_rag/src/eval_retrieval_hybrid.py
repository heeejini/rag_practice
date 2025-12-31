#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# pip install fastembed
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding


# -----------------------------
# 0) Qdrant response 호환
# -----------------------------
def _extract_points_from_query_response(resp: Any) -> List[Any]:
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    if hasattr(resp, "points"):
        return list(resp.points)
    if hasattr(resp, "result") and hasattr(resp.result, "points"):
        return list(resp.result.points)
    return []


# -----------------------------
# 1) QnA 로드
# -----------------------------
def load_qna(qna_path: str) -> List[Dict[str, Any]]:
    with open(qna_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        # 혹시 래핑돼 있으면
        for k in ("qna", "data", "items", "results"):
            if isinstance(obj.get(k), list):
                return [x for x in obj[k] if isinstance(x, dict)]
        return [obj]
    raise ValueError("qna_sample.json 구조를 파악할 수 없다.")


def parse_gold_refs(item: Dict[str, Any]) -> List[str]:
    """
    qna_sample.json에서 정답 판례번호 리스트 추출
    - "참조 판례": "88누6924, 2008도5984" 처럼 올 수 있음
    - 또는 리스트일 수도 있어서 방어
    """
    gold = item.get("참조 판례") or item.get("gold_refs") or item.get("ref")
    if gold is None:
        return []

    if isinstance(gold, list):
        raw = " ".join(str(x) for x in gold)
    else:
        raw = str(gold)

    # 구분자: 콤마/세미콜론/파이프/슬래시/줄바꿈 등
    parts = re.split(r"[,\|;/\n]+", raw)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p)
    return out


# -----------------------------
# 2) Hybrid Retriever
# -----------------------------
class HybridRetriever:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: Optional[str],
        dense_model_name: str = "dragonkue/BGE-m3-ko",
        bm25_model_name: str = "Qdrant/bm25",
    ):
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.dense_model = SentenceTransformer(dense_model_name, trust_remote_code=True)
        self.bm25 = SparseTextEmbedding(bm25_model_name)

    def _bm25_sparse(self, text: str) -> qm.SparseVector:
        sv = next(self.bm25.embed([text]))
        return qm.SparseVector(indices=list(sv.indices), values=list(sv.values))

    def _native_hybrid(
        self,
        collection: str,
        q_dense: List[float],
        q_sparse: qm.SparseVector,
        limit: int,
        query_filter: Optional[qm.Filter] = None,
    ) -> List[Any]:
        # Qdrant 버전에 따라 Prefetch/FusionQuery가 없을 수 있음 -> 상위에서 try/except
        prefetch = [
            qm.Prefetch(query=q_dense, using="dense", limit=max(50, limit * 10), filter=query_filter),
            qm.Prefetch(query=q_sparse, using="bm25", limit=max(50, limit * 10), filter=query_filter),
        ]
        resp = self.client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return _extract_points_from_query_response(resp)

    def _fallback_hybrid(
        self,
        collection: str,
        q_dense: List[float],
        q_sparse: qm.SparseVector,
        limit: int,
        dense_weight: float = 0.3,
        sparse_weight: float = 0.7,
        query_filter: Optional[qm.Filter] = None,
    ) -> List[Any]:
        dense_resp = self.client.query_points(
            collection_name=collection,
            query=q_dense,
            using="dense",
            limit=max(100, limit * 20),
            with_payload=True,
            query_filter=query_filter,
        )
        sparse_resp = self.client.query_points(
            collection_name=collection,
            query=q_sparse,
            using="bm25",
            limit=max(100, limit * 20),
            with_payload=True,
            query_filter=query_filter,
        )

        dense_hits = _extract_points_from_query_response(dense_resp)
        sparse_hits = _extract_points_from_query_response(sparse_resp)

        def _minmax_norm(items: List[Any]) -> Dict[Any, float]:
            if not items:
                return {}
            scores = [float(getattr(x, "score", 0.0)) for x in items]
            mn, mx = min(scores), max(scores)
            if mx - mn < 1e-9:
                return {getattr(x, "id", id(x)): 0.0 for x in items}
            return {
                getattr(x, "id", id(x)): (float(getattr(x, "score", 0.0)) - mn) / (mx - mn)
                for x in items
            }

        dn = _minmax_norm(dense_hits)
        sn = _minmax_norm(sparse_hits)

        by_id: Dict[Any, Any] = {}
        for x in dense_hits + sparse_hits:
            by_id[getattr(x, "id", id(x))] = x

        merged: List[Tuple[float, Any]] = []
        for pid, x in by_id.items():
            # dense_weight와 sparse_weight가 실제로 전달되지 않고 기본값(0.3, 0.7)을 사용하고 있음
            # 이 부분을 수정해서 가중치가 제대로 적용되도록 해야 함
            s = dense_weight * dn.get(pid, 0.0) + sparse_weight * sn.get(pid, 0.0)
            merged.append((s, x))

        merged.sort(key=lambda t: t[0], reverse=True)
        return [x for _, x in merged[:limit]]

    def search_hybrid(
        self,
        collection: str,
        query: str,
        limit: int,
        dense_weight: float = 0.3,
        sparse_weight: float = 0.7,
        query_filter: Optional[qm.Filter] = None,
        force_fallback: bool = False,
    ) -> List[Any]:
        q_dense = self.dense_model.encode([query], normalize_embeddings=True)[0].tolist()
        q_sparse = self._bm25_sparse(query)

        # 1) native 먼저 시도
        try:
            return self._native_hybrid(
                collection=collection,
                q_dense=q_dense,
                q_sparse=q_sparse,
                limit=limit,
                query_filter=query_filter,
            )
        except Exception:
            # 2) fallback
            return self._fallback_hybrid(
                collection=collection,
                q_dense=q_dense,
                q_sparse=q_sparse,
                limit=limit,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                query_filter=query_filter,
            )


# -----------------------------
# 3) 평가 지표
# -----------------------------
def unique_case_numbers_from_hits(hits: List[Any]) -> List[str]:
    """
    hit들은 chunk 단위라 case_number 중복이 많음 -> case_number 기준으로 dedup.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for h in hits:
        pl = getattr(h, "payload", {}) or {}
        cn = (pl.get("case_number") or "").strip()
        if not cn:
            continue
        if cn in seen:
            continue
        seen.add(cn)
        out.append(cn)
    return out


def compute_hit_recall(gold_refs: List[str], retrieved_case_numbers: List[str]) -> Tuple[int, float]:
    gold_set = set([g.strip() for g in gold_refs if g.strip()])
    if not gold_set:
        # gold가 없으면 평가 불가 -> hit=0, recall=0 처리
        return 0, 0.0
    retrieved_set = set(retrieved_case_numbers)
    hit = 1 if len(gold_set & retrieved_set) > 0 else 0
    recall = len(gold_set & retrieved_set) / float(len(gold_set))
    return hit, recall


# -----------------------------
# 4) 메인 평가 루프
# -----------------------------
def eval_hybrid_retrieval(
    qna_path: str,
    qdrant_url: str,
    qdrant_api_key: Optional[str],
    collections: List[str],
    topks: List[int],
    out_csv_path: str,
    dense_weight: float = 0.3,
    sparse_weight: float = 0.7,
    dense_model_name: str = "dragonkue/BGE-m3-ko",
    bm25_model_name: str = "Qdrant/bm25",
):
    qnas = load_qna(qna_path)

    retriever = HybridRetriever(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        dense_model_name=dense_model_name,
        bm25_model_name=bm25_model_name,
    )

    rows: List[Dict[str, Any]] = []

    for col in collections:
        for k in topks:
            for item in tqdm(qnas, desc=f"eval {col} topk={k}"):
                query = (item.get("질문") or item.get("query") or "").strip()
                if not query:
                    continue
                gold_refs = parse_gold_refs(item)

                hits = retriever.search_hybrid(
                    collection=col,
                    query=query,
                    limit=k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    query_filter=None,
                    force_fallback=True, 
                )

                retrieved_case_numbers = unique_case_numbers_from_hits(hits)
                hit, recall = compute_hit_recall(gold_refs, retrieved_case_numbers)

                # 각 hit에서 원본 텍스트 추출하여 추가
                hit_texts = []
                for h in hits:
                    payload = getattr(h, "payload", {})
                    text = payload.get("text", "")
                    if text:
                        hit_texts.append(text)
                
                rows.append(
                    {
                        "collection": col,
                        "topk": k,
                        "query": query,
                        "gold_refs": "|".join(gold_refs),
                        "retrieved_case_numbers": "|".join(retrieved_case_numbers),
                        "hit": hit,
                        "recall": f"{recall:.6f}",
                        "retrieved_texts": "|".join(hit_texts),  # 원본 텍스트 추가
                    }
                )

    # 저장
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["collection", "topk", "query", "gold_refs", "retrieved_case_numbers", "hit", "recall", "retrieved_texts"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print_summary(rows, topks)
    print(f"\n✅ saved: {out_csv_path}")
    print(f"rows={len(rows)} | collections={len(collections)} | topks={topks}")
    print(f"weights(dense={dense_weight}, sparse={sparse_weight})")


def print_summary(rows: List[Dict[str, Any]], topks: List[int]) -> None:
    print("\n" + "=" * 100)
    print(f"[DONE] Saved CSV -> ./retrieval_eval_results.csv")
    print("=" * 100)

    # rows: collection/topk/hit/recall 문자열이 섞여있을 수 있으니 float 변환
    def f(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    # collection별로 묶기
    collections = sorted(set(r["collection"] for r in rows))

    for col in collections:
        print(f"\nCOLLECTION: {col}")
        for k in topks:
            subset = [r for r in rows if r["collection"] == col and int(r["topk"]) == int(k)]
            if not subset:
                continue
            n = len(subset)
            hit_avg = sum(int(r["hit"]) for r in subset) / n
            recall_avg = sum(f(r["recall"]) for r in subset) / n
            print(f"  - top{k}: Hit@{k}={hit_avg:.4f} | Recall@{k}={recall_avg:.4f} | n={n}")



if __name__ == "__main__":
    QNA_PATH = os.environ.get("QNA_JSON", "../data/qna_sample.json")
    QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
    QDRANT_KEY = os.environ.get("QDRANT_API_KEY")

    # ✅ 너가 만든 hybrid 컬렉션 3개 이름
    COLLECTIONS = [
        os.environ.get("COL_A", "cases_bge_m3_ko_A_summary_hybrid"),
        os.environ.get("COL_B", "cases_bge_m3_ko_B_summary_fulltext_hybrid"),
        os.environ.get("COL_C", "cases_bge_m3_ko_C_fieldwise_hybrid"),
    ]

    TOPKS = [3, 5, 10]
    OUT_CSV = os.environ.get("OUT_CSV", "./retrieval_eval_hybrid.csv")

    # fallback일 때만 weight 의미 있음( native RRF면 weight 무시될 수 있음 )
    DENSE_W = float(os.environ.get("DENSE_W", "0.3"))
    SPARSE_W = float(os.environ.get("SPARSE_W", "0.7"))

    eval_hybrid_retrieval(
        qna_path=QNA_PATH,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_KEY,
        collections=COLLECTIONS,
        topks=TOPKS,
        out_csv_path=OUT_CSV,
        dense_weight=DENSE_W,
        sparse_weight=SPARSE_W,
        dense_model_name=os.environ.get("DENSE_MODEL", "dragonkue/BGE-m3-ko"),
        bm25_model_name=os.environ.get("BM25_MODEL", "Qdrant/bm25"),
    )
