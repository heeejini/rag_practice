#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer


# =========================================================
# 0) qdrant-client query_points 호환
# =========================================================
def extract_points_from_query_response(resp: Any) -> List[Any]:
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    if hasattr(resp, "points"):
        return list(resp.points)  # type: ignore
    if hasattr(resp, "result") and hasattr(resp.result, "points"):
        return list(resp.result.points)  # type: ignore
    return []


# =========================================================
# 1) qna_sample.json 로드 + 정답(참조 판례) 파싱
# =========================================================
def load_qna(qna_json_path: str) -> List[Dict[str, Any]]:
    with open(qna_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        for k in ("qna", "data", "items", "results"):
            if isinstance(obj.get(k), list):
                return [x for x in obj[k] if isinstance(x, dict)]
        return [obj]

    raise ValueError("qna_sample.json 구조를 파악할 수 없다.")


def parse_gold_refs(raw: Any) -> List[str]:
    """
    qna_sample.json의 "참조 판례" 값을 정규화해서 리스트로 만든다.
    예:
      "2004다29736"
      "88누6924, 2008도5984"
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []

    # 콤마/세미콜론/슬래시/줄바꿈 등 다 분리
    parts = re.split(r"[,;/\n]+", s)
    refs = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        refs.append(p)
    return refs


# =========================================================
# 2) 검색 결과에서 판례번호 추출
# =========================================================
def normalize_case_number(x: Any) -> str:
    """
    payload["case_number"] 혹은 기타 필드에서 들어온 값을
    비교 가능한 형태로 정규화.
    - 공백 제거
    - 너무 공격적인 정규화는 하지 않음(88누6924 같은 포맷 유지)
    """
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    return s


def extract_case_numbers_from_points(points: Sequence[Any]) -> List[str]:
    """
    query_points 결과 point들에서 payload.case_number를 뽑아온다.
    """
    out: List[str] = []
    for p in points:
        payload = getattr(p, "payload", None) or {}
        cn = payload.get("case_number")
        cn = normalize_case_number(cn)
        if cn:
            out.append(cn)
    return out


def uniq_preserve_order(xs: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# =========================================================
# 3) 평가 지표
# =========================================================
@dataclass
class Metrics:
    hit: int
    recall: float


def compute_hit_recall(retrieved_case_numbers: Sequence[str], gold_refs: Sequence[str]) -> Metrics:
    gold = [normalize_case_number(x) for x in gold_refs if normalize_case_number(x)]
    if not gold:
        # 정답 라벨이 없는 샘플이면 평가에서 제외하는 게 보통이지만,
        # 여기서는 recall을 0으로 두고 hit도 0 처리
        return Metrics(hit=0, recall=0.0)

    ret_set = set(retrieved_case_numbers)
    gold_set = set(gold)

    hit = 1 if len(ret_set.intersection(gold_set)) > 0 else 0
    recall = len(ret_set.intersection(gold_set)) / float(len(gold_set))
    return Metrics(hit=hit, recall=recall)


# =========================================================
# 4) Qdrant 검색 (query_points만 사용)
# =========================================================
def qdrant_retrieve_case_numbers(
    client: QdrantClient,
    collection_name: str,
    query_vec: List[float],
    topk: int,
    chunk_types: Optional[List[str]] = None,
    with_payload: bool = True,
) -> List[str]:
    """
    - chunk_types=None이면 컬렉션 전체에서 topk
    - chunk_types가 있으면, chunk_type별로 동일 topk를 뽑는 게 아니라
      "컬렉션 전체 topk"에서 평가하는 게 공정함.
      (전략마다 chunk 구성이 다르니까)
    """
    flt = None
    if chunk_types:
        flt = qm.Filter(
            should=[
                qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value=ct))
                for ct in chunk_types
            ],
            minimum_should_match=1,
        )

    resp = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=topk,
        with_payload=with_payload,
        query_filter=flt,
    )
    points = extract_points_from_query_response(resp)
    case_nums = extract_case_numbers_from_points(points)
    return uniq_preserve_order(case_nums)


# =========================================================
# 5) 전체 평가 루프
# =========================================================
def evaluate_retrieval(
    qna_json_path: str,
    qdrant_url: str,
    qdrant_api_key: Optional[str],
    model_name: str,
    collections: List[str],
    topk_list: List[int],
    output_csv_path: str,
    device: Optional[str] = None,
):
    qna = load_qna(qna_json_path)

    # 임베딩 모델
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    # Qdrant
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # 결과 저장
    rows: List[Dict[str, Any]] = []

    # 집계용: collection -> k -> list(metric)
    agg_hit: Dict[str, Dict[int, List[int]]] = {c: {k: [] for k in topk_list} for c in collections}
    agg_recall: Dict[str, Dict[int, List[float]]] = {c: {k: [] for k in topk_list} for c in collections}

    for item in tqdm(qna, desc="Evaluate retrieval"):
        query = str(item.get("질문", "")).strip()
        if not query:
            continue

        gold_refs = parse_gold_refs(item.get("참조 판례"))
        if not gold_refs:
            # 정답 라벨 없는 샘플은 스킵(평가 왜곡 방지)
            continue

        # 임베딩
        qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

        for col in collections:
            for k in topk_list:
                retrieved = qdrant_retrieve_case_numbers(
                    client=client,
                    collection_name=col,
                    query_vec=qvec,
                    topk=k,
                    chunk_types=None,  # 공정 비교 위해 None 추천
                )
                m = compute_hit_recall(retrieved, gold_refs)

                agg_hit[col][k].append(m.hit)
                agg_recall[col][k].append(m.recall)

                rows.append(
                    {
                        "collection": col,
                        "topk": k,
                        "query": query,
                        "gold_refs": "|".join(gold_refs),
                        "retrieved_case_numbers": "|".join(retrieved),
                        "hit": m.hit,
                        "recall": f"{m.recall:.6f}",
                    }
                )

    # CSV 저장
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "collection",
                "topk",
                "query",
                "gold_refs",
                "retrieved_case_numbers",
                "hit",
                "recall",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # 요약 출력
    print("\n" + "=" * 100)
    print(f"[DONE] Saved CSV -> {output_csv_path}")
    print("=" * 100)

    for col in collections:
        print(f"\nCOLLECTION: {col}")
        for k in topk_list:
            hits = agg_hit[col][k]
            recalls = agg_recall[col][k]

            n = len(hits)
            if n == 0:
                print(f"  - top{k}: (no labeled samples)")
                continue

            hit_rate = sum(hits) / float(n)
            recall_avg = sum(recalls) / float(n)

            print(f"  - top{k}: Hit@{k}={hit_rate:.4f} | Recall@{k}={recall_avg:.4f} | n={n}")

    print("\nTip) Hit@K가 가장 중요한 1차 지표고, Recall@K는 참조판례가 여러 개인 질문에서 유용하다.\n")


# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    qna_path = os.environ.get("QNA_JSON", "../data/qna_sample.json")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    device = os.environ.get("EMB_DEVICE")  # "cuda" or "cpu"

    # 네가 만든 3개 컬렉션 기본값
    collections = [
        os.environ.get("COL_A", "cases_bge_m3_ko_A_summary"),
        os.environ.get("COL_B", "cases_bge_m3_ko_B_summary_fulltext"),
        os.environ.get("COL_C", "cases_bge_m3_ko_C_fieldwise"),
    ]

    # topK는 3/5/10 추천
    topk_list = [3, 5, 10]

    out_csv = os.environ.get("EVAL_OUT", "./retrieval_eval_results.csv")

    evaluate_retrieval(
        qna_json_path=qna_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_key,
        model_name=os.environ.get("EMB_MODEL", "dragonkue/BGE-m3-ko"),
        collections=collections,
        topk_list=topk_list,
        output_csv_path=out_csv,
        device=device,
    )
