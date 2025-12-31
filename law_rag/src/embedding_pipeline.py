#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# ✅ Sparse(BM25)
# pip install fastembed
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding


# =========================================================
# 0) Qdrant query_points 결과 호환 헬퍼
# =========================================================
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


# =========================================================
# 1) cases.json 로드
# =========================================================
def load_cases(cases_json_path: str) -> List[Dict[str, Any]]:
    with open(cases_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        for k in ("cases", "data", "items", "results"):
            if isinstance(obj.get(k), list):
                return [x for x in obj[k] if isinstance(x, dict)]
        return [obj]

    raise ValueError("cases.json 구조를 파악할 수 없다.")


# =========================================================
# 2) 청킹 + 임베딩 (Dense)
# =========================================================
def embed_cases(
    cases: List[Dict[str, Any]],
    model_name: str = "dragonkue/BGE-m3-ko",
    device: Optional[str] = None,  # "cuda" 또는 "cpu"
    chunk_strategy: str = "B",  # "A" | "B" | "C"
    max_fulltext_chars: int = 1200,
    fulltext_overlap_chars: int = 200,
    include_statute_chunk: bool = True,
    print_samples: int = 2,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    chunk_strategy:
      A: summary(판시사항+판결요지+참조조문)만(+statute optional) / 전문 X
      B: summary + fulltext(+statute optional) / 전문 O
      C: field-wise(판시사항 chunk + 판결요지 chunk + fulltext + statute) / 필드 분리
    """

    def _s(x: Any) -> str:
        return "" if x is None else str(x)

    def _norm_space(s: str) -> str:
        s = s.replace("\u00a0", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _shorten(s: str, limit: int) -> str:
        s = _norm_space(s)
        return s if len(s) <= limit else (s[:limit].rstrip() + "…")

    def to_case_serial_int(case_serial_any: Any) -> int:
        s = str(case_serial_any)
        if s.isdigit():
            return int(s)
        return abs(hash(s)) % (10**9)

    def build_summary_text(c: Dict[str, Any]) -> str:
        parts = []
        if _s(c.get("사건명")):
            parts.append(f"[사건명] {_s(c.get('사건명'))}")
        if _s(c.get("사건번호")):
            parts.append(f"[사건번호] {_s(c.get('사건번호'))}")
        if _s(c.get("법원명")):
            parts.append(f"[법원명] {_s(c.get('법원명'))}")
        if _s(c.get("선고일자")):
            parts.append(f"[선고일자] {_s(c.get('선고일자'))}")

        ps = _s(c.get("판시사항"))
        pj = _s(c.get("판결요지"))
        statutes = _s(c.get("참조조문"))

        if ps:
            parts.append(f"[판시사항]\n{ps}")
        if pj:
            parts.append(f"[판결요지]\n{pj}")
        if statutes:
            parts.append(f"[참조조문]\n{statutes}")

        return _norm_space("\n\n".join([p for p in parts if p.strip()]))

    def build_issue_text(c: Dict[str, Any]) -> str:
        parts = []
        if _s(c.get("사건명")):
            parts.append(f"[사건명] {_s(c.get('사건명'))}")
        if _s(c.get("사건번호")):
            parts.append(f"[사건번호] {_s(c.get('사건번호'))}")
        ps = _s(c.get("판시사항"))
        if ps:
            parts.append(f"[판시사항]\n{ps}")
        return _norm_space("\n\n".join([p for p in parts if p.strip()]))

    def build_holding_text(c: Dict[str, Any]) -> str:
        parts = []
        if _s(c.get("사건명")):
            parts.append(f"[사건명] {_s(c.get('사건명'))}")
        if _s(c.get("사건번호")):
            parts.append(f"[사건번호] {_s(c.get('사건번호'))}")
        pj = _s(c.get("판결요지"))
        if pj:
            parts.append(f"[판결요지]\n{pj}")
        return _norm_space("\n\n".join([p for p in parts if p.strip()]))

    def build_statute_text(c: Dict[str, Any]) -> str:
        statutes = _s(c.get("참조조문"))
        cites = _s(c.get("참조판례"))
        parts = []
        if statutes:
            parts.append(f"[참조조문]\n{statutes}")
        if cites:
            parts.append(f"[참조판례]\n{cites}")
        return _norm_space("\n\n".join(parts))

    def split_fulltext_into_chunks(c: Dict[str, Any], header_hint: str) -> List[Tuple[str, str]]:
        full = _norm_space(_s(c.get("전문")))
        if not full:
            return []

        headers = [(m.start(), m.group(1).strip()) for m in re.finditer(r"【([^】]+)】", full)]
        sections: List[Tuple[str, str]] = []

        if headers:
            for i, (pos, name) in enumerate(headers):
                end = headers[i + 1][0] if i + 1 < len(headers) else len(full)
                body = full[pos:end].strip()
                sections.append((name, body))
        else:
            sections.append(("전문", full))

        out: List[Tuple[str, str]] = []
        for sec_name, sec_text in sections:
            paras = [p.strip() for p in sec_text.split("\n\n") if p.strip()]
            for p in paras:
                p = _norm_space(p)
                if len(p) <= max_fulltext_chars:
                    out.append((sec_name, p))
                else:
                    start = 0
                    while start < len(p):
                        end = min(len(p), start + max_fulltext_chars)
                        chunk = p[start:end].strip()
                        out.append((sec_name, chunk))
                        if end == len(p):
                            break
                        start = max(0, end - fulltext_overlap_chars)

        return [(sec, _norm_space(header_hint + "\n\n" + body)) for sec, body in out]

    # ---- Dense 모델 로드
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    try:
        dim = model.get_sentence_embedding_dimension()
    except Exception:
        dim = len(model.encode(["test"], normalize_embeddings=True)[0])

    points: List[Dict[str, Any]] = []
    printed = 0

    for c in tqdm(cases, desc=f"Chunk+Embed strategy={chunk_strategy}"):
        case_serial = c.get("판례정보일련번호") or str(uuid.uuid4())
        case_serial_int = to_case_serial_int(case_serial)

        case_number = _s(c.get("사건번호"))
        case_name = _s(c.get("사건명"))
        court = _s(c.get("법원명"))
        decision_date = c.get("선고일자")

        base_meta = {
            "doc_type": "case",
            "case_serial": int(case_serial) if str(case_serial).isdigit() else str(case_serial),
            "case_number": case_number,
            "case_name": case_name,
            "court": court,
            "decision_date": int(decision_date) if str(decision_date).isdigit() else decision_date,
            "case_type": _s(c.get("사건종류명")),
            "judgment_type": _s(c.get("판결유형")),
            "chunk_strategy": chunk_strategy,
        }

        base_id = case_serial_int * 10_000
        seq = 0

        def next_point_id() -> int:
            nonlocal seq
            pid = base_id + seq
            seq += 1
            return pid

        header_hint = _norm_space(
            "\n".join(
                [
                    f"[판시사항(요약)] {_shorten(_s(c.get('판시사항')), 250)}" if _s(c.get("판시사항")) else "",
                    f"[판결요지(요약)] {_shorten(_s(c.get('판결요지')), 400)}" if _s(c.get("판결요지")) else "",
                ]
            )
        ).strip()

        created_for_this_case: List[Dict[str, Any]] = []

        if chunk_strategy in ("A", "B"):
            summary_text = build_summary_text(c)
            if summary_text:
                vec = model.encode([summary_text], normalize_embeddings=True)[0].tolist()
                p = {
                    "id": next_point_id(),
                    "dense_vector": vec,  # ✅ 이름 변경 (hybrid에서 사용)
                    "payload": {
                        **base_meta,
                        "chunk_type": "summary",
                        "chunk_id": 0,
                        "section": "summary",
                        "point_key": f"{case_serial}_summary_0",
                    },
                    "text": summary_text,
                }
                points.append(p)
                created_for_this_case.append(p)

        if chunk_strategy == "C":
            issue_text = build_issue_text(c)
            if issue_text:
                vec = model.encode([issue_text], normalize_embeddings=True)[0].tolist()
                p = {
                    "id": next_point_id(),
                    "dense_vector": vec,
                    "payload": {
                        **base_meta,
                        "chunk_type": "issue",
                        "chunk_id": 0,
                        "section": "issue",
                        "point_key": f"{case_serial}_issue_0",
                    },
                    "text": issue_text,
                }
                points.append(p)
                created_for_this_case.append(p)

            holding_text = build_holding_text(c)
            if holding_text:
                vec = model.encode([holding_text], normalize_embeddings=True)[0].tolist()
                p = {
                    "id": next_point_id(),
                    "dense_vector": vec,
                    "payload": {
                        **base_meta,
                        "chunk_type": "holding",
                        "chunk_id": 0,
                        "section": "holding",
                        "point_key": f"{case_serial}_holding_0",
                    },
                    "text": holding_text,
                }
                points.append(p)
                created_for_this_case.append(p)

        if chunk_strategy in ("B", "C"):
            full_chunks = split_fulltext_into_chunks(c, header_hint=header_hint)
            if full_chunks:
                texts = [t for _, t in full_chunks]
                vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
                for idx, ((sec, txt), v) in enumerate(zip(full_chunks, vecs)):
                    p = {
                        "id": next_point_id(),
                        "dense_vector": v.tolist(),
                        "payload": {
                            **base_meta,
                            "chunk_type": "fulltext",
                            "chunk_id": idx,
                            "section": sec,
                            "point_key": f"{case_serial}_fulltext_{idx}",
                        },
                        "text": txt,
                    }
                    points.append(p)
                    created_for_this_case.append(p)

        if include_statute_chunk:
            st = build_statute_text(c)
            if st:
                vec = model.encode([st], normalize_embeddings=True)[0].tolist()
                p = {
                    "id": next_point_id(),
                    "dense_vector": vec,
                    "payload": {
                        **base_meta,
                        "chunk_type": "statute",
                        "chunk_id": 0,
                        "section": "statute",
                        "point_key": f"{case_serial}_statute_0",
                    },
                    "text": st,
                }
                points.append(p)
                created_for_this_case.append(p)

        if printed < print_samples:
            printed += 1
            print("\n" + "=" * 90)
            print(f"[SAMPLE] strategy={chunk_strategy} | {case_number} | {case_name} | {court} | {decision_date}")
            print(f"  created_points_for_case: {len(created_for_this_case)}")
            for p in created_for_this_case[:3]:
                preview = _shorten(p["text"], 180).replace("\n", " ")
                print(
                    f"  - {p['payload']['chunk_type']} / section={p['payload']['section']} "
                    f"/ id={p['id']} / key={p['payload'].get('point_key')}"
                )
                print(f"    text_preview: {preview}")

    return points, dim


# =========================================================
# 3) Qdrant 저장 + 데모 검색 (Hybrid)
# =========================================================
def upsert_qdrant_hybrid(
    points: List[Dict[str, Any]],
    dim: int,
    collection_name: str,
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: Optional[str] = None,
    batch_size: int = 128,
    recreate: bool = False,
    demo_query: Optional[str] = None,
    demo_topk_total: int = 6,
    dense_weight: float = 0.5,   # ✅ fallback에서 사용
    sparse_weight: float = 0.5,  # ✅ fallback에서 사용
    model_name_for_query: str = "dragonkue/BGE-m3-ko",
    bm25_model_name: str = "Qdrant/bm25",
):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        # ✅ multi-vector (dense) + sparse(bm25)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": qm.SparseVectorParams(
                    index=qm.SparseIndexParams(on_disk=False)
                )
            },
        )

    # ✅ BM25 sparse embedder (CPU)
    bm25 = SparseTextEmbedding(bm25_model_name)

    def _bm25_sparse(text: str) -> qm.SparseVector:
        # fastembed는 generator를 반환
        sv = next(bm25.embed([text]))
        return qm.SparseVector(indices=list(sv.indices), values=list(sv.values))

    def to_point(p: Dict[str, Any]) -> qm.PointStruct:
        payload = dict(p["payload"])
        payload["text"] = p["text"]

        return qm.PointStruct(
            id=p["id"],
            vector={
                "dense": p["dense_vector"],
                "bm25": _bm25_sparse(p["text"]),
            },
            payload=payload,
        )

    for i in tqdm(range(0, len(points), batch_size), desc=f"Upsert(HYBRID) -> {collection_name}"):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=[to_point(p) for p in batch],
        )

    print(f"\n✅ Upsert done (HYBRID). collection={collection_name}, points={len(points)}")

    # ------------------------
    # demo: hybrid retrieval
    # ------------------------
    if demo_query:
        dense_model = SentenceTransformer(model_name_for_query, trust_remote_code=True)
        q_dense = dense_model.encode([demo_query], normalize_embeddings=True)[0].tolist()
        q_sparse = _bm25_sparse(demo_query)

        # 1) ✅ Qdrant native hybrid(fusion) 시도
        try:
            # 일부 버전에서 Prefetch/Fusion이 없을 수 있어서 try
            prefetch = [
                qm.Prefetch(query=q_dense, using="dense", limit=max(20, demo_topk_total * 5)),
                qm.Prefetch(query=q_sparse, using="bm25", limit=max(20, demo_topk_total * 5)),
            ]
            resp = client.query_points(
                collection_name=collection_name,
                prefetch=prefetch,
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),  # ✅ RRF fusion
                limit=demo_topk_total,
                with_payload=True,
            )
            results = _extract_points_from_query_response(resp)

            print("\n" + "=" * 90)
            print(f"[DEMO HYBRID - NATIVE] collection={collection_name}")
            print(f"query={demo_query}")
            for r in results:
                pl = getattr(r, "payload", {}) or {}
                score = getattr(r, "score", 0.0)
                print(f"- score={score:.4f} | {pl.get('chunk_type')} | {pl.get('case_number')} | {pl.get('case_name')}")
                txt = (pl.get("text") or "").strip().replace("\n", " ")
                print(f"  {txt[:180]}{'…' if len(txt) > 180 else ''}")

            return client

        except Exception:
            # 2) ✅ fallback: dense / sparse 따로 뽑고 가중합으로 합치기
            pass

        # -------- fallback hybrid --------
        dense_resp = client.query_points(
            collection_name=collection_name,
            query=q_dense,
            using="dense",
            limit=max(50, demo_topk_total * 10),
            with_payload=True,
        )
        sparse_resp = client.query_points(
            collection_name=collection_name,
            query=q_sparse,
            using="bm25",
            limit=max(50, demo_topk_total * 10),
            with_payload=True,
        )

        dense_hits = _extract_points_from_query_response(dense_resp)
        sparse_hits = _extract_points_from_query_response(sparse_resp)

        # score normalize(간단 min-max; 안전하게)
        def _minmax_norm(items: List[Any]) -> Dict[Any, float]:
            if not items:
                return {}
            scores = [float(getattr(x, "score", 0.0)) for x in items]
            mn, mx = min(scores), max(scores)
            if mx - mn < 1e-9:
                return {getattr(x, "id", i): 0.0 for i, x in enumerate(items)}
            return {getattr(x, "id", i): (float(getattr(x, "score", 0.0)) - mn) / (mx - mn) for i, x in enumerate(items)}

        dn = _minmax_norm(dense_hits)
        sn = _minmax_norm(sparse_hits)

        # merge by id
        by_id: Dict[Any, Any] = {}
        for x in dense_hits + sparse_hits:
            by_id[getattr(x, "id", id(x))] = x

        merged: List[Tuple[float, Any]] = []
        for pid, x in by_id.items():
            s = dense_weight * dn.get(pid, 0.0) + sparse_weight * sn.get(pid, 0.0)
            merged.append((s, x))

        merged.sort(key=lambda t: t[0], reverse=True)
        results = [x for _, x in merged[:demo_topk_total]]

        print("\n" + "=" * 90)
        print(f"[DEMO HYBRID - FALLBACK] collection={collection_name}")
        print(f"query={demo_query} | weights(dense={dense_weight}, sparse={sparse_weight})")
        for x in results:
            pl = getattr(x, "payload", {}) or {}
            # fallback score는 우리가 만든 s라서 여기선 원래 score 대신 대략 표시
            print(f"- {pl.get('chunk_type')} | {pl.get('case_number')} | {pl.get('case_name')}")
            txt = (pl.get("text") or "").strip().replace("\n", " ")
            print(f"  {txt[:180]}{'…' if len(txt) > 180 else ''}")

    return client


# =========================================================
# 4) 컬렉션 3개(A/B/C) 자동 생성 (HYBRID)
# =========================================================
def build_three_collections_hybrid(
    cases_path: str,
    qdrant_url: str,
    qdrant_api_key: Optional[str],
    model_name: str = "dragonkue/BGE-m3-ko",
    device: Optional[str] = None,
    recreate: bool = True,
):
    cases = load_cases(cases_path)

    experiments = [
        {"strategy": "A", "collection": "cases_bge_m3_ko_A_summary_hybrid"},
        {"strategy": "B", "collection": "cases_bge_m3_ko_B_summary_fulltext_hybrid"},
        {"strategy": "C", "collection": "cases_bge_m3_ko_C_fieldwise_hybrid"},
    ]

    for exp in experiments:
        print("\n" + "#" * 100)
        print(f"BUILD HYBRID COLLECTION: {exp['collection']} (strategy={exp['strategy']})")

        points, dim = embed_cases(
            cases=cases,
            model_name=model_name,
            device=device,
            chunk_strategy=exp["strategy"],
            include_statute_chunk=True,
            print_samples=2,
        )

        upsert_qdrant_hybrid(
            points=points,
            dim=dim,
            collection_name=exp["collection"],
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            recreate=recreate,
            demo_query="상무이사가 근로기준법상 근로자에 해당되는지.",
            demo_topk_total=6,
            dense_weight=0.5,
            sparse_weight=0.5,
            model_name_for_query=model_name,
            bm25_model_name="Qdrant/bm25",
        )


if __name__ == "__main__":
    cases_path = os.environ.get("CASES_JSON", "../data/cases.json")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    device = os.environ.get("EMB_DEVICE")  # "cuda" or "cpu"

    build_three_collections_hybrid(
        cases_path=cases_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_key,
        model_name="dragonkue/BGE-m3-ko",
        device=device,
        recreate=True,
    )
