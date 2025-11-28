from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


def get_client(host: str, port: int) -> QdrantClient:
    return QdrantClient(host=host, port=port)


def ensure_or_recreate_collection(client: QdrantClient, collection: str, dim: int, recreate: bool):
    if recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        try:
            client.get_collection(collection_name=collection)
        except Exception:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    chunks: List[Dict],
    batch_size: int = 256,
):
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, show_progress_bar=True, batch_size=batch_size).tolist()

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "text": texts[i],
                "idx": i,
                "source_path": chunks[i]["source_path"],
                "source_name": chunks[i]["source_name"],
                "chunk_index": chunks[i]["chunk_index"],
            },
        )
        for i in range(len(texts))
    ]
    client.upsert(collection_name=collection, points=points)


def search(
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    query: str,
    topk: int = 3,
):
    """
    Qdrant에서 query_points / search 호출하고,
    항상 'List[ScoredPoint]' 형태로 반환하도록 정규화하는 래퍼.
    """
    qvec = embedder.encode([query])[0].tolist()

    # 1) 최신 Query API 우선 사용
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection,
            query=qvec,
            limit=topk,
            with_payload=True,
        )
    # 2) 구버전 fallback: search 메서드
    elif hasattr(client, "search"):
        res = client.search(
            collection_name=collection,
            query_vector=qvec,
            limit=topk,
            with_payload=True,
        )
    else:
        raise RuntimeError(
            "QdrantClient에 'query_points'나 'search' 메서드가 없습니다. "
            "qdrant-client 버전을 1.10.0 이상으로 업데이트 해주세요."
        )

    # 🔑 여기서 핵심: QueryResponse(points=[...]) → points 리스트만 꺼내기
    # query_points 결과: QueryResponse(points=[ScoredPoint, ...])
    # search 결과: 보통 list[ScoredPoint]
    if hasattr(res, "points"):
        return res.points  # List[ScoredPoint]
    else:
        return res         # 이미 리스트인 경우