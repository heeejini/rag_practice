from typing import List
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


from .schemas import Chunk

def upsert_chunks(
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    chunks: List[Chunk],
    batch_size: int = 256,
):
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts, show_progress_bar=True, batch_size=batch_size).tolist()

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "text": texts[i],
                "idx": i,
                "source_path": chunks[i].source_path,
                "source_name": chunks[i].source_name,
                "chunk_index": chunks[i].chunk_index,
                "metadata": chunks[i].metadata,
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

    qvec = embedder.encode([query])[0].tolist()


    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection,
            query=qvec,
            limit=topk,
            with_payload=True,
        )
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

    if hasattr(res, "points"):
        return res.points  
    else:
        return res       