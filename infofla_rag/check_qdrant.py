#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qdrant 컬렉션 상태 확인 스크립트"""

import sys

from qdrant_client import QdrantClient

# docker-compose의 qdrant는 포트 6335로 매핑되어 있지만, 
# 컨테이너 내부에서는 6333을 사용하므로 호스트에서는 6335를 사용
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6335  # docker-compose에서 매핑한 포트

def main():
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # 모든 컬렉션 목록 가져오기
        collections = client.get_collections().collections
        print(f"\n=== Qdrant 컬렉션 목록 (총 {len(collections)}개) ===\n")
        
        if not collections:
            print("❌ 컬렉션이 없습니다.")
            return
        
        for col in collections:
            collection_name = col.name
            try:
                # 컬렉션 정보 가져오기
                info = client.get_collection(collection_name)
                points_count = info.points_count
                vectors_count = info.vectors_count
                
                print(f"📦 컬렉션: {collection_name}")
                print(f"   - 포인트 개수: {points_count:,}")
                print(f"   - 벡터 개수: {vectors_count:,}")
                print(f"   - 벡터 차원: {info.config.params.vectors.size}")
                print(f"   - 거리 메트릭: {info.config.params.vectors.distance}")
                
                # 샘플 포인트 확인
                if points_count > 0:
                    try:
                        scroll_result = client.scroll(
                            collection_name=collection_name,
                            limit=1,
                            with_payload=True,
                            with_vectors=False
                        )
                        if scroll_result[0]:
                            sample = scroll_result[0][0]
                            print(f"   - 샘플 ID: {sample.id}")
                            if sample.payload:
                                text_preview = sample.payload.get("text", "")[:100]
                                print(f"   - 샘플 텍스트: {text_preview}...")
                    except Exception as e:
                        print(f"   - 샘플 조회 실패: {e}")
                else:
                    print("   ⚠️  데이터가 없습니다!")
                print()
                
            except Exception as e:
                print(f"❌ 컬렉션 '{collection_name}' 정보 조회 실패: {e}\n")
        
    except Exception as e:
        print(f"❌ Qdrant 연결 실패: {e}")
        print(f"   호스트: {QDRANT_HOST}:{QDRANT_PORT}")
        print("   컨테이너가 실행 중인지 확인하세요: docker ps | grep qdrant")
        sys.exit(1)

if __name__ == "__main__":
    main()

