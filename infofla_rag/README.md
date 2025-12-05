# InfoFla RAG API

InfoFla RAG API는 뉴스 기사를 기반으로 한 검색 증강 생성(Retrieval-Augmented Generation, RAG) 시스템입니다. FastAPI를 사용하여 REST API를 제공하며, Qdrant를 벡터 데이터베이스로, vLLM 또는 HuggingFace Transformers를 LLM 백엔드로 사용합니다.

## 주요 기능

- **RAG 파이프라인**: 문서 청킹, 임베딩, 검색, 답변 생성을 통합적으로 관리합니다.
- **유연한 LLM 지원**: 로컬 HuggingFace 모델 또는 vLLM API 서버를 선택하여 사용할 수 있습니다.
- **벡터 검색**: Qdrant를 사용하여 고속의 유사도 검색을 지원합니다.
- **관리자 기능**: 텍스트 파일 디렉토리에서 인덱스를 생성하거나 재생성하는 API를 제공합니다.
- **웹 인터페이스**: 간단한 테스트를 위한 웹 UI를 포함하고 있습니다.

## 기술 스택

- **Language**: Python 3.10+
- **Web Framework**: FastAPI
- **Vector DB**: Qdrant
- **Embedding**: SentenceTransformers (`Alibaba-NLP/gte-multilingual-base`)
- **LLM**: 
    - `K-intelligence/Midm-2.0-Base-Instruct` (기본값)
    - vLLM (OpenAI 호환 API)
- **Validation**: Pydantic

## 설치 및 실행

### 1. 환경 설정

필요한 패키지를 설치합니다. (가상환경 권장)

```bash
pip install -r requirements.txt
# 또는 uv 사용 시
uv sync
```

### 2. Qdrant 실행

Qdrant 벡터 데이터베이스가 실행 중이어야 합니다. Docker를 사용하는 것이 가장 간편합니다.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. API 서버 실행

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Docker Compose 실행

전체 스택(API + Qdrant + vLLM)을 Docker Compose로 실행할 수 있습니다.

```bash
docker-compose up -d
```

## API 사용법

Swagger UI (`http://localhost:8000/docs`)에서 상세한 API 명세를 확인할 수 있습니다.

### 주요 엔드포인트

- **`GET /`**: 웹 테스트 UI
- **`POST /chat`**: 웹 UI용 채팅 엔드포인트
- **`POST /rag`**: RAG 질의응답 API
    - Request Body:
      ```json
      {
        "query": "질문 내용",
        "topk": 3,
        "use_rag": true
      }
      ```
- **`POST /admin/build_index`**: 문서 인덱싱
    - Request Body:
      ```json
      {
        "src_dir": "/path/to/data",
        "pattern": "*.txt",
        "recreate": false
      }
      ```

## 프로젝트 구조

```
.
├── app/
│   ├── api.py           # FastAPI 애플리케이션 및 엔드포인트
│   ├── templates/       # 웹 UI 템플릿
│   └── static/          # 정적 파일
├── src/
│   ├── pipeline.py      # RAG 파이프라인 로직
│   ├── chunking.py      # 텍스트 청킹 로직
│   ├── qdrant.py        # Qdrant 클라이언트 및 검색 로직
│   ├── llm.py           # HuggingFace LLM 로직
│   ├── vLLM.py          # vLLM 클라이언트
│   ├── config.py        # 설정 클래스 (Pydantic)
│   ├── schemas.py       # 데이터 모델 (Pydantic)
│   ├── prompt.py        # 프롬프트 템플릿
│   └── logging_config.py # 로깅 설정
├── data/                # 데이터 디렉토리
├── Dockerfile           # API 서버 Dockerfile
├── docker-compose.yaml  # Docker Compose 설정
└── main.py              # (Optional) 실행 스크립트
```

## 설정

`src/config.py` 또는 환경 변수를 통해 설정을 변경할 수 있습니다.

- `QDRANT_HOST`, `QDRANT_PORT`: Qdrant 접속 정보
- `VLLM_API_BASE`: vLLM API 주소
- `INDEX_SRC_DIR`: 인덱싱할 데이터의 기본 경로
