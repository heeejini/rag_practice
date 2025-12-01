# ------------------------------------------------------------
# 1) Python 3.11 + uv 설치 베이스 이미지
# ------------------------------------------------------------
FROM python:3.11-slim

# 작업 경로
WORKDIR /app

# 필수 리눅스 패키지 설치
RUN apt-get update && apt-get install -y \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2) uv 설치
# ------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH 등록
ENV PATH="/root/.local/bin:${PATH}"

# uv 버전 표시(디버깅용)
RUN uv --version

# ------------------------------------------------------------
# 3) 프로젝트 파일 복사 + 패키지 설치
# ------------------------------------------------------------
COPY pyproject.toml uv.lock ./

# uv install —> pyproject + uv.lock 기준 패키지 설치
RUN uv sync

# 앱 전체 복사
COPY . .

# FastAPI 포트
EXPOSE 9000

# ------------------------------------------------------------
# 4) uv run으로 FastAPI 실행
# ------------------------------------------------------------
CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "9000"]
