# src/text_utils.py
import re


def normalize_paragraphs(raw_text: str) -> str:
    """PDF에서 잘못 분리된 줄바꿈을 고쳐 문장 단위로 합쳐주는 전처리."""
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3줄 이상 개행 → 2줄

    paragraphs = text.split("\n\n")
    normalized = []

    for p in paragraphs:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if not lines:
            continue
        normalized.append(" ".join(lines))  # 문단 안에서 줄바꿈 제거 → 공백으로 연결

    return "\n\n".join(normalized)


def split_text_to_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 500,
) -> list[str]:
    """긴 텍스트를 chunk_size / overlap 기준으로 잘라서 리스트로 반환"""
    chunks = []
    start = 0
    n = len(text)

    if n == 0:
        return []

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)

        if end == n:
            break
        # 다음 시작 위치: overlap만큼 겹치게
        start = end - overlap if end - overlap > 0 else end

    return chunks
