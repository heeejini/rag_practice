from typing import Any, Dict, List


def build_rag_context(hits, max_chunks: int = 3, max_each: int = 2000) -> str:
    """
    Qdrant에서 가져온 hits를 바탕으로
    [문서 번호] + (제목/출처/요약) + 본문 일부를 하나의 문자열로 만든다.
    이 문자열은 그대로 프롬프트와 UI에 같이 사용된다.
    """
    parts: List[str] = []

    for rank, h in enumerate(hits[:max_chunks], 1):
        payload: Dict[str, Any] = getattr(h, "payload", {}) or {}
        meta: Dict[str, Any] = payload.get("metadata") or {}

        text = payload.get("text", "") or ""
        if not text.strip():
            continue

        # title / link / summary 는 top-level 또는 metadata 안에서 가져오도록 둘 다 지원
        title = payload.get("title") or meta.get("title")
        link = payload.get("link") or meta.get("link")

        lines: List[str] = []

        # 헤더
        lines.append(f"[문서 {rank}]")
        if title:
            lines.append(f"제목: {title}")
        if link:
            lines.append(f"출처: {link}")

        lines.append("")  # 빈 줄

        # 본문 청크
        body_snippet = text[:max_each].strip()
        lines.append("본문:")
        lines.append(body_snippet)

        parts.append("\n".join(lines))

    # 문서 사이 구분선
    return "\n\n---\n\n".join(parts)

def make_prompt_chat(query: str, rag: str | None = None, max_context_chars: int = 3000) -> List[Dict]:
    ref = (rag or "").strip()
    context_block = ""
    if ref:
        ref = ref[:max_context_chars].rstrip()
        context_block = (
            "아래는 질문 답변 시 도움이 될 수 있는 참고자료입니다.\n"
            "답변 생성 시 이를 참고하세요. \n\n"
            "[참고]\n" + ref + "\n\n"
        )

    system_prompt = (
        "너는 한국어로 간결하고 정확하게 답하는 assistant 입니다. \n "
        "불필요한 서론 없이 바로 핵심 답변 한 단락 정도 출력하세요."
    )

    user_prompt = (
        f"{context_block}"
        f"질문:\n{query.strip()}\n\n"
        "요구사항: 핵심만 간결하게 한 단락으로 답변하세요."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
