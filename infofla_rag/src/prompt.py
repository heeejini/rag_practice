from typing import List, Dict

def build_rag_context(hits, max_chunks=3, max_each=2000) -> str:
    parts = []
    for h in hits[:max_chunks]: # qdrant 에서 뽑은 , query 와 관련된 top n 개 문서만 추출 
        t = h.payload.get("text", "")
        if t:
            parts.append(t[:max_each].strip())
    return "---\n".join(parts)

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
