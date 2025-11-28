# llm_vllm.py
from typing import List, Dict, Optional
import requests
from .config import LLMConfig


class VLLMClient:
    """
    vLLM OpenAI 호환 서버를 호출하는 클라이언트.
    - 엔드포인트: /v1/chat/completions
    - messages는 기존 Mi:dm chat_template 그대로 사용 가능
    """

    def __init__(self, cfg: LLMConfig):
        self.model = cfg.model_id                                # ex: "K-intelligence/Midm-2.0-Base-Instruct"
        self.api_base = cfg.vllm_api_base.rstrip("/")            # ex: "http://localhost:8000"
        self.default_max_new_tokens = cfg.max_new_tokens
        self.default_temperature = cfg.temperature
        self.default_do_sample = cfg.do_sample

    def chat(
        self,
        messages: List[Dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:

        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        do_sample = do_sample if do_sample is not None else self.default_do_sample

        # OpenAI 호환 payload 구성
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": float(temperature),
        }

        # deterministic mode (sampling off)
        if not do_sample:
            payload["temperature"] = 0.0

        resp = requests.post(
            f"{self.api_base}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"].strip()
