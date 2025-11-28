import copy
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)


# generationconfig 사용하는 것보다는 hf 에 있는 model config 직접 사용
# langchain llm wrapper 를 사용 x -> kt mi:dm 모델은 chat template 가 존재함
def load_llm(model_id: str):
    """
    8bit 양자화된 LLM + 토크나이저 + 기본 GenerationConfig 로드
    """
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",           # GPU 자동 분배
        quantization_config=bnb_config,
        # 8bit 에서는 torch_dtype 생략해도 되고, 필요하면 float16 정도로 설정
        # torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # hf hub 에 저장된 기본 generation 설정 (generation_config.json) 을 불러옴
    generation_config = GenerationConfig.from_pretrained(model_id)

    # pad_token_id 가 없는 모델이면 eos 토큰으로 맞춰줌 (generate 시 warning 방지)
    if generation_config.pad_token_id is None and tokenizer.eos_token_id is not None:
        generation_config.pad_token_id = tokenizer.eos_token_id

    if generation_config.eos_token_id is None and tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer, generation_config


def generate_answer(
    model,
    tokenizer,
    base_generation_config: GenerationConfig,
    messages: List[Dict],
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> str:
    """
    chat_template 를 사용하는 Mi:dm / chat 계열 LLM 에서 답변 한 번 생성하는 헬퍼 함수.
    - messages: [{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]
    """

    # chat template 을 이용해 프롬프트 → input_ids 텐서로 변환
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # device = next(model.parameters()).device 대신:
    # model.device 속성이 있으면 우선 사용하고, 없으면 cuda / cpu 중 하나 선택
    # model.parameters() => pytorch 의 모든 학습 가능한 파라미터들을 이터레이터 형태로 반환
    # next(model.parameters()) -> iterator 에서 첫 번째 파라미터 텐서 하나만 꺼내옴
    # .device => 꺼내온 텐서 device 정보를 읽음 , 결론 : 모델이 로드된 디바이스를 추정해서 input tensor 의 위치를 옮김
    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)

    # base_generation_config 를 직접 수정하지 않고, 매 호출마다 deepcopy 해서 사용
    generation_config = copy.deepcopy(base_generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = do_sample

    # pad / eos 토큰이 비어 있으면 tokenizer 기준으로 재설정
    if generation_config.pad_token_id is None and tokenizer.eos_token_id is not None:
        generation_config.pad_token_id = tokenizer.eos_token_id
    if generation_config.eos_token_id is None and tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id

    # 추론 시에는 gradient 계산이 필요 없으므로 no_grad 로 감싸기
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            # 아래 인자들은 generation_config 에도 들어가 있지만,
            # 명시적으로 넘겨주는 형태를 유지 (네 기존 스타일 존중)
            max_new_tokens=generation_config.max_new_tokens,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            do_sample=generation_config.do_sample,
        )

    # 프롬프트 길이 이후부터 LLM 이 생성한 답변 토큰만 잘라냄
    gen_tokens = out[0, input_ids.shape[1]:]

    # special token 은 스킵하고 문자열로 디코딩
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
