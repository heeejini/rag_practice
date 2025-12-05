import copy
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)


def load_llm(model_id: str):

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",          
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

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

    generation_config = copy.deepcopy(base_generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = do_sample

    if generation_config.pad_token_id is None and tokenizer.eos_token_id is not None:
        generation_config.pad_token_id = tokenizer.eos_token_id
    if generation_config.eos_token_id is None and tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id

    # 추론 시에는 gradient 계산이 필요 없으므로 no_grad 로 감싸기
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=generation_config.max_new_tokens,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            do_sample=generation_config.do_sample,
        )

    # 프롬프트 길이 이후부터 LLM 이 생성한 답변 토큰만 잘라냄
    gen_tokens = out[0, input_ids.shape[1]:]

    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
