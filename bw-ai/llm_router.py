import os
from langchain_core.language_models.chat_models import BaseChatModel

from llm.openai_llm import build_openai_llm
from llm.gemini_llm import build_gemini_llm

# 베베슈어 사용 llm 목록
# 여기서 llm on/off 제어
REGISTERED_MODELS = {
    "gemini": build_gemini_llm,
    "openai": build_openai_llm,
}


def _enabled_llms() -> list[str]:
    raw = os.getenv("ENABLED_LLMS", "gemini")
    return [x.strip() for x in raw.split(",") if x.strip()]


def get_llm_by_key(model_key: str):
    key = model_key.strip()
    if key not in REGISTERED_MODELS:
        raise ValueError(f"알 수 없는 모델 키: {key}")
    return REGISTERED_MODELS[key]()


def get_active_llm() -> BaseChatModel:
    enabled = _enabled_llms()
    model_key = enabled[0]
    print(f"[LLM Router] 현재 사용된 모델: {model_key}")
    if model_key not in REGISTERED_MODELS:
        raise ValueError(f"알 수 없는 모델 키: {model_key}. 등록된 모델: {list(REGISTERED_MODELS)}")
    return REGISTERED_MODELS[model_key]()

# 멀티 LLM 동시 호출 
# MULTI_LLM_MODE=parallel 일 때만 호출할 예정
def get_all_enabled_llms() -> list[tuple[str, BaseChatModel]]:
    enabled = _enabled_llms()
    print(f"[LLM Router] ENABLED_LLMS={enabled}")
    result = []
    for key in enabled:
        if key in REGISTERED_MODELS:
            result.append((key, REGISTERED_MODELS[key]()))
    return result