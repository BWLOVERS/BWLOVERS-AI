import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

# from llm.openai_llm import build_openai_llm
# 배포 환경 안정화를 위해 openai provider는 잠시 비활성화함
from llm.gemini_llm import build_gemini_llm

# 베베슈어 사용 llm 목록
# 여기서 llm on/off 제어
REGISTERED_MODELS = {
    "gemini":  build_gemini_llm,
    #"openai":  build_openai_llm,
}

# llm 활성화 여부 확인 
# 기본값: openai 사용 -> gemini 사용해보기
# ENABLED_LLMS=openai,claude,gemini 처럼 .env 파일에서 켜기
ENABLED_LLMS: list[str] = os.getenv("ENABLED_LLMS", "gemini").split(",")

def get_active_llm() -> BaseChatModel:
    """
    현재 활성화된 단일 LLM을 반환합니다.
    ENABLED_LLMS 첫 번째 항목을 기본으로 사용합니다.
    """
    model_key = ENABLED_LLMS[0].strip()
    print(f"[LLM Router] 현재 사용된 모델: {model_key}")
    if model_key not in REGISTERED_MODELS:
        raise ValueError(f"알 수 없는 모델 키: {model_key}. 등록된 모델: {list(REGISTERED_MODELS)}")
    return REGISTERED_MODELS[model_key]()


# 멀티 LLM 동시 호출 -> 아직 사용/구현 전
# MULTI_LLM_MODE=parallel 일 때만 호출할 예정
def get_all_enabled_llms() -> list[tuple[str, BaseChatModel]]:
    """등록된 모든 활성 LLM 반환 (비교 평가 및 ragas 평가용)"""
    result = []
    for key in ENABLED_LLMS:
        key = key.strip()
        if key in REGISTERED_MODELS:
            result.append((key, REGISTERED_MODELS[key]()))
    return result
