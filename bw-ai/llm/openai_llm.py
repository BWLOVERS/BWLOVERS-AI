import os
from langchain_openai import ChatOpenAI

def build_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=api_key)