import os
from langchain_anthropic import ChatAnthropic
# pip install langchain-anthropic 설치 필요
def build_claude_llm(model: str = "claude-3-5-sonnet-20241022", temperature: float = 0):
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("CLAUDE_API_KEY가 설정되지 않았습니다.")
    return ChatAnthropic(model=model, temperature=temperature, claude_api_key=api_key)