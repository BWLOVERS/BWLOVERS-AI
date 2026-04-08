import os
from langchain_google_genai import ChatGoogleGenerativeAI
# pip install langchain-google-genai 설치 필요
def build_gemini_llm(model: str = "gemini-2.5-flash", temperature: float = 0):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, gemini_api_key=api_key)