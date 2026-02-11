import os
import time
from typing import Optional, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()  # .env 파일 로드

# =========================
# 1. OpenAI API 키 설정
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =========================
# 2. FAISS Vectorstore 로드
# =========================

DB_DIR = os.getenv("FAISS_DB_DIR", "../faiss_index")

if not os.path.exists(DB_DIR):
    print(f"FAISS DB 폴더({DB_DIR})가 없습니다. 새로운 DB를 생성합니다.")

# 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# FAISS DB 불러오기 
try:
    vectorstore = FAISS.load_local(
        DB_DIR, 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print(f"FAISS DB 로드됨: {DB_DIR}")
except Exception as e:
    print(f"FAISS DB 로드 실패 ({e})")
    vectorstore = None
    retriever = None

# LLM 설정
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# =========================
# 3. RAG 함수 정의 (reranker top 5)
# =========================

def ask_question(query: str, profile: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    직접 구현한 RAG 함수 (LangChain chains X)
    """
    try:
        start_time = time.time()
        
        # 1. 관련 문서 검색
        if retriever:
            docs = retriever.invoke(query)
            context_docs = len(docs)
        else:
            docs = []
            context_docs = 0
        
        # 2. 검색된 문서들을 context로 변환
        context_parts = []
        for i, doc in enumerate(docs[:5]):  # 상위 5개만 사용
            content = doc.page_content[:800]  # 너무 길지 않게
            context_parts.append(f"[문서 {i+1}]\n{content}\n---")
        
        context = "\n".join(context_parts)
        
        # 3. 프로필 정보 포맷팅
        profile_info = ""
        if profile:
            gestational_week = profile.get("gestational_week", "알 수 없음")
            is_firstbirth = "초산" if profile.get("is_firstbirth", True) else "경산"
            risk_factors = profile.get("risk_factors", [])
            
            profile_info = f"""
임신 주수: {gestational_week}주차
출산 경험: {is_firstbirth}
건강 상태: {', '.join(risk_factors) if risk_factors else '양호'}
"""
        
        # 4. LLM 프롬프트 생성
        prompt = f"""
당신은 임신부 보험 전문 AI 어시스턴트입니다. 제공된 보험 약관 정보를 바탕으로 임신부의 상황에 맞는 보험 상품을 추천해주세요.

[보험 약관 정보]
{context}

[임신부 정보]
{profile_info}

[질문]
{query}

답변은 임신부의 상황을 정확히 반영하고, 약관 정보를 바탕으로 한 구체적인 추천을 제공하세요.
"""
        
        # 5. LLM 호출
        response = llm.invoke([{"role": "user", "content": prompt}])
        answer = response.content if hasattr(response, 'content') else str(response)
        
        processing_time = round(time.time() - start_time, 2)
        
        result = {
            "query": query,
            "answer": answer,
            "context_docs": context_docs,
            "processing_time": processing_time,
            "profile_used": profile_info.strip()
        }
        
        print("FAISS RAG 쿼리 완료")
        print(f"관련 문서: {result['context_docs']}개")
        print(f"처리 시간: {processing_time}초")
        
        return result
        
    except Exception as e:
        print(f"RAG 쿼리 실패: {e}")
        return {
            "query": query,
            "answer": f"죄송합니다. 질문을 처리하는 중 오류가 발생했습니다: {str(e)}",
            "error": str(e)
        }

def format_profile_info(profile: Dict[str, Any]) -> str:
    """프로필 정보 포맷팅"""
    if not profile:
        return "프로필 정보 없음"
    
    gestational_week = profile.get("gestational_week", "알 수 없음")
    is_firstbirth = "초산" if profile.get("is_firstbirth", True) else "경산"
    risk_factors = profile.get("risk_factors", [])
    
    info_parts = [
        f"임신 주수: {gestational_week}주차",
        f"출산 경험: {is_firstbirth}"
    ]
    
    if risk_factors:
        info_parts.append(f"건강 위험 요인: {', '.join(risk_factors)}")
    
    return " | ".join(info_parts)

def print_response(response: Dict[str, Any]):
    """응답 출력"""
    print("\n" + "="*50)
    print("AI 추천 답변")
    print("="*50)
    print(response.get("answer", "답변 없음"))
    print("\n" + "-"*30)
    print(f"참고한 문서 수: {response.get('context_docs', 0)}")
    print(f"처리 시간: {response.get('processing_time', 0)}초")
    print("="*50 + "\n")

def check_rag_system():
    """RAG 시스템 상태 확인"""
    checks = {
        "openai_api": bool(OPENAI_API_KEY),
        "faiss_db": os.path.exists(DB_DIR) and vectorstore is not None,
        "embedding_model": True,
        "llm_model": True,
        "retriever": retriever is not None
    }
    
    print("RAG 시스템 상태 체크:")
    for component, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}: {'정상' if status else '오류'}")
    
    all_ok = all(checks.values())
    print(f"\n 전체 상태: {'✅ 모든 컴포넌트 정상' if all_ok else '❌ 일부 컴포넌트 오류'}")
    
    return checks

if __name__ == "__main__":
    check_rag_system()
    
    # 테스트
    test_query = "임신 20주차 산모에게 어떤 보험이 적합할까요?"
    test_profile = {
        "gestational_week": 20,
        "is_firstbirth": True,
        "risk_factors": ["고혈압"]
    }
    
    print("\n 테스트 쿼리 실행:")
    response = ask_question(test_query, test_profile)
    if response:
        print_response(response)