import os
import time
from typing import Optional, Dict, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# =========================
# 1. OpenAI API 키 설정
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# =========================
# 2. Chroma Vectorstore 로드
# =========================

DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

if not os.path.exists(DB_DIR):
    raise RuntimeError(f"저장된 DB 폴더({DB_DIR})가 없습니다. 먼저 chroma_db를 생성하세요.")

# 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# DB 불러오기
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model,
    collection_name="insurance_terms",
)

# 검색기 & LLM 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# =========================
# 3. 프롬프트 & 체인 정의
# =========================

base_system = """
[역할(role)]
너는 보험 약관 전문 분석가다.
너의 임무는 사용자의 질문에 대해 제공된 약관 문맥(context)만 근거로 정확하게 답하는 것이다.

[상황/문제 정의]
- 너에게는 약관의 일부가 {context}로 제공된다.
- 너는 이 문맥에서 근거를 찾아 답해야 한다.
- 사용자는 불필요한 중복 약관, 특약에 가입하고 싶지 않다.
- 불필요한 중복 약관과 특약은, C 특약이 A,B의 경우도 보장할 수 있는 경우, B특약, A특약, C특약을 모두 가입하는 것을 말한다.
  이럴 경우 사용자에겐 다른 것도 함께 포괄적으로 보장하는 C 특약, 약관을 추천할 수 있다.
- 불필요한 중복 약관을 찾아서 설명할 때는, 사용자가 직접 조항을 확인하고 판단할 수 있도록 실제 자료를 제시하되,
  도움말을 제공하는 것이 좋다.

[문맥 사용 규칙]
1) 모든 답변은 반드시 제공된 문맥에 근거해야 한다.
2) {context}에 없는 사항은 추측하지 말고 “문맥에 해당 내용이 없다/확실하지 않다”라고 명확히 말하라.
3) 답변에는 근거가 되는 조항/페이지를 포함하라. (metadata의 page 활용)
4) 문맥이 모호하거나 조항이 여러 개면, 가능한 해석을 나누어 설명하라.

[판단 기준]
- 정의 조항 > 보장 조항 > 제한 조항 > 면책 조항 > 특약 순으로 우선 참고한다.
- 질문이 “보장 여부/조건/금액/예외/용어 정의” 중 무엇인지 먼저 분류하고 그에 맞게 답하라.
- 보장 관련 질문이면 “보장 조건 → 지급 사유 → 지급 제외(면책) → 제한” 순서로 설명하라.

[출력 형식]
항상 아래 순서로 작성하라:
0. 질문 타입: 대괄호 안에 어떤 유형인지 표시.
1. 결론: 한 문장으로 답
2. 근거: 관련 조항/페이지와 핵심 문장 요지
   근거에는 다음을 포함:
   - 페이지(page): 숫자 , metadata에 page/조항명이 없으면 "페이지 정보 없음"이라고 표시
   - 조항 요지: 1~2문장
   - 원문 인용: 문맥에 있는 문장을 그대로 1문장 이내(가능하면)
3. 해석: 사용자가 이해하기 쉬운 말로 풀어 설명
4. 주의/예외: 면책, 제한, 특약 등 중요한 예외
5. 문맥 부족 시: 추가로 필요한 정보가 무엇인지
6. 원본: 근거로 든 조항의 전체를 원문 그대로 출력한다.

5번 문맥 부족 시에는 사용자 답변 출력에 '5.문맥 부족 시: ' 라고 적지 말고,
추가로 필요한 정보만 요청할 것. (예시: ~를 위해서는 ~에 대한 문서 정보가 추가로 필요합니다.)

이 형식을 지키고, 불확실한 부분은 반드시 불확실하다고 밝혀라.
"""

# 1) 요약/검색 프롬프트
summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            base_system
            + "\n\n[질문 유형: 요약/검색]"
            "\n- base_system의 출력 형식을 그대로 따르되,"
            "\n- 결론/근거/해석을 '요약 중심'으로 간결하게 작성하라.",
        ),
        ("human", "{input}"),
    ]
)

# 2) 추천 프롬프트
recommend_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            base_system
            + """
[질문 유형: 추천]
- base_system의 출력 형식을 그대로 따르라.
- 니즈/상황을 추출해 가장 적합한 보장/특약을 추천하라.
- 중복 가능성이 있으면 상위 특약 대안을 우선 제시하고 근거로 설명하라.
""",
        ),
        ("human", "{input}"),
    ]
)

summary_chain = create_retrieval_chain(
    retriever, create_stuff_documents_chain(llm, summary_prompt)
)

recommend_chain = create_retrieval_chain(
    retriever, create_stuff_documents_chain(llm, recommend_prompt)
)

# 라우터 LLM (질문 타입 분류용)
router_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
너는 사용자 질문을 두 가지 중 하나로 분류한다.
1) summary_search: 약관 내용 요약/해석/어디에 있는지 찾기
2) recommend: 사용자 상황에 맞는 보장/특약 추천

예시:
- “~해줘/추천해줘/가입해야 해?” 같이 선택을 요구하면 recommend
- “~뭐야/어디 나와/보장돼?” 같이 내용 확인이면 summary_search

다른 말 없이 딱 한 단어로만 출력: summary_search 또는 recommend
""",
        ),
        ("human", "{input}"),
    ]
)


def classify_query(query: str) -> str:
    """질문을 summary_search / recommend 두 타입 중 하나로 분류"""
    res = router_llm.invoke(router_prompt.format_messages(input=query))
    label = res.content.strip()
    if label not in ("summary_search", "recommend"):
        label = "summary_search"  # fallback
    return label


# =========================
# 4. 산모 프로필 텍스트 변환
# =========================

def build_profile_text(profile: Dict[str, Any]) -> str:
    """
    FastAPI/Spring에서 넘어온 산모 프로필 JSON(dict)를
    프롬프트에 삽입하기 좋은 한국어 설명 문자열로 변환
    """

    user = profile.get("user", {})
    preg = profile.get("pregnancyInfo", {})
    health = profile.get("healthStatus", {})

    # User
    name = user.get("name")
    email = user.get("email")
    user_id = user.get("user_id")

    # PregnancyInfo
    age = preg.get("age")
    height = preg.get("height")
    weight_pre = preg.get("weight_pre")
    weight_current = preg.get("weight_current")
    is_firstbirth = preg.get("is_firstbirth")
    gestational_week = preg.get("gestational_week")
    expected_date = preg.get("expected_date")
    is_multiple_pregnancy = preg.get("is_multiple_pregnancy")
    miscarriage_history = preg.get("miscarriage_history")

    # HealthStatus (JSON 문자열 그대로 사용)
    past_history = health.get("past_history_json")
    medicine = health.get("medicine_json")
    current_condition = health.get("current_condition")
    chronic_conditions = health.get("chronic_conditions_json")
    pregnancy_complications = health.get("pregnancy_complications_json")

    profile_text = f"""
[산모 기본 정보]
- 사용자 ID: {user_id}
- 이름: {name}
- 이메일: {email}
- 나이: {age}
- 키: {height}cm
- 임신 전 체중: {weight_pre}kg
- 현재 체중: {weight_current}kg
- 초산 여부: {"초산" if is_firstbirth else "경산" if is_firstbirth is not None else "정보 없음"}
- 다태아 여부: {"다태아" if is_multiple_pregnancy else "단태아" if is_multiple_pregnancy is not None else "정보 없음"}
- 임신 주차: {gestational_week}주차
- 출산 예정일: {expected_date}
- 유산 경험: {miscarriage_history}회

[과거 병력(past_history_json)]
{past_history}

[복용 중인 약물(medicine_json)]
{medicine}

[만성 질환(chronic_conditions_json)]
{chronic_conditions}

[임신 중 합병증/위험요인(pregnancy_complications_json)]
{pregnancy_complications}

[현재 증상]
{current_condition}
""".strip()

    return profile_text


# =========================
# 5. 메인 질의 함수
# =========================

def ask_question(query: str, profile: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    query: 사용자가 실제로 물어본 질문 (자연어)
    profile: 산모 프로필 JSON(dict). 있으면 recommend 시 함께 사용.
    """
    start = time.time()
    try:
        qtype = classify_query(query)

        # 1) 추천 타입 + 프로필이 있는 경우 → 산모 정보 + 질문 함께 사용
        if qtype == "recommend" and profile is not None:
            profile_text = build_profile_text(profile)
            full_input = f"""다음은 한 산모의 상태이다.

{profile_text}

위 산모의 상태를 고려하여, 다음 질문에 답하라:

{query}
"""
            response = recommend_chain.invoke({"input": full_input})

        # 2) 추천인데 프로필이 없는 경우 → 기존 query만 사용
        elif qtype == "recommend":
            response = recommend_chain.invoke({"input": query})

        # 3) 요약/검색 타입 → summary_chain 사용
        else:
            response = summary_chain.invoke({"input": query})

        pages = [doc.metadata.get("page", "?") for doc in response.get("context", [])]

    except Exception as e:
        print("에러 발생:", e)
        return None

    print(f"[type={qtype}] 소요 시간: {time.time() - start:.4f}초")
    response["pages"] = pages
    return response

import json

def print_response(response):
    """
    LangChain RAG 응답을 사람이 읽기 좋은 형태로 출력
    """

    print("\n==================== RAG 결과 요약 ====================\n")

    # 1) answer 정리
    answer = response.get("answer")
    if answer:
        print("[AI 최종 답변]\n")
        print(answer)
        print("\n")

    # 2) context 요약
    context = response.get("context", [])
    print("[참조된 약관 문맥]\n")
    for i, doc in enumerate(context, start=1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")

        print(f"--- 문서 #{i} (page={page}, source={source}) ---")
        print(doc.page_content[:300] + "...")
        print()

    # 3) pages
    pages = response.get("pages", [])
    print("[참조된 페이지 목록]:", pages)

    print("\n=======================================================\n")
