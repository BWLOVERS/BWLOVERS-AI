import json
import os
import uuid
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

# FAISS 기반 RAG + LLM 관련 임포트
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    # rag_pipeline import (LLM + 프롬프트)
    from rag_pipeline import ask_question

    # 임베딩 모델 초기화
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # FAISS 벡터스토어 절대 경로 설정
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    faiss_path = os.path.join(CURRENT_DIR, "..", "faiss_index")

    # FAISS 벡터스토어 로드
    if os.path.exists(os.path.join(faiss_path, "index.faiss")):
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ 기존 FAISS 벡터스토어 로드됨: {vectorstore.index.ntotal}개 문서")
    else:
        vectorstore = None
        print("🆕 FAISS 벡터스토어 생성 예정")

    RAG_AVAILABLE = True
    LLM_AVAILABLE = True

except Exception as e:
    print(f"❌ RAG 시스템 초기화 실패: {e}")
    vectorstore = None
    embeddings = None
    RAG_AVAILABLE = False
    LLM_AVAILABLE = False
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 보험료 및 가입금액 테이블 로드
PRICE_MAP = {}
SUM_INSURED_MAP = {}
PRICE_FILE = os.path.join(CURRENT_DIR, "prices.json")
SUM_INSURED_FILE = os.path.join(CURRENT_DIR, "sum_insured.json")


def _load_data_maps():
    global PRICE_MAP, SUM_INSURED_MAP
    for file_path, target_map, name in [
        (PRICE_FILE, PRICE_MAP, "보험료"),
        (SUM_INSURED_FILE, SUM_INSURED_MAP, "가입금액"),
    ]:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    target_map.update(json.load(f))
                print(f"✅ {name} 테이블 로드 완료")
            except Exception as e:
                print(f"❌ {name} 로드 실패: {e}")


_load_data_maps()


# ---- 보험사명 추출(오답 방지용) ----
INSURER_NAMES = [
    "삼성화재",
    "현대해상",
    "DB손해보험",
    "KB손해보험",
    "메리츠화재",
    "한화손해보험",
    "흥국화재",
    "롯데손해보험",
    "MG손해보험",
]


def extract_insurer_name(text: Optional[str]) -> str:
    if not text:
        return ""
    for name in INSURER_NAMES:
        if name in text:
            return name
    return ""


def looks_like_plan_name(s: str) -> bool:
    if not s:
        return False
    # 보험상품명(플랜명)에서 흔히 보이는 키워드
    return any(k in s for k in ["무배당", "다이렉트", "해약환급금", "보험", "형", "보장"]) or len(s) >= 15


def looks_like_contract_name(s: str) -> bool:
    if not s:
        return False
    # 특약/담보명에서 흔히 보이는 키워드
    return any(k in s for k in ["특약", "특별약관", "진단비", "실손", "위로금", "입원의료비"])


class InsuranceRecommender:
    def __init__(self):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        if RAG_AVAILABLE:
            self._load_insurance_data()

    def _load_insurance_data(self):
        """절대 경로를 사용하여 모든 JSON 데이터를 FAISS에 로드"""
        try:
            # 이미 로드되었다면 스킵
            if self.vectorstore and self.vectorstore.index.ntotal > 0:
                return

            documents = []
            data_dir = os.path.join(CURRENT_DIR, "..", "json", "Llama_json")

            if not os.path.exists(data_dir):
                print(f"❌ 데이터 디렉토리 없음: {data_dir}")
                return

            json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
            print(f"📂 {len(json_files)}개 파일 분석 중...")

            for filename in json_files:
                filepath = os.path.join(data_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        content = item.get("content", "").strip()
                        if content and len(content) > 20:
                            doc = Document(
                                page_content=content,
                                metadata={**item.get("metadata", {}), "source_file": filename},
                            )
                            documents.append(doc)

            if documents:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                os.makedirs(faiss_path, exist_ok=True)
                self.vectorstore.save_local(faiss_path)
                print(f"✅ FAISS 생성 완료: {len(documents)}개 문서")
        except Exception as e:
            print(f"❌ 데이터 로드 치명적 오류: {e}")

    def search_relevant_documents(self, query: str, n_results: int = 10):
        if not self.vectorstore:
            print("⚠️ 검색 불가: 벡터스토어가 비어있음")
            return []
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=n_results)
            return [doc for doc, score in docs_with_scores]
        except Exception as e:
            print(f"❌ FAISS 검색 실패: {e}")
            return []

    def generate_rag_recommendation(self, user_profile: Dict[str, Any], health_status: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 1. 사용자 분석
            analysis = self._analyze_user_profile(user_profile, health_status)

            # 2. 검색 쿼리 생성 및 문서 검색
            search_query = self._build_rag_query(analysis)
            relevant_docs = self.search_relevant_documents(search_query, n_results=12)

            if not relevant_docs:
                print("⚠️ 검색된 문서 없음 -> Fallback 실행")
                return self._fallback_recommendation(user_profile, health_status)

            # 3. LLM 질문 생성 및 호출
            context = self._build_context_from_documents(relevant_docs)
            llm_question = self._build_llm_question(analysis, context)

            print(f"🤖 LLM 요청 중... (주수: {analysis['gestational_week']}주)")
            rag_result = ask_question(llm_question, profile=analysis)

            if rag_result and "answer" in rag_result:
                result = self._parse_llm_response_to_recommendation(rag_result["answer"], analysis, relevant_docs)
                if not result.get("items"):
                    return self._fallback_recommendation(user_profile, health_status)
                return result

            return self._fallback_recommendation(user_profile, health_status)
        except Exception as e:
            print(f"❌ RAG 프로세스 실패: {e}")
            return self._fallback_recommendation(user_profile, health_status)

    def _build_rag_query(self, analysis: Dict[str, Any]) -> str:
        parts = ["임신 보험", "태아 보장"]
        week = analysis.get("gestational_week", 0)
        if week > 0:
            parts.append(f"{week}주")
        if analysis.get("is_multiple_pregnancy"):
            parts.append("다태아 쌍둥이")
        if analysis.get("risk_factors"):
            parts.extend(analysis.get("risk_factors")[:2])
        return " ".join(parts)

    def _build_context_from_documents(self, documents) -> str:
        parts = []
        for i, doc in enumerate(documents[:8]):  # 8개로 제한
            md = doc.metadata or {}
            parts.append(
                f"[문서 {i+1}] 상품:{md.get('product_name','?')}, 페이지:{md.get('page_number','?')}\n"
                f"내용:{doc.page_content[:800]}"
            )
        return "\n\n".join(parts)

    # ✅ 프롬프트(출력 스키마) 스펙대로 수정함
    def _build_llm_question(self, analysis: Dict[str, Any], context: str) -> str:
        return f"""
역할: 보험 전문 언더라이터
임신부 정보: {analysis['gestational_week']}주차, 위험요인({analysis.get('risk_factors', [])}), 다태아({analysis['is_multiple_pregnancy']})

지침:
1. 제공된 [보험 약관 정보]만 근거로 가장 적합한 보험 상품 2-3개를 추천하라.
2. 반드시 JSON 형식으로만 답변하라. (설명 문장/코드블록/주석 금지)
3. evidence는 문맥에서 그대로 인용한 문장과 페이지를 포함하라.
4. 매우 중요:
   - insurance_company에는 "삼성화재"처럼 '보험사명'만 작성
   - product_name에는 "무배당 ... 보험(...)"처럼 '보험상품명'만 작성
   - 특약명은 special_contracts 배열에만 작성 (product_name에 특약명 쓰지 말 것)

[보험 약관 정보]
{context}

출력 형식(반드시 이 키로만):
{{
  "recommendations": [
    {{
      "insurance_company": "보험사명",
      "product_name": "보험상품명",
      "monthly_cost": 30000,
      "reason": "주수와 위험요인을 고려한 구체적 추천 이유",
      "special_contracts": ["특약명1", "특약명2"],
      "evidence": "인용문... (page=숫자)"
    }}
  ]
}}
"""

    def _parse_llm_response_to_recommendation(self, llm_response: str, analysis: Dict[str, Any], relevant_docs) -> Dict[str, Any]:
        try:
            json_block = re.search(r"(\{.*\})", llm_response, re.DOTALL)
            if not json_block:
                return {"items": []}

            data = json.loads(self._fix_json_string(json_block.group(1)))
            recs = data.get("recommendations", [])

            items = []
            for idx, rec in enumerate(recs[:3]):
                doc = relevant_docs[idx] if idx < len(relevant_docs) else relevant_docs[0]
                md = doc.metadata or {}

                # ✅ 스펙 키로 파싱
                comp = (rec.get("insurance_company") or "").strip()
                prod = (rec.get("product_name") or "").strip()

                # ✅ 방어: LLM이 또 뒤집어 쓰는 경우 교정
                # - comp가 플랜명처럼 보이고 prod가 특약처럼 보이면
                #   -> prod는 comp(플랜명)로, comp는 플랜명에서 보험사명만 추출
                if looks_like_plan_name(comp) and looks_like_contract_name(prod):
                    plan_name = comp
                    comp = extract_insurer_name(plan_name) or comp  # 최악의 경우 원본 유지
                    prod = plan_name

                # 가입금액 및 보험료 테이블 매칭(보험사+상품명 기준)
                sum_insured = self._get_sum_insured(comp, prod)
                monthly_cost = self._get_insurance_price(comp, prod)

                special_contracts = rec.get("special_contracts", []) or []
                # 혹시 LLM이 special_contracts를 문자열로 줄 수도 있어서 방어
                if isinstance(special_contracts, str):
                    special_contracts = [special_contracts]

                items.append({
                    "itemId": uuid.uuid4().hex[:8],
                    "insurance_company": comp,
                    "product_name": prod,
                    "is_long_term": True,
                    "sum_insured": int(sum_insured),
                    "monthly_cost": str(monthly_cost),
                    "insurance_recommendation_reason": rec.get("reason", ""),
                    "special_contracts": [
                        {
                            "contract_name": str(c),
                            "contract_description": "약관 기반 맞춤 보장",
                            "contract_recommendation_reason": f"{analysis['gestational_week']}주차 맞춤 특약",
                            "key_features": ["보장 범위 확인 완료"],
                            "page_number": int(md.get("page_number", 1))
                        } for c in special_contracts
                    ],
                    "evidence_sources": [
                        {
                            "page_number": int(md.get("page_number", 1)),
                            "text_snippet": rec.get("evidence", "")
                        }
                    ],
                })

            return {
                "resultId": uuid.uuid4().hex[:8],
                "items": items,
                "rag_metadata": {
                    "documents_used": len(relevant_docs),
                    "gestational_week": analysis["gestational_week"],
                },
            }
        except Exception as e:
            print(f"❌ LLM 응답 파싱 실패: {e}")
            return {"items": []}

    def _fix_json_string(self, s: str) -> str:
        s = s.replace("「", "'").replace("」", "'").replace("“", "'").replace("”", "'")
        return s.replace("True", "true").replace("False", "false").replace("None", "null")

    def _get_sum_insured(self, c, p):
        return SUM_INSURED_MAP.get(c, {}).get(p, 10000000)

    def _get_insurance_price(self, c, p):
        return PRICE_MAP.get(c, {}).get(p, 30000)

    def _fallback_recommendation(self, up, hs):
        return {"resultId": "fallback", "items": [], "rag_metadata": {"fallback": True}}

    def _analyze_user_profile(self, user_profile: Dict[str, Any], health_status: Dict[str, Any]) -> Dict[str, Any]:
        """Java의 다양한 필드명(Camel/Snake) 완벽 대응"""

        p_info = user_profile.get("pregnancyInfo") or user_profile

        gest_week = p_info.get("gestationalWeek") or p_info.get("gestational_week") or 0
        is_multiple = p_info.get("isMultiplePregnancy") or p_info.get("is_multiple_pregnancy") or False
        miscarriage = p_info.get("miscarriageHistory") or p_info.get("miscarriage_history") or 0

        analysis = {
            "gestational_week": int(gest_week),
            "is_multiple_pregnancy": bool(is_multiple),
            "miscarriage_history": int(miscarriage),
            "risk_factors": [],
        }

        comps = health_status.get("pregnancyComplications") or health_status.get("pregnancy_complications") or []
        for c in comps:
            c_type = c if isinstance(c, str) else (c.get("pregnancyComplicationType") or c.get("complication_type"))
            if c_type == "PREECLAMPSIA":
                analysis["risk_factors"].append("임신중독증")
            elif c_type == "PRETERM_RISK":
                analysis["risk_factors"].append("조산위험")

        return analysis


recommender = InsuranceRecommender()
