import json
import os
import uuid
import re
from typing import Dict, Any, List
from datetime import datetime

# FAISS 기반 RAG + LLM
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
    
    # FAISS 벡터스토어 경로
    faiss_path = "../faiss_index"
    
    # FAISS 벡터스토어 로드 또는 생성
    if os.path.exists(f"{faiss_path}/index.faiss"):
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ 기존 FAISS 벡터스토어 로드됨: {vectorstore.index.ntotal}개 문서")
    else:
        vectorstore = None
        print("🆕 FAISS 벡터스토어 생성 예정")
    
    RAG_AVAILABLE = True
    LLM_AVAILABLE = True
    print("✅ FAISS + LLM 기반 RAG 시스템 활성화됨")
    
except ImportError as e:
    print(f"RAG 시스템 임포트 실패: {e}")
    vectorstore = None
    RAG_AVAILABLE = False
    LLM_AVAILABLE = False
    
except Exception as e:
    print(f"RAG 시스템 초기화 실패: {e}")
    vectorstore = None
    RAG_AVAILABLE = False
    LLM_AVAILABLE = False

# 파일 상단에 위치
PRICE_MAP = {}
# __file__의 절대 경로를 가져와서 경로 문제 방지
PRICE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.json")

SUM_INSURED_MAP = {}
SUM_INSURED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sum_insured.json")

def _load_price_map():
    """보험료 테이블 로드"""
    global PRICE_MAP
    if not os.path.exists(PRICE_FILE):
        print(f"보험료 파일 없음: {PRICE_FILE} (기본값 사용)")
        return

    try:
        with open(PRICE_FILE, "r", encoding="utf-8") as f:
            PRICE_MAP = json.load(f)
        print(f"✅ 보험료 테이블 로드됨: {len(PRICE_MAP)}개 보험사")
    except json.JSONDecodeError:
        print(f"보험료 파일 형식이 잘못되었습니다 (JSON 파싱 실패)")
    except Exception as e:
        print(f"보험료 테이블 로드 중 예상치 못한 오류 발생: {e}")


def _load_sum_insured_map():
    """보험가입금액 테이블 로드"""
    global SUM_INSURED_MAP
    if not os.path.exists(SUM_INSURED_FILE):
        print(f"가입금액 파일 없음: {SUM_INSURED_FILE} (기본값 사용)")
        return

    try:
        with open(SUM_INSURED_FILE, "r", encoding="utf-8") as f:
            SUM_INSURED_MAP = json.load(f)
        print(f"✅ 가입금액 테이블 로드됨: {len(SUM_INSURED_MAP)}개 보험사")
    except json.JSONDecodeError:
        print("가입금액 파일 형식이 잘못되었습니다 (JSON 파싱 실패)")
    except Exception as e:
        print(f"가입금액 테이블 로드 중 예상치 못한 오류 발생: {e}")

# 실행부
_load_price_map()
_load_price_map()
_load_sum_insured_map()

class InsuranceRecommender:
    """
    FAISS + LLM 기반 진짜 RAG 보험 추천 시스템
    """
    
    def __init__(self):
        if RAG_AVAILABLE:
            self.vectorstore = vectorstore
            self.embeddings = embeddings
            self._load_insurance_data()
        else:
            print("RAG 없이 기본 추천 모드로 작동")
    
    def _load_insurance_data(self):
        """모든 JSON 데이터를 FAISS에 로드"""
        try:
            if self.vectorstore and hasattr(self.vectorstore, 'index') and self.vectorstore.index.ntotal > 0:
                print(f"✅ FAISS에 이미 {self.vectorstore.index.ntotal}개 문서 존재")
                return
            
            documents = []
            data_dir = "../json/Llama_json"
            
            if os.path.exists(data_dir):
                json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
                for filename in json_files:
                    filepath = os.path.join(data_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if isinstance(data, list):
                            for item in data:
                                content = item.get('content', '').strip()
                                metadata = item.get('metadata', {})
                                if content and len(content) > 20:
                                    doc = Document(
                                        page_content=content,
                                        metadata={**metadata, 'source_file': filename, 'chunk_type': 'full_content'}
                                    )
                                    documents.append(doc)
                    except Exception as e:
                        print(f" {filename} 처리 실패: {e}")
            
            if documents:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                self.vectorstore.save_local(faiss_path)
                print(f"✅ FAISS 벡터스토어 생성 완료: {len(documents)}개 문서")
        except Exception as e:
            print(f"데이터 로드 실패: {e}")

    def search_relevant_documents(self, query: str, n_results: int = 10) -> List[Document]:
        if not RAG_AVAILABLE or not self.vectorstore:
            return []
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=n_results)
            return [doc for doc, score in docs_with_scores]
        except Exception as e:
            print(f"FAISS 검색 실패: {e}")
            return []

    def generate_rag_recommendation(self, user_profile: Dict[str, Any], health_status: Dict[str, Any]) -> Dict[str, Any]:
        if not LLM_AVAILABLE:
            return self._fallback_recommendation(user_profile, health_status)
        try:
            analysis = self._analyze_user_profile(user_profile, health_status)
            search_query = self._build_rag_query(analysis)
            relevant_docs = self.search_relevant_documents(search_query, n_results=15)
            
            if not relevant_docs:
                return self._fallback_recommendation(user_profile, health_status)
            
            context = self._build_context_from_documents(relevant_docs)
            llm_question = self._build_llm_question(analysis, context)
            rag_result = ask_question(llm_question, profile=analysis)
            
            if rag_result and 'answer' in rag_result:
                structured_recommendation = self._parse_llm_response_to_recommendation(
                    rag_result['answer'], analysis, relevant_docs
                )
                self._log_rag_performance(rag_result, analysis, relevant_docs)
                return structured_recommendation
            return self._fallback_recommendation(user_profile, health_status)
        except Exception as e:
            print(f"RAG 추천 생성 실패: {e}")
            return self._fallback_recommendation(user_profile, health_status)

    def _build_rag_query(self, analysis: Dict[str, Any]) -> str:
        parts = ["임신 보험", f"{analysis.get('gestational_week', 0)}주차"]

        if analysis.get("is_multiple_pregnancy"):
            parts.append("다태아")
        
        if (analysis.get("miscarriage_history") or 0) > 0:
            parts.append("유산력")
        
        if analysis.get("has_preeclampsia"):
            parts.extend(["임신중독증", "고혈압", "진단비"])
        
        if analysis.get("has_preterm_risk"):
            parts.extend(["조산", "미숙아", "NICU", "입원"])
        
        if analysis.get("has_diabetes"):
            parts.extend(["당뇨", "합병증"])

        rf = analysis.get("risk_factors") or []
        parts.extend(rf)
        return " ".join(parts)

    def _build_context_from_documents(self, documents: List[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(documents[:10]):
            md = doc.metadata or {}
            context_part = (
                f"[문서 {i+1}]\n"
                f"product_name: {md.get('product_name', '알 수 없음')}\n"
                f"page_number: {md.get('page_number', md.get('page', 'N/A'))}\n"
                f"source_file: {md.get('source_file', md.get('source', 'N/A'))}\n"
                f"content: {doc.page_content[:1000]}\n"
                f"---"
            )
            context_parts.append(context_part)
        return "\n".join(context_parts)

    def _build_llm_question(self, analysis: Dict[str, Any], context: str) -> str:
        return f"""
[절대 준수 사항]
- 답변은 반드시 유효한 JSON 포맷이어야 한다.
- 모든 문자열은 표준 큰따옴표(")로만 감싸야 한다.
- 한국어 특수 따옴표(「, 」, 『, 』)는 절대 사용하지 마라. 인용 시에도 표준 큰따옴표(")를 사용하라.

[역할]
너는 보험 약관 전문 분석가다. 제공된 문맥(context)만 근거로 답해야 한다. 반드시 JSON만 출력하라. (다른 문장, 설명, 코드블록 금지)
모든 값은 큰따옴표 " 만 사용한다. (작은따옴표/「」 금지)

[핵심 원칙]
- 문맥에 없는 내용은 추측하지 말고 “문맥에 없음”으로 명시한다.
- 근거는 반드시 문맥에서 1~2문장 그대로 인용하라.
- 근거에는 반드시 페이지 번호를 포함하라. (context에 page_number가 있다)

[정확도 강화 규칙]
- 결론에는 반드시 질문 키워드(예: 임신/당뇨/조산) + 보장항목/특약명을 함께 포함.
- 근거는 최소 2개 제시(가능하면 서로 다른 문장).
- 관련 조항이 없으면 “문맥에 없음”으로 명시하되, 유사/상위 범주는 제시하라.

[임신부 정보]
- 임신 주수: {analysis.get('gestational_week', 0)}주차
- 위험요인: {analysis.get('risk_factors') or []}
- 다태아: {analysis.get('is_multiple_pregnancy', False)}
- 유산력: {analysis.get('miscarriage_history', 0)}

[보험 약관 정보]
{context}

[출력 규칙]
- 반드시 JSON만 출력 (설명 문장/마크다운/코드블록 금지)
- evidence는 문맥에서 “그대로 인용한 문장”만 허용
- evidence 안에 반드시 (page=숫자) 포함
- special_contracts는 문자열 배열(1~3개)
- monthly_cost는 정수 (문맥에 없으면 합리적 추정, 추정임을 reason에 명시)

{{
  "recommendations": [
    {{
      "company": "보험사명(문맥에 없으면 '알 수 없음')",
      "product": "상품명(문맥에 없으면 '알 수 없음')",
      "monthly_cost": 10000,
      "reason": "추천 이유(키워드+보장/특약 연결, 문맥 근거 언급)",
      "special_contracts": ["특약1", "특약2"],
      "evidence": "문맥 인용문... (page=숫자)"
    }}
  ]
}}
""".strip()

    def _parse_llm_response_to_recommendation(
        self,
        llm_response: str,
        analysis: Dict[str, Any],
        relevant_docs: List[Document],
    ) -> Dict[str, Any]:
        try:
            json_block = re.search(r"(\{.*\})", llm_response, re.DOTALL)
            if not json_block:
                return self._fallback_recommendation(analysis, {})

            raw = json_block.group(1)
            fixed = self._fix_json_string(raw)

            try:
                llm_json = json.loads(fixed)
            except Exception:
                import ast
                fixed2 = fixed.replace("true", "True").replace("false", "False").replace("null", "None")
                llm_json = ast.literal_eval(fixed2)

            recs = llm_json.get("recommendations", [])
            if not isinstance(recs, list):
                return self._fallback_recommendation(analysis, {})

            items: List[Dict[str, Any]] = []
            for idx, rec in enumerate(recs[:3]):
                if not isinstance(rec, dict):
                    continue

                doc = relevant_docs[idx] if idx < len(relevant_docs) else (relevant_docs[0] if relevant_docs else None)
                md = (doc.metadata or {}) if doc else {}
                doc_page = md.get("page_number") or md.get("page") or 1
                doc_product = md.get("product_name") or "알 수 없음"

                company = rec.get("company", "알 수 없음")
                product = rec.get("product", "추천 상품")
                llm_cost = rec.get("monthly_cost", 0) or 0
                sum_insured = self._get_sum_insured(company, product)
                monthly_cost = self._get_insurance_price(company, product)
                if monthly_cost == 10000 and llm_cost > 0:
                    monthly_cost = llm_cost
                reason = rec.get("reason", "") or ""

                if company in ("보험사명", "알 수 없음", "", None):
                    company = self._extract_company_from_metadata(md) if doc else "알 수 없음"
                if product in ("상품명", "추천 상품", "알 수 없음", "", None):
                    product = doc_product

                contracts = rec.get("special_contracts", []) or []
                if not isinstance(contracts, list):
                    contracts = []

                special_contracts_out: List[Dict[str, Any]] = []
                for c in contracts[:3]:
                    name = c if isinstance(c, str) else str(c)
                    special_contracts_out.append({
                        "contract_name": name,
                        "contract_description": f"{name}에 대한 약관 기반 보장/조건 요약",
                        "contract_recommendation_reason": (
                            f"{analysis.get('gestational_week', 0)}주차 및 "
                            f"위험요인({', '.join(analysis.get('risk_factors', [])) or '없음'}) 기준 추천"
                        ),
                        "key_features": [
                            "약관 근거로 보장 범위/조건 확인",
                            "임신 주수 및 건강상태 기반 필요 보장 우선순위 반영",
                        ],
                        "page_number": int(doc_page),
                    })

                evidence = rec.get("evidence", "") or ""
                if not evidence and doc:
                    evidence = doc.page_content[:200]

                evidence_sources_out = [{
                    "page_number": int(doc_page),
                    "text_snippet": str(evidence)[:500],
                }]

                items.append({
                    "itemId": f"rag-{uuid.uuid4().hex[:8]}",
                    "insurance_company": company,
                    "product_name": product,
                    "is_long_term": True,
                    "sum_insured": int(sum_insured),
                    "monthly_cost": str(monthly_cost),
                    "insurance_recommendation_reason": reason,
                    "special_contracts": special_contracts_out,
                    "evidence_sources": evidence_sources_out,
                })

            return {
                "resultId": f"rag-{uuid.uuid4().hex[:8]}",
                "expiresInSec": 600,
                "items": items,
                "rag_metadata": {
                    "llm_response_quality": self._evaluate_response_quality(llm_response, analysis),
                    "documents_used": len(relevant_docs),
                    "gestational_week": analysis.get("gestational_week", 0),
                    "risk_factors": analysis.get("risk_factors", []),
                },
            }

        except Exception as e:
            print(f"파싱/구조화 실패: {e}")
            return self._fallback_recommendation(analysis, {})
        

    
    def _fix_json_string(self, json_str: str) -> str:
        if not json_str:
            return ""
        
        # 1. 한국어 특수 따옴표를 '작은따옴표'로 변환 (큰따옴표 중첩 방지 핵심!)
        json_str = json_str.replace("「", "'").replace("」", "'")
        json_str = json_str.replace("“", "'").replace("”", "'")
        json_str = json_str.replace("『", "'").replace("』", "'")
        json_str = json_str.replace("‘", "'").replace("’", "'")
        
        # 2. 값 내부의 줄바꿈 처리
        def replace_br(match):
            # 매칭된 값 내부의 엔터만 \n 문자로 바꿈
            return match.group(0).replace('\n', '\\n').replace('\r', '\\n')
        
        json_str = re.sub(r'":\s*"(.*?)"', replace_br, json_str, flags=re.DOTALL)

        # 3. 파이썬 스타일 불리언/None 변환
        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        # 4. 제어 문자 제거 (비정상적인 아스키 문자 제거)
        json_str = re.sub(r"[\x00-\x1F\x7F]", "", json_str)
        
        # 5. 불필요한 공백 정리 (선택사항)
        json_str = json_str.strip()
        
        return json_str


    # 회사명 고정해야 할 필요 O (아직 수정 전임)
    def _extract_company_from_metadata(self, md: Dict) -> str:
        # 메타데이터에서 회사명을 추출하는 로직
        source = md.get("source_file", "")
        if "현대" in source: return "현대해상"
        if "삼성" in source: return "삼성화재"
        if "DB" in source or "동부" in source: return "DB손해보험"
        return md.get("company", "알 수 없음")

    
    def _get_sum_insured(self, company: str, product: str) -> int:
        """
        보험사 + 상품명으로 가격 조회 (유사도 매칭 포함)
        """
        if not SUM_INSURED_MAP:
            return 10000  # 기본값
        
        if not company or not product:
            return 10000
        
        # 정확 매칭
        if company in SUM_INSURED_MAP:
            if product in SUM_INSURED_MAP[company]:
                return SUM_INSURED_MAP[company][product]
            
            # 유사도 매칭
            import difflib
            products = list(SUM_INSURED_MAP[company].keys())
            matches = difflib.get_close_matches(product, products, n=1, cutoff=0.8)
            
            if matches:
                matched_product = matches[0]
                print(f"🔍 가격 매칭: '{product}' → '{matched_product}'")
                return PRICE_MAP[company][matched_product]
        
        return 10000  # 기본값

    def _get_insurance_price(self, company: str, product: str) -> int:
        """
        보험사 + 상품명으로 가격 조회 (유사도 매칭 포함)
        """
        if not PRICE_MAP:
            return 10000  # 기본값
        
        if not company or not product:
            return 10000
        
        # 정확 매칭
        if company in PRICE_MAP:
            if product in PRICE_MAP[company]:
                return PRICE_MAP[company][product]
            
            # 유사도 매칭
            import difflib
            products = list(PRICE_MAP[company].keys())
            matches = difflib.get_close_matches(product, products, n=1, cutoff=0.8)
            
            if matches:
                matched_product = matches[0]
                print(f"🔍 가격 매칭: '{product}' → '{matched_product}'")
                return PRICE_MAP[company][matched_product]
        
        return 10000  # 기본값

    # RAGAS 답변 평가
    def _evaluate_response_quality(self, llm_response: str, analysis: Dict[str, Any]) -> float:
        score = 0.0
        if str(analysis.get("gestational_week", "")) in llm_response: score += 0.3
        if any(risk in llm_response for risk in analysis.get("risk_factors", [])): score += 0.4
        if "recommendations" in llm_response: score += 0.3
        return score

    def _log_rag_performance(self, rag_result: Dict, analysis: Dict, documents: List[Document]):
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "user_profile": analysis,
                "response_quality_score": self._evaluate_response_quality(rag_result.get('answer', ''), analysis)
            }
            os.makedirs("../logs", exist_ok=True)
            with open("../logs/rag_performance.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        except: pass

    def _fallback_recommendation(self, user_profile: Dict[str, Any], health_status: Dict[str, Any]) -> Dict[str, Any]:
        return {"resultId": "fallback", "items": [], "rag_metadata": {"fallback": True}}

    def _analyze_user_profile(self, user_profile: Dict[str, Any], health_status: Dict[str, Any]) -> Dict[str, Any]:
        pregnancy_info = user_profile.get("pregnancyInfo") or user_profile.get("pregnancy_info") or {}

        gest_week = (
            user_profile.get("gestationalWeek")
            or user_profile.get("gestational_week")
            or pregnancy_info.get("gestationalWeek")
            or pregnancy_info.get("gestational_week")
            or 0
        )

        is_firstbirth = (
            user_profile.get("isFirstbirth")
            if "isFirstbirth" in user_profile
            else user_profile.get("is_firstbirth", pregnancy_info.get("isFirstbirth", pregnancy_info.get("is_firstbirth", True)))
        )

        is_multiple = (
            user_profile.get("isMultiplePregnancy")
            if "isMultiplePregnancy" in user_profile
            else user_profile.get("is_multiple_pregnancy", pregnancy_info.get("isMultiplePregnancy", pregnancy_info.get("is_multiple_pregnancy", False)))
        )

        miscarriage = (
            user_profile.get("miscarriageHistory")
            or user_profile.get("miscarriage_history")
            or pregnancy_info.get("miscarriageHistory")
            or pregnancy_info.get("miscarriage_history")
            or 0
        )

        analysis = {
            "gestational_week": int(gest_week) if str(gest_week).isdigit() else (gest_week or 0),
            "is_firstbirth": bool(is_firstbirth),
            "is_multiple_pregnancy": bool(is_multiple),
            "miscarriage_history": int(miscarriage) if str(miscarriage).isdigit() else (miscarriage or 0),
            "has_preeclampsia": False,
            "has_preterm_risk": False,
            "has_diabetes": False,
            "has_hypertension": False,
            "risk_factors": [],
        }

        past = health_status.get("pastDiseases") or health_status.get("past_diseases") or []
        chronic = health_status.get("chronicDiseases") or health_status.get("chronic_diseases") or []
        comps = health_status.get("pregnancyComplications") or health_status.get("pregnancy_complications") or []

        for d in past:
            if isinstance(d, dict) and (d.get("pastDiseaseType") or d.get("past_disease_type")) == "HYPERTENSION":
                analysis["has_hypertension"] = True
                analysis["risk_factors"].append("고혈압")

        for d in chronic:
            if isinstance(d, dict) and (d.get("chronicDiseaseType") or d.get("chronic_disease_type")) == "DIABETES":
                analysis["has_diabetes"] = True
                analysis["risk_factors"].append("당뇨")

        for c in comps:
            c_type = None
            if isinstance(c, str):
                c_type = c
            elif isinstance(c, dict):
                c_type = (
                    c.get("pregnancyComplicationType") 
                    or c.get("complication_type") 
                    or c.get("pregnancy_complication_type")
                )
            
            if c_type == "PREECLAMPSIA":
                analysis["has_preeclampsia"] = True
                analysis["risk_factors"].append("임신중독증")
            elif c_type == "PRETERM_RISK":
                analysis["has_preterm_risk"] = True
                analysis["risk_factors"].append("조산위험")

        return analysis

# 인스턴스 생성 및 외부 호출 함수
recommender = InsuranceRecommender()

def generate_recommendations_db(request) -> Dict[str, Any]:
    user_profile = request.user_profile.model_dump()
    health_status = request.health_status.model_dump()
    return recommender.generate_rag_recommendation(user_profile, health_status)