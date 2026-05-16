import json
import os
import uuid
import re
import unicodedata
from typing import Dict, Any, List, Optional
from datetime import datetime
# RAGAS 평가를 위한 모듈
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_router import get_active_llm, get_all_enabled_llms, get_llm_by_key
from ragas_evaluation import score_candidates

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def _first_existing_dir(candidates: List[Optional[str]]) -> Optional[str]:
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def _resolve_faiss_dir() -> str:
    env_dir = os.getenv("FAISS_DB_DIR")
    candidates = [
        env_dir,
        os.path.join(CURRENT_DIR, "faiss_index"),
        os.path.join(CURRENT_DIR, "..", "faiss_index"),
        "/app/faiss_index",
        "/faiss_index",
    ]
    return _first_existing_dir(candidates) or os.path.abspath(os.path.join(CURRENT_DIR, "faiss_index"))


def _resolve_llama_json_dir() -> str:
    env_dir = os.getenv("INSURANCE_JSON_DIR")
    candidates = [
        env_dir,
        os.path.join(CURRENT_DIR, "json", "Llama_json"),
        os.path.join(CURRENT_DIR, "..", "json", "Llama_json"),
        "/app/json/Llama_json",
        "/json/Llama_json",
    ]
    return _first_existing_dir(candidates) or os.path.abspath(os.path.join(CURRENT_DIR, "..", "json", "Llama_json"))


FAISS_DIR = _resolve_faiss_dir()
LLAMA_JSON_DIR = _resolve_llama_json_dir()

# FAISS 기반 RAG 및 LLM 임포트
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    from llm_router import get_active_llm
    from dotenv import load_dotenv
    load_dotenv()

    # 임베딩 모델 초기화
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


    index_file = os.path.join(FAISS_DIR, "index.faiss")
    # FAISS 벡터스토어 로드
    if os.path.exists(index_file):
        vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ 기존 FAISS 벡터스토어 로드 완료: {vectorstore.index.ntotal}개 문서 (dir={FAISS_DIR})")
    else:
        vectorstore = None
        print("FAISS 벡터스토어 생성 예정")

    llm = get_active_llm()
    RAG_AVAILABLE = True
    LLM_AVAILABLE = True

except Exception as e:
    print(f"RAG 시스템 초기화 실패: {e}")
    vectorstore = None
    embeddings = None
    llm = None
    RAG_AVAILABLE = False
    LLM_AVAILABLE = False

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
                print(f"{name} 로드 실패: {e}")


_load_data_maps()


# 오답 방지를 위한 보험사명 추출
INSURER_ALIASES = {
    "삼성생명": ["삼성생명"],
    "현대해상": ["현대해상"],
    "DB손해보험": ["DB손해보험", "동부화재", "프로미라이프"],
    "KB손해보험": ["KB손해보험", "KB손보", "KB"],
    "교보라이프플래닛생명": ["교보라이프플래닛생명", "교보라이프플래닛", "교보라플", "라이프플래닛"],
    "교보생명": ["교보생명"],
    "메리츠화재": ["메리츠화재"],
    "한화손해보험": ["한화손해보험"],
    "흥국화재": ["흥국화재"],
    "롯데손해보험": ["롯데손해보험"],
    "MG손해보험": ["MG손해보험"],
    "신한라이프생명": ["신한라이프생명", "신한라이프"],
    "삼성생명": ["삼성생명"],
    "동양생명": ["동양생명"],
    "메트라이프생명": ["메트라이프생명", "메트라이프"],
    "ABL생명": ["ABL생명"],
    "DB생명": ["DB생명"],
    "우체국": ["우체국", "우체국보험"],
}


def _norm_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("·", "ㆍ")
    s = re.sub(r"\s+", "", s)
    return s

def extract_insurer_name(text: Optional[str]) -> str:
    t = _norm_text(text)
    if not t:
        return ""

    for canonical, aliases in INSURER_ALIASES.items():
        for alias in aliases:
            if _norm_text(alias) in t:
                return canonical

    for insurer in PRICE_MAP.keys():
        if _norm_text(insurer) in t:
            return insurer
    for insurer in SUM_INSURED_MAP.keys():
        if _norm_text(insurer) in t:
            return insurer

    return ""


def looks_like_plan_name(s: str) -> bool:
    if not s:
        return False
    # 보험상품명에서 흔히 등장하는 키워드
    return any(k in s for k in ["무배당", "다이렉트", "해약환급금", "보험", "형", "보장"]) or len(s) >= 15


def looks_like_contract_name(s: str) -> bool:
    if not s:
        return False
    # 특약/담보명에서 흔히 등장하는 키워드
    return any(k in s for k in ["특약", "특별약관", "진단비", "실손", "위로금", "입원의료비"])


class InsuranceRecommender:
    def __init__(self):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = llm
        if RAG_AVAILABLE:
            self._load_insurance_data()

    def _load_insurance_data(self):
        """절대 경로를 사용하여 모든 JSON 데이터를 FAISS에 로드하기"""
        try:
            # 이미 로드되었다면 스킵하기
            if self.vectorstore and self.vectorstore.index.ntotal > 0:
                return

            documents = []
            data_dir = LLAMA_JSON_DIR

            if not os.path.exists(data_dir):
                print(f"데이터 디렉토리 없음: {data_dir}")
                return

            json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
            print(f"{len(json_files)}개 파일 분석 중...")

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
                os.makedirs(FAISS_DIR, exist_ok=True)
                self.vectorstore.save_local(FAISS_DIR)
                print(f"FAISS 생성 완료: {len(documents)}개 문서 (dir={FAISS_DIR})")
        except Exception as e:
            print(f"데이터 로드 오류 발생: {e}")

    def search_relevant_documents(self, query: str, n_results: int = 10):
        if not self.vectorstore:
            print("검색 불가능: 벡터스토어가 비어있음")
            return []
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=n_results)
            return [doc for doc, score in docs_with_scores]
        except Exception as e:
            print(f"FAISS 검색 실패: {e}")
            return []

    def generate_rag_recommendation(self, user_profile: Dict[str, Any], health_status: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 사용자 분석
            analysis = self._analyze_user_profile(user_profile, health_status)
            enabled_llms = get_all_enabled_llms()

            if not enabled_llms:
                print("LLM 사용 불가 -> Fallback")
                return self._fallback_recommendation(user_profile, health_status)

            # 검색 쿼리 생성 및 문서 검색
            search_query = self._build_rag_query(analysis)
            relevant_docs = self.search_relevant_documents(search_query, n_results=12)

            if not relevant_docs:
                print("검색된 문서 없음 -> Fallback")
                return self._fallback_recommendation(user_profile, health_status)

            # LLM 질문 생성 및 호출하기
            context = self._build_context_from_documents(relevant_docs)
            messages = self._build_recommendation_llm_messages(analysis, context)

            # RAGAS 평가를 위한 후보 생성 (openai, gemini)
            def _run_one(model_key: str, model_obj):
                print(f"[{model_key}] start")
                try:
                    answer = self._invoke_recommendation_llm(messages, llm_override=model_obj)
                    print(f"[{model_key}] answer_len={len(answer) if answer else 0}")
                    if not answer:
                        print(f"[{model_key}] drop: empty answer")
                        return None
                    
                    preview = answer[:200].replace("\n", " ")
                    print(f"[{model_key}] answer_preview={preview}")

                    parsed = self._parse_llm_response_to_recommendation(answer, analysis, relevant_docs)
                    if parsed is None:
                        print(f"[{model_key}] drop: parsed is None")
                        return None
                    
                    item_count = len(parsed.get("items", []))
                    print(f"[{model_key}] parsed_items={item_count}")
                    if item_count == 0:
                        keys = list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__
                        print(f"[{model_key}] drop: parsed empty items, parsed_keys={keys}")
                        return None
                    
                    print(f"[{model_key}] keep")
                    return { 
                        "model_key": model_key,
                        "answer_text": answer,
                        "parsed_result": parsed,
                    }
                except Exception as e:  
                    print(f"[{model_key}] fail: {type(e).__name__}: {e}")
                    return None 
    

            candidates = []
            with ThreadPoolExecutor(max_workers=len(enabled_llms)) as ex:
                futures = [ex.submit(_run_one, k, m) for k, m in enabled_llms]
                for f in as_completed(futures):
                    item = f.result()
                    if item:
                        candidates.append(item)
            print("[RAGAS] candidate_count =", len(candidates))
            print("[RAGAS] candidate_models =", [c["model_key"] for c in candidates])
            if not candidates:
                return self._fallback_recommendation(user_profile, health_status)
        
            contexts = [(d.page_content or "")[:2000] for d in relevant_docs[:6]]
            judge_llm = get_llm_by_key("openai")    # 평가용 고정 (openai)
            
            try:
                best, scored = score_candidates(
                    question=search_query,
                    contexts=contexts,
                    candidates=candidates,
                    judge_llm=judge_llm,
                )
                selected = best["parsed_result"]
                selected.setdefault("rag_metadata", {})
                selected["rag_metadata"].update({
                    "documents_used": len(relevant_docs),
                    "gestational_week": analysis["gestational_week"],
                    "selected_model": best["model_key"],
                    "faithfulness": best["faithfulness"],
                    "answer_relevancy": best["answer_relevancy"],
                    "total_score": best["total_score"],
                    "candidate_count": len(candidates),
                })
                return selected
            except Exception as e:
                print(f"RAGAS 평가 실패 -> 첫 후보 사용: {e}")
                first = candidates[0]["parsed_result"]
                first.setdefault("rag_metadata", {})
                first["rag_metadata"].update({
                    "documents_used": len(relevant_docs),
                    "gestational_week": analysis["gestational_week"],
                    "selected_model": candidates[0]["model_key"],
                    "ragas_failed": True,
                })
                return first

        except Exception as e:
            print(f"RAG 프로세스 실패: {e}")
            return self._fallback_recommendation(user_profile, health_status)

    def _build_rag_query(self, analysis: Dict[str, Any]) -> str:
        parts = ["임신 보험", "태아 보험", "산모 특약", "태아 보장"]
        week = analysis.get("gestational_week", 0)
        if week > 0:
            parts.append(f"{week}주")
            parts.append(f"임신 {week}주")
        if analysis.get("is_multiple_pregnancy"):
            parts.append("다태아 쌍둥이")
            parts.append("다태임신")
        if analysis.get("risk_factors"):
            parts.extend(analysis.get("risk_factors")[:2])
        parts.extend(["가입 가능", "보장", "특약"])
        return " ".join(parts)

    def _build_context_from_documents(self, documents) -> str:
        parts = ["<retrieved_documents>"]

        filtered_docs = []
        for doc in documents:
            md = doc.metadata or {}
            page = int(md.get("page_number", 0) or 0)
            section_title = md.get("section_title", "")
            content = doc.page_content or ""

            # 표지/인사말/가이드북/목차성 페이지 제외
            if page < 11:
                continue
            if any(k in section_title for k in ["CEO인사말", "약관 이용 가이드", "목차"]):
                continue
            # 200자 이내에 CEO인사말, 약관 이용 가이드 북 문장 제외
            if any(k in content[:200] for k in ["CEO인사말", "약관 이용 가이드 북"]):
                continue
            # 필터링된 문서 추가
            filtered_docs.append(doc)
        # 필터링된 문서가 있으면 필터링된 문서 사용, 없으면 원본 문서 사용
        target_docs = filtered_docs if filtered_docs else documents

        for i, doc in enumerate(target_docs[:6]): # 참고 문서 -> 상위 6개 문서
            md = doc.metadata or {}
            insurer = extract_insurer_name(
                f"{md.get('company', '')} "
                f"{md.get('insurance_company', '')} "
                f"{md.get('company_name', '')} "
                f"{md.get('insurer', '')} "
                f"{md.get('product_name', '')} "
                f"{md.get('source_file', '')}"
            )
            content = (doc.page_content or "").strip().replace("\n", " ")

            parts.append(
                f'<document id="DOC-{i+1}">\n'
                f"<insurance_company>{insurer or md.get('company', '?')}</insurance_company>\n"
                f"<product_name>{md.get('product_name', '?')}</product_name>\n"
                f"<section_title>{md.get('section_title', '?')}</section_title>\n"
                f"<clause_type>{md.get('clause_type', '?')}</clause_type>\n"
                f"<page_number>{md.get('page_number', '?')}</page_number>\n"
                f"<source_file>{md.get('source_file', '?')}</source_file>\n"
                f"<content>{content[:2500]}</content>\n" # 참고 문서마다 -> 상위 2500자만 추출
                f"</document>"
            )

        parts.append("</retrieved_documents>")
        return "\n".join(parts)
    
    def _build_recommendation_system_prompt(self) -> str:
        return """
            당신은 보험 RAG 시스템의 최종 추천을 하는 역할입니다. 하지만 당신은 보험 문서를 "생성"하는 모델이 아닙니다.
            <retrieved_context>에 존재하는 정보만 추출하고, 조합해야합니다. <retrieved_context>에 없는 보장, 특약, 조건을 절대 추가하지 마세요.
            따라서 모든 추천의 이유와 특약 설명은 반드시 <retrieved_context>의 실제 문장을 근거로 해야합니다.

            추천의 개수는 5~10개의 범위에서 가능한 많이 생성해야합니다. 

            당신의 목표는 "정확하면서도 누락이 적은 추천"입니다.
            특히 <retrieved_context>에 등장하는 상품을 과도하게 제외하지 마세요.

            당신의 유일한 근거는 user message 안의 <retrieved_context>에 포함된 <retrieved_documents>와 <available_products_catalog_json>입니다.
            여기서 retrieved context는 일반적인 대화 문맥이 아니라, 검색된 실제 보험 문서 조각입니다.

            [최우선 원칙]
            1. <retrieved_context>에 근거가 없는 내용은 절대 생성하지 마세요.
            2. 상품명과 보험사명은 제공된 상품 카탈로그의 값과 정확히 일치해야 합니다.
            3. <retrieved_context>에 근거가 부족하면 추천 수를 줄이거나 빈 배열을 반환하세요.
            4. 출력은 반드시 유효한 JSON 객체 하나만 반환하세요.
            5. reasoning을 장황하게 드러내지 말고, 최종 결과만 출력하세요.
            6. evidence에는 반드시 문서 id와 페이지 번호를 함께 넣으세요.
            예: "[p.18] 태아 관련 보장을 제공합니다."
            7. 같은 보험사를 반복해도 되지만, 정확성을 해치지 않는 범위에서 다양성을 우선하세요.
            8. special_contracts는 해당 상품 문맥에서 직접 확인 가능한 특약만 포함하세요.
            9. monthly_cost와 sum_insured는 모델이 추정하는 값이 아닙니다.
            이 두 필드는 서버가 prices.json, sum_insured.json으로 후처리하므로 반드시 null로 두세요.
            10. page_number는 반드시 같은 <document> 블록 안에 있는 <page_number> 값을 그대로 사용하세요.
            11. page_number를 추정하거나 다른 문서의 page_number와 섞지 마세요.
            12. evidence에는 반드시 [DOC-x][p.y] 형식을 포함하세요.
            - DOC-x는 해당 근거 문장이 들어 있는 <document id>와 같아야 합니다.
            - p.y는 같은 document 블록의 <page_number> 값과 같아야 합니다.
            """

    def _build_recommendation_llm_messages(self, analysis: Dict[str, Any], context: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self._build_recommendation_system_prompt()},
            {"role": "user", "content": self._build_llm_question(analysis, context)},
        ]

    def _build_llm_question(self, analysis: Dict[str, Any], context: str) -> str:
        available_products_map = {
            insurer: sorted(products.keys())
            for insurer, products in PRICE_MAP.items()
            if isinstance(products, dict) and products
        }
        products_catalog = json.dumps(
            available_products_map,
            ensure_ascii=False,
            indent=2,
        )
        risk_factors = analysis.get("risk_factors", [])
        risk_text = ", ".join(risk_factors) if risk_factors else "없음"
        multiple_text = "예" if analysis.get("is_multiple_pregnancy") else "아니오"

        return f"""
<task>
임신부 프로필과 <retrieved_context>를 기반으로 보험상품을 추천하세요.

가능한 많은 적합 상품을 추천하세요.
최소 5개 이상 추천해야 하며, 충분히 관련성이 있는 경우 개수 제한 없이 추천 가능합니다.
5개라는 개수는 최소이며, 더 추천 가능하다면 반드시 더 많은 개수의 추천을 생성하세요. 

<retrieved_context>에 등장하는 상품들은 우선적으로 추천 후보군에 포함하세요.

완전히 무관한 상품만 제외하세요.

추천 정확도도 중요하지만,
관련 상품을 과도하게 누락(false negative)하지 않는 것을 우선합니다.

빈 배열은 <retrieved_context>전체가 사용자 조건과 무관할 때만 허용됩니다.
</task>

<user_profile>
  <gestational_week>{analysis['gestational_week']}</gestational_week>
  <is_multiple_pregnancy>{multiple_text}</is_multiple_pregnancy>
  <miscarriage_history>{analysis.get('miscarriage_history', 0)}</miscarriage_history>
  <risk_factors>{risk_text}</risk_factors>
</user_profile>

<ranking_policy>
1. 현재 임신 주수에 적합한 상품을 우선합니다.
2. 위험요인과 직접 연결되는 특약 근거가 있는 상품을 우선합니다.
3. evidence, special_contracts, 추천 사유가 같은 문서 문맥에서 자연스럽게 이어지는 상품만 선택합니다.
4. 보험사 다양성보다 사용자 적합성과 문맥 관련성을 우선합니다. 관련성이 높다면 동일 보험사의 여러 상품도 허용됩니다.
</ranking_policy>

<hard_rules>
1. insurance_company는 아래 catalog JSON의 key 중 하나여야 합니다.
2. product_name은 해당 보험사의 배열에 있는 문자열과 완전히 같아야 합니다.
3. product_name에 특약명, 섹션명, 설명 문구를 섞지 마세요.
4. special_contracts는 1개~8개의 범위에서 가능한 많이 포함해야하고, 각 특약은 <retrieved_context>의 <content>에서 직접 확인 가능해야 합니다. 관련 특약이 제한적일 경우 가장 관련성 높은 특약을 포함해도 됩니다.
5. reason은 사용자 프로필과 상품 적합성을 연결한 2~3문장으로 작성하세요.
6. evidence는 짧은 직접 인용 1개와 문서 id, 페이지 번호를 포함하세요. [DOC-x][p.y] <retrieved_context>의 <content>에서 가져온 실제 인용문입니다.
7. monthly_cost와 sum_insured는 추천 판단 대상이 아니며, 반드시 null로 출력하세요.
8. 실제 monthly_cost와 sum_insured 값은 서버가 prices.json, sum_insured.json에서 채웁니다.
9. catalog에 없거나 context 근거가 약한 상품은 제외하세요.
10. JSON 객체 외의 다른 텍스트는 절대 출력하지 마세요.
11. JSON 문자열 안에서 백슬래시를 절대 사용하지 마세요.
12. 구분 표현이 필요하면 역슬래시 대신 슬래시(/) 또는 쉼표를 사용하세요.
13. evidence는 반드시 "[DOC-1][p.12] 인용문..." 형식으로 작성하세요.
</hard_rules>

<available_products_catalog_json>
{products_catalog}
</available_products_catalog_json>

<retrieved_context>
{context}
</retrieved_context>

<output_json_schema>
{{
  "recommendations": [
    {{
      "insurance_company": "보험사명 (위 목록에서)",
      "product_name": "보험상품명 (위 목록에서 정확히 선택)",
      "monthly_cost": null,
      "sum_insured": null,
      "reason": "주수와 위험요인을 고려한 구체적 추천 이유 2~3문장",
      "special_contracts": [
        {{
          "contract_name": "특약명1",
          "contract_description": "약관에서 확인되는 보장 내용을 2~3문장으로 요약",
          "contract_recommendation_reason": "이 특약이 현재 사용자에게 필요한 이유",
          "key_features": ["특징1", "특징2", "특징3"],
          "page_number": 12
        }}
      ],
      "evidence": "인용문... (page=숫자)"
    }}
  ]
}}
</output_json_schema>

<final_validation>
- insurance_company가 catalog key와 정확히 일치하는가?
- product_name이 해당 보험사의 배열 안에 실제로 존재하는가?
- monthly_cost와 sum_insured를 null로 두었는가?
- evidence가 [DOC-x][p.y] 형식을 포함하는가?
- special_contracts의 page_number가 같은 document의 <page_number>와 일치하는가?
- JSON만 출력했는가?
</final_validation>
"""

    # def _invoke_recommendation_llm(self, messages: List[Dict[str, str]]) -> str:
    #    response = self.llm.invoke(messages)
    #    return response.content if hasattr(response, "content") else str(response)
    def _invoke_recommendation_llm(self, messages: List[Dict[str, str]], llm_override=None) -> str:
        target_llm = llm_override or self.llm
        response = target_llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def _parse_llm_response_to_recommendation(self, llm_response: str, analysis: Dict[str, Any], relevant_docs) -> Dict[str, Any]:
        try:
            json_payload = self._extract_json_payload(llm_response)
            if not json_payload:
                return {"items": []}

            data = self._load_json_with_repair(json_payload)
            recs = data.get("recommendations", [])

            items = []
            for idx, rec in enumerate(recs[:8]):
                doc = self._find_best_matching_document(rec, relevant_docs)
                md = doc.metadata or {}

                comp = (rec.get("insurance_company") or "").strip()
                prod = (rec.get("product_name") or "").strip()

                # FAISS 문서의 메타데이터에서 product_name이 있으면 확인
                doc_product_name = md.get("product_name", "").strip()
                if doc_product_name and doc_product_name != "?":
                    # 문서에서 보험사명 추출
                    # 보험명 수정
                    doc_source_file = unicodedata.normalize("NFC", md.get("source_file", "").strip())
                    doc_insurer = extract_insurer_name(f"{doc_product_name} {doc_source_file}")
                    # 추천된 보험사와 문서의 보험사가 일치하는지 확인
                    if doc_insurer and (comp in doc_insurer or doc_insurer in comp or comp in doc_insurer):
                        # 일치하면 문서의 product_name과 보험사명 사용
                        prod = doc_product_name
                        comp = doc_insurer
                        print(f"FAISS 문서 사용: {comp} / {prod}")
                    else:
                        # 불일치하면 LLM 추천값 사용
                        print(f"보험사 불일치 - LLM: {comp}, 문서: {doc_insurer}")
                        print(f"LLM 추천값 유지: {comp} / {prod}")

                # 보험사명 정규화 (prices.json과 매칭 개선)
                comp = self._find_matching_insurer(comp)

                if looks_like_plan_name(comp) and looks_like_contract_name(prod):
                    plan_name = comp
                    comp = extract_insurer_name(plan_name) or comp
                    prod = plan_name

                #  테이블에서 먼저 조회, 없으면 LLM 값 사용
                #  테이블에서 보험료 조회
                monthly_cost, found_in_table = self._get_insurance_price(comp, prod)
                
                # 테이블에 값이 없으면 LLM 값 사용하기
                if not found_in_table:
                    llm_monthly_cost = rec.get("monthly_cost")
                    if llm_monthly_cost is not None:
                        if isinstance(llm_monthly_cost, str):
                            num_str = re.sub(r'[^\d]', '', llm_monthly_cost)
                            if num_str:
                                monthly_cost = int(num_str)
                                print(f"LLM 제공 보험료 사용 (테이블 없음): {monthly_cost}원 ({comp} / {prod})")
                        else:
                            monthly_cost = int(llm_monthly_cost)
                            print(f"LLM 제공 보험료 사용 (테이블 없음): {monthly_cost}원 ({comp} / {prod})")
                    else:
                        print(f"보험료 조회 실패 (테이블/LLM 모두 없음): {comp} / {prod}")
                else:
                    print(f"테이블에서 보험료 조회: {monthly_cost}원 ({comp} / {prod})")

                # 테이블에서 먼저 조회, 없으면 LLM 값 사용
                # 테이블에서 가입금액 조회
                sum_insured, found_in_table = self._get_sum_insured(comp, prod)
                
                # 테이블에 값이 없으면 LLM 값 사용하기
                if not found_in_table:
                    llm_sum_insured = rec.get("sum_insured")
                    if llm_sum_insured is not None:
                        if isinstance(llm_sum_insured, str):
                            sum_insured = self._parse_sum_insured_value(llm_sum_insured)
                            print(f"LLM 제공 가입금액 사용 (테이블 없음): {sum_insured} ({comp} / {prod})")
                    else:
                        print(f"가입금액 조회 실패 (테이블/LLM 모두 없음): {comp} / {prod}")
                else:
                    print(f"테이블에서 가입금액 조회: {sum_insured} ({comp} / {prod})")

                # 특약 정보 처리 (LLM이 상세 정보를 제공한 경우)
                special_contracts_data = rec.get("special_contracts", []) or []
                
                # 기존 형식(문자열 배열)과 새 형식(객체 배열) 모두 지원
                special_contracts_list = []
                for sc in special_contracts_data:
                    if isinstance(sc, dict):
                        # 새 형식: 상세 정보가 포함된 객체
                        special_contracts_list.append({
                            "contract_name": sc.get("contract_name", ""),
                            "contract_description": sc.get("contract_description", "약관 기반 맞춤 보장"),
                            "contract_recommendation_reason": sc.get("contract_recommendation_reason", f"{analysis['gestational_week']}주차 맞춤 특약"),
                            "key_features": sc.get("key_features", ["보장 범위 확인 완료"]),
                            "page_number": int(sc.get("page_number", md.get("page_number", 1)))
                        })
                    else:
                        # 기존 형식: 문자열만 제공된 경우
                        special_contracts_list.append({
                            "contract_name": str(sc),
                            "contract_description": "약관 기반 맞춤 보장",
                            "contract_recommendation_reason": f"{analysis['gestational_week']}주차 맞춤 특약",
                            "key_features": ["보장 범위 확인 완료"],
                            "page_number": int(md.get("page_number", 1))
                        })

                items.append({
                    "itemId": uuid.uuid4().hex[:8],
                    "insurance_company": comp,
                    "product_name": prod,
                    "is_long_term": True,
                    "sum_insured": str(sum_insured),
                    "monthly_cost": str(monthly_cost),
                    "insurance_recommendation_reason": rec.get("reason", ""),
                    "special_contracts": special_contracts_list,
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
            print(f"LLM 응답 파싱 실패: {e}")
            import traceback
            traceback.print_exc()
            return {"items": []}

    def _extract_json_payload(self, response_text: str) -> str:
        if not response_text:
            return ""
        cleaned = re.sub(r"```json\s*|```", "", response_text, flags=re.IGNORECASE).strip()
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _fix_json_string(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        # 코드블록 제거
        s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^```\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        # Python 스타일 값 보정
        s = s.replace("True", "true").replace("False", "false").replace("None", "null")
        # 잘못된 escape(\특, \수 등) 보정
        s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
        # trailing comma 제거
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s
    
    def _load_json_with_repair(self, json_payload: str) -> Dict[str, Any]:
        fixed = self._fix_json_string(json_payload)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            start = max(e.pos - 200, 0)
            end = min(e.pos + 200, len(fixed))
            print(f"JSON 파싱 실패 위치: line={e.lineno}, col={e.colno}, pos={e.pos}")
            print(f"JSON 문제 주변부: {fixed[start:end]}")
            # 여기서 억지로 전체 개행을 바꾸지 말고 실패를 올려서 fallback으로 넘김
            raise

    def _find_best_matching_document(self, rec: Dict[str, Any], relevant_docs):
        if not relevant_docs:
            return None

        rec_company = self._find_matching_insurer((rec.get("insurance_company") or "").strip())
        rec_product = self._normalize_product_name((rec.get("product_name") or "").strip())
        evidence = (rec.get("evidence") or "").strip()
        page_match = re.search(r"(?:page=|p\.?|페이지[:=]?)\s*(\d+)", evidence, re.IGNORECASE)
        evidence_page = int(page_match.group(1)) if page_match else None

        best_doc = relevant_docs[0]
        best_score = -1

        for doc in relevant_docs:
            md = doc.metadata or {}
            doc_product = self._normalize_product_name(md.get("product_name", "").strip())
            doc_source = unicodedata.normalize("NFC", md.get("source_file", "").strip())
            doc_company = self._find_matching_insurer(
                extract_insurer_name(
                    f"{md.get('company', '')} "
                    f"{md.get('insurance_company', '')} "
                    f"{md.get('company_name', '')} "
                    f"{md.get('insurer', '')} "
                    f"{md.get('product_name', '')} "
                    f"{doc_source}"
                )
            )
            doc_page = md.get("page_number")

            score = 0
            if rec_company and doc_company and (rec_company == doc_company):
                score += 4
            if rec_product and doc_product and (rec_product == doc_product):
                score += 5
            elif rec_product and doc_product and (rec_product in doc_product or doc_product in rec_product):
                score += 3

            if evidence_page is not None and doc_page is not None:
                try:
                    if int(doc_page) == evidence_page:
                        score += 3
                except Exception:
                    pass

            if score > best_score:
                best_score = score
                best_doc = doc

        return best_doc

    def _normalize_product_name(self, name: str) -> str:
        """보험 상품명 정규화 (유사 문자 통일)"""
        if not name:
            return ""
        # 유사한 문자들을 통일하기
        name = name.replace("·", "ㆍ")  # 중점 통일 (· → ㆍ)
        name = name.replace(" ", "")  # 공백 제거
        name = name.replace("(해약환급금 미지급형)", "(해약환급금 미지급형Ⅱ)")  # Ⅱ 추가
        name = name.replace("(해약환급금 미지급형II)", "(해약환급금 미지급형Ⅱ)")  # II → Ⅱ
        name = name.replace("(해약환급금 미지급형2)", "(해약환급금 미지급형Ⅱ)")  # 2 → Ⅱ
        return name.strip()

    def _normalize_insurer_name(self, name: str) -> str:
        """보험사명 정규화"""
        if not name:
            return ""
        return name.strip()
    
    def _find_matching_insurer(self, insurer_name: str) -> str:
        """prices.json에서 일치하는 보험사명 찾기"""
        normalized = self._normalize_insurer_name(insurer_name)
        
        # 정확한 매칭
        if normalized in PRICE_MAP:
            return normalized
        
        # 부분 매칭 (ex: "교보라이프플래닛" -> "교보라이프플래닛생명")
        for key in PRICE_MAP.keys():
            if normalized in key or key in normalized:
                return key
        
        return insurer_name  # 매칭 실패 시 원본 반환하기

    def _get_sum_insured(self, c, p):
        # 정규화된 상품명
        normalized_p = self._normalize_product_name(p)
        
        # 정확한 매칭 시도
        if c in SUM_INSURED_MAP:
            for product_name, value in SUM_INSURED_MAP[c].items():
                normalized_product = self._normalize_product_name(product_name)
                # 정규화된 이름이 같거나, 원본이 서로 포함되어 있으면 매칭
                if normalized_p == normalized_product or p in product_name or product_name in p:
                    if value != "재확인 필요":
                        return self._parse_sum_insured_value(value), True
        
        # 부분 매칭 시도 (보험사명)
        for insurer_name, products in SUM_INSURED_MAP.items():
            if c in insurer_name or insurer_name in c:
                # 상품명 부분 매칭
                for product_name, value in products.items():
                    normalized_product = self._normalize_product_name(product_name)
                    # 정규화된 이름 비교
                    if normalized_p == normalized_product or p in product_name or product_name in p:
                        if value != "재확인 필요":
                            return self._parse_sum_insured_value(value), True
        
        # 매칭 실패 시 기본값
        print(f"가입금액 매칭 실패: {c} / {p} (정규화: {normalized_p})")
        return "1,000만원", False

    def _get_insurance_price(self, c, p):
        """보험료 조회 (부분 매칭 지원) - (값, 찾았는지 여부) 튜플 반환"""
        # 정규화된 상품명
        normalized_p = self._normalize_product_name(p)
        
        # 정확한 매칭 시도
        if c in PRICE_MAP:
            for product_name, value in PRICE_MAP[c].items():
                normalized_product = self._normalize_product_name(product_name)
                # 정규화된 이름이 같거나, 원본이 서로 포함되어 있으면 매칭
                if normalized_p == normalized_product or p in product_name or product_name in p:
                    if value != "재확인 필요":
                        return self._parse_price_value(value), True
        
        # 부분 매칭 시도 (보험사명)
        for insurer_name, products in PRICE_MAP.items():
            if c in insurer_name or insurer_name in c:
                # 상품명 부분 매칭
                for product_name, value in products.items():
                    normalized_product = self._normalize_product_name(product_name)
                    # 정규화된 이름 비교
                    if normalized_p == normalized_product or p in product_name or product_name in p:
                        if value != "재확인 필요":
                            return self._parse_price_value(value), True
        
        # 매칭 실패 시 기본값
        print(f"보험료 매칭 실패: {c} / {p} (정규화: {normalized_p})")
        return 30000, False

    # 가입 금액 문자열로 유지하기
    def _parse_sum_insured_value(self, value):
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        # int인 경우 string으로 변환하기
        if isinstance(value, (int, float)):
            n = int(value)
            if n % 10000 == 0:
                return f"{n // 10000:,}만원"
            return f"{n:,}원"

        return "1,000만원"

    # 보험료 문자열로 유지하기
    def _parse_price_value(self, value):
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        # int인 경우 string으로 변환하기
        if isinstance(value, (int, float)):
            n = int(value)
            if n % 10000 == 0:
                return f"{n // 10000:,}만원"
            return f"{n:,}원"

        return "1,000원"

    def _fallback_recommendation(self, up, hs):
        return {"resultId": "fallback", "items": [], "rag_metadata": {"fallback": True}}

    def _analyze_user_profile(self, user_profile: Dict[str, Any], health_status: Dict[str, Any]) -> Dict[str, Any]:

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