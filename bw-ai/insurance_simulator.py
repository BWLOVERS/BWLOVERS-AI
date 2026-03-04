import json
import os
import uuid
import re
from typing import Dict, Any, List, Optional

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
    return _first_existing_dir(candidates) or os.path.abspath(
        os.path.join(CURRENT_DIR, "faiss_index")
    )


FAISS_DIR = _resolve_faiss_dir()

try:
    from dotenv import load_dotenv
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    load_dotenv()

    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    index_file = os.path.join(FAISS_DIR, "index.faiss")
    if os.path.exists(index_file):
        vectorstore = FAISS.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )
        print(
            f"✅ 보험 시뮬레이션: FAISS 벡터스토어 로드됨: "
            f"{vectorstore.index.ntotal}개 문서 (dir={FAISS_DIR})"
        )
    else:
        vectorstore = None
        print(f"보험 시뮬레이션: FAISS 벡터스토어 없음 (dir={FAISS_DIR})")

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    RAG_AVAILABLE = True

except Exception as e:
    print(f"보험 시뮬레이션 RAG/LLM 시스템 초기화 실패: {e}")
    vectorstore = None
    embeddings = None
    llm = None
    RAG_AVAILABLE = False


class InsuranceSimulator:
    def __init__(self):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = llm

    # ──────────────────────────────────────────────
    # 공개 메서드
    # ──────────────────────────────────────────────

    def analyze_simulation(
        self,
        insurance_company: str,
        product_name: str,
        special_contracts: List[Dict[str, Any]],
        question: str,
    ) -> Dict[str, Any]:
        try:
            if not self.vectorstore:
                return {
                    "simulationId": uuid.uuid4().hex[:8],
                    "result": "보험 약관 데이터베이스를 사용할 수 없습니다.",
                }
            if not self.llm:
                return {
                    "simulationId": uuid.uuid4().hex[:8],
                    "result": "LLM을 사용할 수 없습니다. OPENAI_API_KEY를 확인해주세요.",
                }

            relevant_docs = self._search_simulation_documents(
                insurance_company=insurance_company,
                product_name=product_name,
                special_contracts=special_contracts,
                question=question,
            )

            if not relevant_docs:
                return {
                    "simulationId": uuid.uuid4().hex[:8],
                    "result": "관련 약관 정보를 찾을 수 없어 보장 여부를 확인할 수 없습니다.",
                }

            pages = sorted({
                int(d.metadata.get("page_number"))
                for d in relevant_docs
                if (d.metadata or {}).get("page_number") is not None
            })
            print(f"시뮬레이션 선택 문서 수: {len(relevant_docs)}")
            print(f"시뮬레이션 선택 문서 페이지: {pages}")

            context = self._build_simulation_context(relevant_docs, special_contracts)
            prompt = self._build_simulation_llm_question(
                insurance_company=insurance_company,
                product_name=product_name,
                special_contracts=special_contracts,
                question=question,
                context=context,
            )

            print(f"보험 시뮬레이션 LLM 요청 중... (보험사: {insurance_company})")
            result_text = self._invoke_simulation_llm(prompt)
            prased = self._parse_simulation_response(result_text)

            return {
                "simulationId": uuid.uuid4().hex[:8], 
                "result": {
                    **prased,
                    "rag_metadata": {"documents_used": len(relevant_docs)},
                },
            }

        except Exception as e:
            print(f"시뮬레이션 분석 실패: {e}")
            return {
                "simulationId": uuid.uuid4().hex[:8],
                "result": f"분석 중 오류가 발생했습니다: {str(e)}",
            }

    # ──────────────────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────────────────

    def _normalize(self, s: Optional[str]) -> str:
        if not s:
            return ""
        return re.sub(r"\s+", "", str(s)).lower()

    def _is_same_product(self, md: Dict[str, Any], insurance_company: str, product_name: str) -> bool:
        n_co = self._normalize(insurance_company)
        n_pr = self._normalize(product_name)
        co = self._normalize(md.get("company", "") or md.get("insurance_company", ""))
        pr = self._normalize(md.get("product_name", ""))
        sf = self._normalize(md.get("source_file", ""))
        product_ok = n_pr and (n_pr in pr or n_pr in sf)
        company_ok = (not n_co) or (n_co in co or n_co in pr or n_co in sf)
        return bool(product_ok and company_ok)

    def _contract_keywords(self, contract_name: str) -> List[str]:
        """특약명에서 의미있는 키워드만 추출."""
        stopwords = {
            "특약", "특별약관", "약관", "무배당", "갱신형", "진단", "진단비",
            "보장", "계약", "d", "k", "ii", "iii", "iv", "무", "배당"
        }
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", contract_name.lower())
        kws = [t for t in tokens if t not in stopwords and len(t) >= 2]
        return kws if kws else [self._normalize(contract_name)]

    # ──────────────────────────────────────────────
    # 문서 검색: 특약 중심, 핵심 페이지만
    # ──────────────────────────────────────────────

    def _search_simulation_documents(
        self,
        insurance_company: str,
        product_name: str,
        special_contracts: List[Dict[str, Any]],
        question: str,
    ) -> List:
        """
        전략:
        1) 특약명 키워드로 직접 검색 → 가장 정확한 조항 청크
        2) 요청 page_number 근처 ±50 페이지 검색
        3) 질문 내용으로 시맨틱 검색
        4) 모두 동일 상품 필터 후 중복 제거
        → 총 20~30개 이내로 제한 (LLM 집중도 유지)
        """
        if not self.vectorstore:
            return []

        try:
            pool: Dict[str, Any] = {}  # chunk_key -> (doc, priority)

            n_co = self._normalize(insurance_company)
            n_pr = self._normalize(product_name)

            def add_docs(docs_iter, priority: int):
                for doc in docs_iter:
                    md = doc.metadata or {}
                    if not self._is_same_product(md, insurance_company, product_name):
                        continue
                    key = md.get("chunk_id") or (
                        f"{md.get('page_number','?')}::{self._normalize(doc.page_content[:80])}"
                    )
                    # 우선순위 낮을수록 중요 (1이 가장 중요)
                    if key not in pool or pool[key][1] > priority:
                        pool[key] = (doc, priority)

            # ① 특약별 핵심 검색
            for sc in special_contracts:
                cn = sc.get("contract_name", "")
                pn = sc.get("page_number")
                kws = self._contract_keywords(cn)

                # 특약명 직접 쿼리 (가장 높은 우선순위)
                q1 = f"{cn} 지급사유 보험기간 면책 보장개시일 진단 확정"
                r1 = self.vectorstore.similarity_search(q1, k=50)
                add_docs(r1, priority=1)

                # 키워드 하나씩 쿼리
                for kw in kws[:3]:
                    q2 = f"{insurance_company} {product_name} {kw} 약관 조항"
                    r2 = self.vectorstore.similarity_search(q2, k=30)
                    add_docs(r2, priority=2)

                # 요청 page_number 기반: 같은 상품 전체에서 페이지 범위 필터
                if pn:
                    try:
                        req_page = int(pn)
                        # 시맨틱 검색 풀에서 페이지 범위가 맞는 것 찾기
                        q3 = f"{cn} 약관"
                        r3 = self.vectorstore.similarity_search(q3, k=120)
                        for doc in r3:
                            md = doc.metadata or {}
                            if not self._is_same_product(md, insurance_company, product_name):
                                continue
                            dp = md.get("page_number")
                            if dp is not None and abs(int(dp) - req_page) <= 50:
                                key = md.get("chunk_id") or (
                                    f"{dp}::{self._normalize(doc.page_content[:80])}"
                                )
                                if key not in pool or pool[key][1] > 3:
                                    pool[key] = (doc, 3)
                    except Exception:
                        pass

            # ② 질문 내용으로 시맨틱 검색
            q4 = f"{insurance_company} {product_name} {question}"
            r4 = self.vectorstore.similarity_search(q4, k=60)
            add_docs(r4, priority=4)

            if not pool:
                return []

            # ③ 우선순위 낮은 순으로 정렬 후 상위 25개
            sorted_docs = sorted(pool.values(), key=lambda x: x[1])
            top_docs = [doc for doc, _ in sorted_docs[:25]]

            # ④ 페이지 기준 정렬 (읽기 순서 유지)
            top_docs.sort(key=lambda d: int((d.metadata or {}).get("page_number", 10**9)))

            return top_docs

        except Exception as e:
            print(f"시뮬레이션 문서 검색 실패: {e}")
            return []

    # ──────────────────────────────────────────────
    # 컨텍스트 빌드
    # ──────────────────────────────────────────────

    def _build_simulation_context(
        self,
        documents: List,
        special_contracts: List[Dict[str, Any]],
    ) -> str:
        parts = []

        if special_contracts:
            parts.append("=== 가입 특약 정보 ===")
            for sc in special_contracts:
                parts.append(
                    f"- {sc.get('contract_name', '')} "
                    f"(요청 페이지: {sc.get('page_number', '?')})"
                )
            parts.append("")

        parts.append("=== 보험 약관 정보 ===")
        for i, doc in enumerate(documents, 1):
            md = doc.metadata or {}
            pn = md.get("page_number", "?")
            title = (md.get("section_title") or "")[:80]
            content = (doc.page_content or "").strip()
            parts.append(
                f"\n[약관 {i}] p.{pn} | {title}\n{content}\n---"
            )

        return "\n".join(parts)
    
    def _parse_simulation_response(self, raw_text: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 블록 추출 및 파싱"""
        try:
            json_block = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if not json_block:
                return {"raw": raw_text}

            fixed = json_block.group(1)
            fixed = fixed.replace("「", "'").replace("」", "'")
            fixed = fixed.replace("\u201c", "'").replace("\u201d", "'")
            fixed = fixed.replace("True", "true").replace("False", "false").replace("None", "null")
            return json.loads(fixed)
        except Exception as e:
            print(f"시뮬레이션 응답 JSON 파싱 실패: {e}")
            return {"raw": raw_text}

    # ──────────────────────────────────────────────
    # 프롬프트: JSON 없이 자연어로 직접 출력
    # ──────────────────────────────────────────────

    def _build_simulation_llm_question(
        self,
        insurance_company: str,
        product_name: str,
        special_contracts: List[Dict[str, Any]],
        question: str,
        context: str,
    ) -> str:
        sc_text = (
            "\n".join(
                f"- {sc.get('contract_name', '')} (요청 페이지: {sc.get('page_number', '?')})"
                for sc in special_contracts
            )
            if special_contracts
            else "없음"
        )

        return f"""당신은 10년 경력의 보험 약관 전문 상담사입니다.
아래 [보험 약관 정보]를 꼼꼼히 읽고, 고객의 질문에 대해 전문적이고 친절하게 답변하세요.
반드시 약관에 명시된 내용만 근거로 사용하고, 약관에 없는 내용은 추측하지 마세요.
반드시 JSON 형식으로만 답변하세요.

[보험 정보]
- 보험사: {insurance_company}
- 상품명: {product_name}
- 가입 특약:
{sc_text}

[고객 질문]
{question}

[출력 형식 - 반드시 이 JSON 키로만 작성]
{{
  "insurance_company": "{insurance_company}",
  "product_name": "{product_name}",
  "is_long_term": true,
  "conclusion": "보장 가능 여부를 한 문장으로 명확하게 말함 (예: 네, 보장됩니다. / 아니요, 보장되지 않습니다. / 조건에 따라 다릅니다.)",
  "reasoning": "약관 어떤 조항을 근거로 결론을 내렸는지 2~4문장으로 설명 (조항명과 약관 페이지 반드시 포함. / 예: 약관 제2-2조(보험금의 지급사유, p.466)에 따르면 ...)",
  "payment_criteria": {{
    "is_covered": "지금 여부에 따라 예/아니오로 대답",
    "payment_basis": "지급 기준 / 약관에 명시된 지급률 (예: 특약가입금액의 2%)",
    "payment_count": "지급 횟수 (예: 최초 1회 한정)",
    "payment_restrictions": "지급 제한 사항 (없으면 특별한 제한 없음으로 기재할 것)"
  }},
  "coverage_period": "이 특약의 보험기간이 언제부터 언제까지인지, 보장이 시작되는 시점이 언제인지 설명 (예: 계약일부터 분만일까지 등)",
  "condition_check": [
    {{
      "condition": "조건명",
      "is_satisfied": "고객이 말한 상황(임신 주차, 진단 등)이 해당 약관의 지급 조건을 충족하는지 항목별로 체크해 알려줄 것",
      "description": "충족 여부 및 근거 설명"
    }}
  ],
  "exclusions": "면책 및 제외 사항 (약관에 없으면 약관에 별도 면책 조항 없음)",
  "required_documents": [
    "보험금 청구에 실제로 필요한 서류 및 절차 1 (예: 진단서, 진단 확정일 기재된 의무기록 등)",
    "보험금 청구에 실제로 필요한 서류 및 절차 2",
    "보험금 청구에 실제로 필요한 서류 및 절차 3"
  ],
  "evidence_sources": [
    {{
      "page_number": 123,
      "text_snippet": "핵심 조항 인용 요약",
      "content": "해당 조항 전문 내용"
    }}
  ]
}}

[보험 약관 정보]
{context}
"""

    # ──────────────────────────────────────────────
    # LLM 호출 (JSON 파싱 없음 → 자연어 그대로 반환)
    # ──────────────────────────────────────────────

    def _invoke_simulation_llm(self, prompt: str) -> str:
        response = self.llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "당신은 보험 약관 전문 상담사입니다. "
                        "고객이 이해하기 쉽도록 친절하고 정확하게 답변하되, "
                        "반드시 제공된 약관 조항을 근거로만 답변합니다. "
                        "약관에 없는 내용은 절대 추측하거나 만들어내지 마세요. "
                        "답변은 항상 구조화된 형식으로, 충분히 상세하게 작성하세요."
                        "만약 질문이 보험 시뮬레이션과 무관하거나 의미를 알 수 없는 내용이면, "
                        "coclusion에 상광에 맞춰 '질문을 이해할 수 없습니다.' 혹은 '질문을 다시 입력해주세요.'라고 답해줘."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        return response.content if hasattr(response, "content") else str(response)


# 싱글톤 인스턴스
simulator = InsuranceSimulator()