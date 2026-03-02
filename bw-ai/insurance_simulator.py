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
    return _first_existing_dir(candidates) or os.path.abspath(os.path.join(CURRENT_DIR, "faiss_index"))

FAISS_DIR = _resolve_faiss_dir()

# FAISS 기반 RAG + LLM 관련 임포트
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from rag_pipeline import ask_question
    
    # insurance_recommender에서 이미 로드된 vectorstore 재사용 가능
    # 또는 독립적으로 로드
    
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    index_file = os.path.join(FAISS_DIR, "index.faiss")
    if os.path.exists(index_file):
        vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ 보험 시뮬레이션: FAISS 벡터스토어 로드됨: {vectorstore.index.ntotal}개 문서 (dir={FAISS_DIR})")
    else:
        vectorstore = None
        print(f"보험 시뮬레이션: FAISS 벡터스토어 없음 (dir={FAISS_DIR})")
    
    RAG_AVAILABLE = True
except Exception as e:
    print(f"보험 시뮬레이션 RAG 시스템 초기화 실패: {e}")
    vectorstore = None
    embeddings = None
    RAG_AVAILABLE = False
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class InsuranceSimulator:
    def __init__(self):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
    
    def analyze_simulation(
        self,
        insurance_company: str,
        product_name: str,
        special_contracts: List[Dict[str, Any]],
        question: str
    ) -> Dict[str, Any]:
        """
        특정 보험 상품과 특약에 대한 시뮬레이션 질문을 분석합니다.
        
        Returns:
            {
                "simulationId": str,
                "result": str  # LLM이 생성한 자연어 답변
            }
        """
        try:
            if not self.vectorstore:
                return {
                    "simulationId": uuid.uuid4().hex[:8],
                    "result": "보험 약관 데이터베이스를 사용할 수 없습니다."
                }
            
            # 1. 검색 쿼리 생성
            search_query = self._build_simulation_query(
                insurance_company,
                product_name,
                special_contracts,
                question
            )
            
            # 2. 관련 문서 검색
            relevant_docs = self._search_simulation_documents(
                insurance_company,
                product_name,
                special_contracts,
                question,
                search_query
            )
            
            if not relevant_docs:
                return {
                    "simulationId": uuid.uuid4().hex[:8],
                    "result": "관련 약관 정보를 찾을 수 없어 보장 여부를 확인할 수 없습니다."
                }
            
            # 3. LLM을 통한 시뮬레이션 분석
            context = self._build_simulation_context(relevant_docs, special_contracts)
            llm_question = self._build_simulation_llm_question(
                insurance_company,
                product_name,
                special_contracts,
                question,
                context
            )
            
            print(f"보험 시뮬레이션 LLM 요청 중... (보험사: {insurance_company})")
            rag_result = ask_question(llm_question)
            
            if rag_result and "answer" in rag_result:
                # LLM의 답변을 포맷팅하여 result로 사용
                result_text = self._format_llm_response(rag_result["answer"])
                
                return {
                    "simulationId": uuid.uuid4().hex[:8],
                    "result": result_text
                }
            
            return {
                "simulationId": uuid.uuid4().hex[:8],
                "result": "분석 중 오류가 발생했습니다."
            }
            
        except Exception as e:
            print(f"시뮬레이션 분석 실패: {e}")
            return {
                "simulationId": uuid.uuid4().hex[:8],
                "result": f"분석 중 오류가 발생했습니다: {str(e)}"
            }
    
    def _build_simulation_query(
        self,
        insurance_company: str,
        product_name: str,
        special_contracts: List[Dict[str, Any]],
        question: str
    ) -> str:
        """시뮬레이션 검색 쿼리 생성"""
        parts = [question]
        parts.append(insurance_company)
        parts.append(product_name)
        
        # 특약명 추가
        for sc in special_contracts:
            contract_name = sc.get("contract_name", "")
            if contract_name:
                parts.append(contract_name)
        
        return " ".join(parts)
    
    def _search_simulation_documents(
        self,
        insurance_company: str,
        product_name: str,
        special_contracts: List[Dict[str, Any]],
        question: str,
        search_query: str
    ) -> List:
        """시뮬레이션 관련 문서 검색 (특약 페이지 우선)"""
        if not self.vectorstore:
            return []
        
        try:
            # 일반 검색
            docs_with_scores = self.vectorstore.similarity_search_with_score(search_query, k=20)
            docs = [doc for doc, score in docs_with_scores]
            
            # 보험사 및 상품명 필터링
            filtered_docs = []
            for doc in docs:
                md = doc.metadata or {}
                doc_product = md.get("product_name", "")
                
                # 보험사명이 포함되어 있는지 확인
                if insurance_company in doc_product or insurance_company in str(md):
                    # 상품명이 포함되어 있는지 확인
                    if product_name in doc_product or product_name in str(md):
                        filtered_docs.append(doc)
            
            # 특약 페이지 우선 검색
            special_contract_docs = []
            for sc in special_contracts:
                page_num = sc.get("page_number")
                contract_name = sc.get("contract_name", "")
                
                if page_num:
                    # 해당 페이지 번호의 문서 검색
                    for doc in filtered_docs:
                        md = doc.metadata or {}
                        doc_page = md.get("page_number")
                        if doc_page and abs(int(doc_page) - int(page_num)) <= 5:  # ±5 페이지 범위
                            if contract_name in doc.page_content or contract_name in str(md):
                                special_contract_docs.append(doc)
            
            # 특약 문서를 우선, 나머지는 일반 문서
            result = []
            seen = set()
            for doc in special_contract_docs + filtered_docs:
                md = doc.metadata or {}
                doc_id = f"{md.get('page_number', '')}_{doc.page_content[:50]}"
                if doc_id not in seen:
                    seen.add(doc_id)
                    result.append(doc)
            return result[:10]  # !!! 상위 10개만
            
        except Exception as e:
            print(f"시뮬레이션 문서 검색 실패: {e}")
            return []
    
    def _build_simulation_context(
        self,
        documents: List,
        special_contracts: List[Dict[str, Any]]
    ) -> str:
        """시뮬레이션 컨텍스트 생성"""
        parts = []
        
        # 특약 정보 추가
        if special_contracts:
            parts.append("=== 가입 특약 정보 ===")
            for sc in special_contracts:
                parts.append(f"- {sc.get('contract_name', '')} (페이지: {sc.get('page_number', '?')})")
            parts.append("")
        
        # 문서 내용 추가
        parts.append("=== 보험 약관 정보 ===")
        for i, doc in enumerate(documents[:8]):
            md = doc.metadata or {}
            page_num = md.get("page_number", "?")
            parts.append(
                f"[문서 {i+1}] 페이지:{page_num}\n"
                f"내용:{doc.page_content[:1000]}"
            )
            parts.append("---")
        
        return "\n".join(parts)
    
    def _build_simulation_llm_question(
        self,
        insurance_company: str,
        product_name: str,
        special_contracts: List[Dict[str, Any]],
        question: str,
        context: str
    ) -> str:
        """시뮬레이션 LLM 질문 생성"""
        special_contracts_text = "\n".join([
            f"- {sc.get('contract_name', '')} (페이지: {sc.get('page_number', '?')})"
            for sc in special_contracts
        ]) if special_contracts else "없음"
        
        return f"""
역할: 보험 약관 전문가 및 보장 분석가

보험 정보:
- 보험사: {insurance_company}
- 상품명: {product_name}
- 가입 특약:
{special_contracts_text}

시뮬레이션 질문:
{question}

지침:
1. 제공된 보험 약관 정보를 바탕으로 위 시뮬레이션 상황에 대한 보장 가능 여부를 분석하세요.
2. 보장 가능한 경우: 보장 내용, 지급 금액, 조건 등을 구체적으로 설명하세요.
3. 보장 불가능한 경우: 보장되지 않는 이유, 제한 사항 등을 명확히 설명하세요.
4. 특약이 가입되어 있다면 해당 특약의 보장 범위를 우선적으로 확인하세요.
5. 사용자가 이해하기 쉽도록 자연스러운 한국어로 답변하세요.
6. 약관의 구체적인 내용과 페이지 번호를 참고하여 정확하게 답변하세요.
7. 답변은 자연스러운 문장으로 작성하되, 보장 여부, 보장 내용, 제한 사항 등을 명확히 포함하세요.

[보험 약관 정보]
{context}

위 질문에 대해 보험 약관을 바탕으로 상세하고 정확하게 답변해주세요.
"""
    
    def _format_llm_response(self, llm_response: str) -> str:
        """LLM 응답을 자연어 형식으로 변환"""
        try:
            # JSON 형식인지 확인
            json_block = re.search(r"(\{.*\})", llm_response, re.DOTALL)
            if json_block:
                data = json.loads(self._fix_json_string(json_block.group(1)))
                
                # 자연어 형식으로 변환
                parts = []
                
                # 보장 여부
                is_covered = data.get("is_covered", False)
                if is_covered:
                    parts.append("보장 가능합니다.")
                else:
                    parts.append("보장되지 않습니다.")
                parts.append("")
                
                # 분석 내용
                if data.get("analysis"):
                    parts.append(f"{data['analysis']}")
                    parts.append("")
                
                # 보장 상세 내용
                if data.get("coverage_details"):
                    parts.append(f"보장 상세 내용:")
                    parts.append(f"{data['coverage_details']}")
                    parts.append("")
                
                # 제한 사항
                limitations = data.get("limitations", [])
                if limitations:
                    parts.append("제한 사항:")
                    for limitation in limitations:
                        parts.append(f"- {limitation}")
                
                return "\n".join(parts)
            else:
                # JSON이 아니면 그대로 반환 (이미 자연어 형식)
                return llm_response.strip()
        except Exception as e:
            print(f"LLM 응답 포맷팅 실패, 원본 반환: {e}")
            # 파싱 실패 시 원본 반환
            return llm_response.strip()
    
    def _fix_json_string(self, s: str) -> str:
        """JSON 문자열 정리"""
        s = s.replace("「", "'").replace("」", "'").replace(""", "'").replace(""", "'")
        return s.replace("True", "true").replace("False", "false").replace("None", "null")


# 싱글톤 인스턴스
simulator = InsuranceSimulator()