from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any, Optional, List, Union
import os
import uuid
from datetime import datetime, date
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import json
import logging
import httpx
from insurance_recommender import recommender

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bw-ai")

app = FastAPI(title="BWLOVERS AI", version="1.0.0")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    log.error("[422] url=%s errors=%s body=%s", request.url, exc.errors(), body.decode("utf-8", "ignore"))
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def yyyymmdd_to_date(v: int) -> date:
    s = str(v)
    return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def any_to_date(v):
    if v is None or isinstance(v, date):
        return v
    if isinstance(v, int):
        s = str(v)
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    if isinstance(v, list) and len(v) == 3:
        return date(int(v[0]), int(v[1]), int(v[2]))
    if isinstance(v, str):
        try:
            return date.fromisoformat(v)
        except:
            return None
    return v

# --- 요청 스키마 (Java DTO와 일치화) ---

class UserProfile(BaseModel):
    """Java의 PregnancyInfoRequest와 1:1 매칭"""
    userId: Optional[Union[int, str]] = None
    birthDate: Optional[Any] = None
    height: Optional[int] = None
    weightPre: Optional[int] = None
    weightCurrent: Optional[int] = None
    isFirstbirth: Optional[bool] = None
    gestationalWeek: Optional[int] = None # Java: Integer gestationalWeek
    expectedDate: Optional[Any] = None
    isMultiplePregnancy: Optional[bool] = None # Java: Boolean isMultiplePregnancy
    miscarriageHistory: Optional[int] = 0
    jobName: Optional[str] = None

    @field_validator("birthDate", "expectedDate", mode="before")
    def parse_dates(cls, v):
        return any_to_date(v)

class PastDisease(BaseModel):
    pastDiseaseType: str
    pastCured: bool
    pastLastTreatedAt: Optional[str] = None

class ChronicDisease(BaseModel):
    chronicDiseaseType: str
    chronicOnMedication: bool

class HealthStatus(BaseModel):
    userId: Optional[Union[int, str]] = None
    pastDiseases: List[PastDisease] = Field(default_factory=list)
    chronicDiseases: List[ChronicDisease] = Field(default_factory=list)
    pregnancyComplications: List[str] = Field(default_factory=list)

class BackendRequest(BaseModel):
    """Java: private PregnancyInfoRequest user_profile;"""
    user_profile: UserProfile
    health_status: HealthStatus

# --- 응답 스키마 ---
class EvidenceSourceOut(BaseModel):
    page_number: int
    text_snippet: str

class SpecialContractOut(BaseModel):
    contract_name: str
    contract_description: str
    contract_recommendation_reason: str
    key_features: List[str]
    page_number: int

class ItemOut(BaseModel):
    itemId: str
    insurance_company: str
    product_name: str
    is_long_term: bool
    sum_insured: int
    monthly_cost: str
    insurance_recommendation_reason: Optional[str] = None
    special_contracts: Optional[List[SpecialContractOut]] = None
    evidence_sources: Optional[List[EvidenceSourceOut]] = None

class RecommendListResponseOut(BaseModel):
    resultId: str
    expiresInSec: int = 600
    items: List[ItemOut]

# 서버 상태 확인
@app.get("/")
async def root():
    return {"message": "BWLOVERS AI 서버 실행 중", "version": "1.0.0", "status": "healthy"}



@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 보험 추천 API (백엔드 데이터 수신 → AI 처리 → 백엔드 콜백)
@app.post("/ai/recommend")
async def recommend(request: BackendRequest):
    """RAG 기반 보험 추천 API"""
    try:
        log.info(f"[요청] user_id={request.user_profile.userId}")
        
        # 직접 recommender.generate_rag_recommendation 호출
        user_profile = request.user_profile.model_dump()
        health_status = request.health_status.model_dump()
        recommendation_result = recommender.generate_rag_recommendation(user_profile, health_status)
        
        # RAGAS 성능 확인
        if "rag_metadata" in recommendation_result:
            metadata = recommendation_result["rag_metadata"]
            log.info(f"[RAGAS] ragas 점수={metadata.get('llm_response_quality', 0):.2f}, "
                    f"참고 문서수={metadata.get('documents_used', 0)}, "
                    f"사용자 임신주수={metadata.get('gestational_week', 0)}")
        
        return RecommendListResponseOut(**{
            "resultId": recommendation_result.get("resultId", f"rec-{uuid.uuid4().hex[:8]}"),
            "expiresInSec": 600,
            "items": recommendation_result.get("items", [])
        })
        
    except Exception as e:
        log.error(f"[오류] 추천 생성 실패: {e}")
        # 폴백 추천 (RAG 기반 추천 실패 시 기본 추천으로 응답함)
        fallback_items = [
            ItemOut(
                itemId=f"fallback-{uuid.uuid4().hex[:8]}",
                insurance_company="교보라이프플래닛",
                product_name="무배당 교보라플 어린이보험",
                is_long_term=True,
                sum_insured=10000000,
                monthly_cost="1000원",
                insurance_recommendation_reason="시스템 오류로 기본 추천 제공"
            )
        ]
        return RecommendListResponseOut(
            resultId=f"error-{uuid.uuid4().hex[:8]}",
            expiresInSec=600,
            items=fallback_items
        )


async def send_to_backend(result_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    추천 결과를 백엔드 콜백으로 전송
    """
    url = f"{BACKEND_URL}/ai/callback/recommend"
    payload = {
        "resultId": result_id,
        "expiresInSec": 600,
        "items": items
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "BWLOVERS-AI/1.0"
    }

    access_token = os.getenv("BACKEND_ACCESS_TOKEN")
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            log.info("[AI] 백엔드로 결과 전송 시도... url=%s", url)
            resp = await client.post(url, json=payload, headers=headers)

        result: Dict[str, Any] = {"status_code": resp.status_code, "success": resp.status_code == 200}
        if resp.status_code != 200:
            result["error"] = resp.text
        return result

    except httpx.RequestError as e:
        return {"status_code": None, "success": False, "error": str(e)}



# 하드코딩된 예시 추천 데이터 변환
def generate_recommendations_db(request: BackendRequest) -> List[Dict[str, Any]]:
    """
    테스트용 예시 추천 데이터 반환
    """
    return [
        {
            "insurance_company": "교보라이프플래닛",
            "is_long_term": True,
            "product_name": "무배당 교보라플 어린이보험",
            "insurance_recommendation_reason": "일반보험 추천 이유 (사용자에 따라 달라지는 것)",
            "monthly_cost": 1000,
            "special_contracts": [
                {
                    "contract_name": "태아 검사비 지원 특약",
                    "contract_description": "특약 설명",
                    "contract_recommendation_reason": "특약 추천 이유 (사용자에 따라 달라지는 것)",
                    "key_features": ["다이렉트 보험으로 보험료 저렴", "태아 가입 시 선천성이상 및 저체중아 보장 특약 선택 가능"],
                    "page_number": 12
                }
            ],
            "evidence_sources": [
                {"page_number": 12, "text_snippet": "제 5조(보상하는 손해) ... 태아 검사비 지원에 관한 사항"}
            ]
        },
        {
            "insurance_company": "교보라이프플래닛2",
            "is_long_term": True,
            "product_name": "무배당 교보라플 어린이보험",
            "insurance_recommendation_reason": "일반보험 추천 이유 (사용자에 따라 달라지는 것)",
            "monthly_cost": 1000,
            "special_contracts": [
                {
                    "contract_name": "임신 중독증 진단비",
                    "contract_description": "특약 설명",
                    "contract_recommendation_reason": "특약 추천 이유 (사용자에 따라 달라지는 것)",
                    "key_features": ["다이렉트 보험으로 보험료 저렴", "태아 가입 시 선천성이상 및 저체중아 보장 특약 선택 가능"],
                    "page_number": 45
                }
            ],
            "evidence_sources": [
                {"page_number": 45, "text_snippet": "특약 별표 2 ... 임신 중독증 진단 확정 기준"}
            ]
        }
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")