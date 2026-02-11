from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List, Union
import os
import uuid
from datetime import datetime, date
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
import httpx
from insurance_recommender import recommender

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bw-ai")

app = FastAPI(title="BWLOVERS AI", version="1.0.0")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# --- 에러 핸들러 ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    log.error("[422] url=%s errors=%s body=%s", request.url, exc.errors(), body.decode("utf-8", "ignore"))
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# --- 날짜 변환 유틸리티 ---
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
            return date.fromisoformat(v.split('T')[0])
        except:
            return None
    return v

# --- 요청 스키마 (Java DTO와 1:1 매칭) ---
class UserProfileIn(BaseModel):
    """Java의 PregnancyInfoRequest 구조"""
    userId: Optional[Union[int, str]] = None
    birthDate: Optional[Any] = None
    height: Optional[int] = None
    weightPre: Optional[int] = None
    weightCurrent: Optional[int] = None
    isFirstbirth: Optional[bool] = None
    gestationalWeek: Optional[int] = None
    expectedDate: Optional[Any] = None
    isMultiplePregnancy: Optional[bool] = None
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

class HealthStatusIn(BaseModel):
    """Java의 HealthStatusRequest 구조"""
    userId: Optional[Union[int, str]] = None
    pastDiseases: List[PastDisease] = Field(default_factory=list)
    chronicDiseases: List[ChronicDisease] = Field(default_factory=list)
    pregnancyComplications: List[str] = Field(default_factory=list)

class BackendRequest(BaseModel):
    """Java: FastApiRequest { user_profile, health_status }"""
    user_profile: UserProfileIn
    health_status: HealthStatusIn

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

# --- API 엔드포인트 ---

@app.get("/")
async def root():
    return {"message": "BWLOVERS AI 서버 실행 중", "status": "healthy"}

@app.post("/ai/recommend")
async def recommend(request: BackendRequest):
    try:
        # 1. 데이터 추출
        u_prof = request.user_profile
        h_stat = request.health_status
        
        log.info(f"[요청 수신] user_id={u_prof.userId}, 주수={u_prof.gestationalWeek}")
        
        # 2. 추천 엔진용 Dictionary 변환 (필드명 보정)
        # 추천 엔진 내부의 _analyze_user_profile이 기대하는 키값들을 명시적으로 세팅
        user_profile_dict = u_prof.model_dump()
        user_profile_dict['gestational_week'] = u_prof.gestationalWeek
        user_profile_dict['is_multiple_pregnancy'] = u_prof.isMultiplePregnancy
        user_profile_dict['miscarriage_history'] = u_prof.miscarriageHistory
        
        health_status_dict = h_stat.model_dump()
        
        # 3. RAG 추천 엔진 호출
        recommendation_result = recommender.generate_rag_recommendation(user_profile_dict, health_status_dict)
        
        # 4. 결과 처리 (Fallback 여부 확인 및 ID 정리)
        items = recommendation_result.get("items", [])
        raw_id = recommendation_result.get("resultId", uuid.uuid4().hex[:8])
        clean_id = raw_id.replace("rag-", "")

        # 메타데이터 로깅
        if "rag_metadata" in recommendation_result:
            meta = recommendation_result["rag_metadata"]
            log.info(f"[RAG 결과] 문서수={meta.get('documents_used', 0)}, 주수={meta.get('gestational_week', 0)}")
        
        return RecommendListResponseOut(
            resultId=clean_id,
            expiresInSec=600,
            items=items
        )
        
    except Exception as e:
        log.error(f"[메인 오류] 추천 프로세스 실패: {e}", exc_info=True)
        return RecommendListResponseOut(
            resultId=f"err-{uuid.uuid4().hex[:8]}",
            items=[]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)