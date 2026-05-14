from fastapi import FastAPI, HTTPException, Request, Header
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
from insurance_simulator import InsuranceSimulator
simulator = InsuranceSimulator()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bw-ai")

app = FastAPI(title="BWLOVERS AI", version="1.0.0")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# 에러 핸들러 
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    log.error("[422] url=%s errors=%s body=%s", request.url, exc.errors(), body.decode("utf-8", "ignore"))
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# 날짜 변환
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

# 직업 스키마
class JobIn(BaseModel):
    jobId: Optional[int] = None
    jobName: Optional[str] = None
    riskLevel: Optional[int] = None

# 보험 추천 요청 스키마
class UserProfileIn(BaseModel):
    infoId: Optional[int] = None
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
    jobs: Optional[List[JobIn]] = Field(default_factory=list)

    @field_validator("birthDate", "expectedDate", mode="before")
    def parse_dates(cls, v):
        return any_to_date(v)

class PastDisease(BaseModel):
    pastId: Optional[int] = None
    pastDiseaseType: str
    pastCured: bool
    pastLastTreatedYm: Optional[str] = None #YYYYMM

class ChronicDisease(BaseModel):
    chronicId: Optional[int] = None
    chronicDiseaseType: str
    chronicOnMedication: bool

class PregnancyComplication(BaseModel):
    complicationId: Optional[int] = None
    pregnancyComplicationType: str

class HealthStatusIn(BaseModel):
    statusId: Optional[int] = None
    userId: Optional[Union[int, str]] = None
    createdAt: Optional[Any] = None
    pastDiseases: List[PastDisease] = Field(default_factory=list)
    chronicDiseases: List[ChronicDisease] = Field(default_factory=list)
    pregnancyComplications: List[PregnancyComplication] = Field(default_factory=list)

class BackendRequest(BaseModel):
    pregnancyInfo: UserProfileIn
    healthStatus: HealthStatusIn  

# 보험 추천 응답

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
    sum_insured: str
    monthly_cost: str
    insurance_recommendation_reason: Optional[str] = None
    special_contracts: Optional[List[SpecialContractOut]] = None
    evidence_sources: Optional[List[EvidenceSourceOut]] = None

class RecommendListResponseOut(BaseModel):
    resultId: str
    expiresInSec: int = 600
    items: List[ItemOut]

# 시뮬레이션 요청
class SpecialContractIn(BaseModel):
    contract_name: str
    page_number: int

class SimulationRequestIn(BaseModel):
    insurance_company: str
    product_name: str
    special_contracts: List[SpecialContractIn] = Field(default_factory=list)
    question: str

# 시뮬레이션 응답 
class SpecialContractOut(BaseModel):
    contract_name: str
    page_number: int

class SimulationResponseOut(BaseModel):
    resultId: str
    insurance_company: str
    product_name: str
    special_contracts: List[SpecialContractOut]
    question: str
    result: str

# --- 보험 추천 API 엔드포인트 ---

@app.get("/")
async def root():
    return {"message": "BWLOVERS AI 서버 실행 중", "status": "healthy"}

@app.post("/ai/recommend")
async def recommend(request: BackendRequest):
    try:
        # 데이터 추출
        u_prof = request.pregnancyInfo
        h_stat = request.healthStatus
        
        log.info(f"[요청 수신] user_id={u_prof.userId}, 주수={u_prof.gestationalWeek}")
        
        # 1. 추천 엔진용 Dictionary 변환 
        user_profile_dict = u_prof.model_dump()
        user_profile_dict['gestational_week'] = u_prof.gestationalWeek
        user_profile_dict['is_multiple_pregnancy'] = u_prof.isMultiplePregnancy
        user_profile_dict['miscarriage_history'] = u_prof.miscarriageHistory
        # 2. jobs 리스트에서 첫 번째 직업명 추출
        if u_prof.jobs:
            user_profile_dict['jobName'] = u_prof.jobs[0].jobName
        else:
            user_profile_dict['jobName'] = None
        
        health_status_dict = h_stat.model_dump()
        # pregnancyComplications 객체 배열 → 문자열 배열로 변환
        health_status_dict['pregnancyComplications'] = [
            c.pregnancyComplicationType for c in h_stat.pregnancyComplications
        ]
        # pastDiseases의 날짜 필드명 보정 (Ym → At)
        for d in health_status_dict.get('pastDiseases', []):
            if 'pastLastTreatedYm' in d:
                d['pastLastTreatedAt'] = d.pop('pastLastTreatedYm')
        
        # 3. RAG 추천 엔진 호출
        recommendation_result = recommender.generate_rag_recommendation(user_profile_dict, health_status_dict)
        
        # 4. 결과 처리
        items = recommendation_result.get("items", [])
        raw_id = recommendation_result.get("resultId", uuid.uuid4().hex[:8])
        clean_id = raw_id.replace("rag-", "")

        # 메타데이터
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

# --- 보험 시뮬레이션 API 엔드포인트 ---
@app.post("/ai/simulation")
async def simulation(
    request: SimulationRequestIn,
    authorization: Optional[str] = Header(None, alias="Authorization")
):

    try:
        # accessToken 검증 
        # 제어: REQUIRE_AUTH=false로 설정시 검증 스킵
        require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
        
        if require_auth:
            access_token = None
            if authorization:
                parts = authorization.split()
                if len(parts) == 2 and parts[0].lower() == "bearer":
                    access_token = parts[1]
                else:
                    access_token = authorization
            
            if not access_token:
                raise HTTPException(status_code=401, detail="accessToken이 필요합니다.")
        
        log.info(f"[시뮬레이션 요청] 보험사={request.insurance_company}, 상품={request.product_name}")
        
        # 시뮬레이션 분석 실행
        simulation_result = simulator.analyze_simulation(
            insurance_company=request.insurance_company,
            product_name=request.product_name,
            special_contracts=[sc.model_dump() for sc in request.special_contracts],
            question=request.question
        )
        
        # 응답 형식 변환
        return SimulationResponseOut(
            resultId=simulation_result.get("resultId", uuid.uuid4().hex[:8]),
            insurance_company=request.insurance_company,
            product_name=request.product_name,
            special_contracts=[
                SpecialContractOut(
                    contract_name=sc.contract_name,
                    page_number=sc.page_number
                ) for sc in request.special_contracts
            ],
            question=request.question,
            result=simulation_result.get("result", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[시뮬레이션 오류] 분석 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"시뮬레이션 분석 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)