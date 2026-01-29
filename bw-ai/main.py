from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import requests
import os
import json
import uuid

app = FastAPI()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# 백엔드에서 받을 데이터 구조
class PregnancyComplication(BaseModel):
    pregnancyComplicationType: str

class HealthStatus(BaseModel):
    pastDiseases: Optional[List[Dict[str, Any]]] = []
    chronicDiseases: Optional[List[Dict[str, Any]]] = []
    pregnancyComplications: Optional[List[PregnancyComplication]] = []

class PregnancyInfo(BaseModel):
    userId: int
    gestationalWeek: int
    isFirstbirth: bool
    # 필요한 다른 필드들...

class BackendRequest(BaseModel):
    pregnancyInfo: PregnancyInfo
    healthStatus: HealthStatus

@app.get("/")
async def root():
    return {"message": "AI server running"}

@app.post("/ai/recommend")  
async def recommend(request: BackendRequest):  
    """
    백엔드에서 산모 데이터 받아서 처리 후 보험 추천 결과 전송
    """
    print("[AI] 백엔드에서 데이터 받음:")
    print(f"사용자 ID: {request.pregnancyInfo.userId}")
    print(f"임신 주차: {request.pregnancyInfo.gestationalWeek}")
    print(f"초산 여부: {request.pregnancyInfo.isFirstbirth}")
    print(f"임신 합병증: {len(request.healthStatus.pregnancyComplications)}개")
    
    # AI 추천 결과 (테스트용 간단 ver.)
    recommendations = [{
        "itemId": "rec-1",
        "insurance_company": "교보라이프플래닛",
        "product_name": "무배당 교보라플 어린이보험",
        "is_long_term": True,
        "monthly_cost": 1000,
        "summary_reason": f"임신 {request.pregnancyInfo.gestationalWeek}주차에 적합한 보험"
    }]
    
    # 백엔드로 추천 결과 전송
    result_id = str(uuid.uuid4())
    response = requests.post(
        f"{BACKEND_URL}/ai/callback/recommend",
        json={
            "resultId": result_id,
            "expiresInSec": 600,
            "items": recommendations
        },
        headers={"Content-Type": "application/json"}
    )
    
    print(f"[AI] 백엔드로 보험 추천 결과 전송: {response.status_code}")
    
    return {
        "success": True,
        "resultId": result_id,
        "processed_data": {
            "userId": request.pregnancyInfo.userId,
            "gestationalWeek": request.pregnancyInfo.gestationalWeek,
            "complications_count": len(request.healthStatus.pregnancyComplications)
        },
        "backend_response": {
            "status_code": response.status_code,
            "success": response.status_code == 200
        }
    }