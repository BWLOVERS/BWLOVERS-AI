from fastapi import FastAPI
from models.maternity import MaternityProfile
from rag_pipeline import ask_question, print_response

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI AI server is running"}

@app.post("/ai/recommend")
async def recommend(profile: MaternityProfile):
    """
    Spring에서 보내준 산모 데이터를 그대로 다시 돌려주기 (디버깅용)
    """
    data = profile.model_dump()
    print("[FastAPI] 받은 산모 데이터:")
    print(data)
    
    user_query = "이 산모에게 불필요한 중복 특약과 가입하면 좋은 특약을 알려줘."
    rag_result = ask_question(user_query, profile=data)
    print_response(rag_result)
    return {"success": True, "profile": data, "rag_result": rag_result}
