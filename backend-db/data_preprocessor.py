import json

def test_user_data(user_data):
    """
    JSON 형태의 데이터를 받아 구조를 분석하고 모든 값을 출력.
    user.name 을 자동 추출하여 확인.
    """
    if not user_data:
        print("[AI Processor] 경고: 분석할 데이터가 비어 있습니다.")
        return None

    print("\n[AI Processor] ===== 데이터 구조 확인 =====")
    print(f"[AI Processor] 타입: {type(user_data)}")

    if isinstance(user_data, dict):
        print(f"[AI Processor] 최상위 키 목록: {list(user_data.keys())}")

    print("\n[AI Processor] ===== 전체 데이터 pretty print =====")
    print(json.dumps(user_data, indent=4, ensure_ascii=False))
    
    user_block = user_data.get("user", {})
    user_name = user_block.get("name", "NoName")
    user_email = user_block.get("email", "NoEmail")

    print("\n[AI Processor] ===== 핵심 데이터 추출 =====")
    print(f"산모 이름: {user_name}")
    print(f"산모 이메일: {user_email}")

    # 다른 값들도 확인하고 싶으면 출력
    preg = user_data.get("pregnancy_info", {})
    health = user_data.get("health_status", {})

    print("\n[AI Processor] 임신 정보:")
    print(json.dumps(preg, indent=4, ensure_ascii=False))

    print("\n[AI Processor] 건강 상태 정보:")
    print(json.dumps(health, indent=4, ensure_ascii=False))

    # 최종 return (나중에 AI 넘길 때 수정 가능)
    return {
        "name": user_name,
        "email": user_email,
        "pregnancy_info": preg,
        "health_status": health,
        "extracted_username": user_name  
    }

# 단독 테스트용
if __name__ == "__main__":
    dummy = {
        "user": {"name": "봉원맘", "email": "test@test.com"},
        "pregnancy_info": {},
        "health_status": {}
    }
    test_user_data(dummy)
