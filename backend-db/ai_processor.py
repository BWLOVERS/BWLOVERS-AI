def test_user_data(user_data):
    """
    JSON (딕셔너리) 형태의 데이터를 받아 AI 분석을 수행하는 함수
    """
    if not user_data:
        print("경고: 분석할 데이터가 비어 있습니다.")
        return None

    print("\n[AI Processor Start]")
    user_name = user_data.get("username_for_ai", "Unknown User")

    # 데이터를 변수로 바로 사용
    print(f"User: {user_name}")

# --- 테스트 용도로만 사용: 메인 실행문 ---
if __name__ == "__main__":
    # 파일 자체 테스트 시 사용할 더미 데이터
    dummy_data = {
        "username_for_ai": "임시 사용자" 
    }
    test_user_data(dummy_data)