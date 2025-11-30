import requests
import json
import os

# 환경 변수에서 백엔드 주소 가져오기 (로컬 개발 환경 주소)
BACKEND_BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")
USER_ID = 1
TEST_ENDPOINT = f"/ai/data/{USER_ID}"

def fetch_user_data_from_backend():
    """
    백엔드 API를 호출하여 AI 분석에 필요한 사용자 데이터를 JSON 형태로 가져오기
    """
    full_url = BACKEND_BASE_URL + TEST_ENDPOINT
    print(f"[Data Loader] Connecting to: {full_url}")

    try:
        response = requests.get(full_url)
        print(f"[Data Loader] HTTP Status: {response.status_code}")
        response.raise_for_status()

        # JSON 응답을 파이썬 딕셔너리/리스트로 변환하여 반환
        data = response.json()

        print("\n[Data Loader] Raw JSON from backend:")
        print(json.dumps(data, indent=4, ensure_ascii=False))

        return data

    except requests.exceptions.RequestException as e:
        print(f"[Data Loader] Error fetching data from backend: {e}")
        # 실패 시 빈 딕셔너리 반환 또는 예외 발생
        return {}

# --- 테스트 용도로만 사용: 메인 실행문 ---
if __name__ == "__main__":
    # 이 파일이 직접 실행될 때만 데이터를 가져와 출력
    data = fetch_user_data_from_backend()
    print("\n[Data Loader Test Result]")
    print(json.dumps(data, indent=4, ensure_ascii=False))
