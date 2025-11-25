import data_loader
import ai_processor

def run_ai_pipeline():
    print("--- 1. 백엔드로부터 데이터 로드 시작 ---")
    
    # data_loader 모듈에서 데이터 가져오기 함수 호출
    user_data_dict = data_loader.fetch_user_data_from_backend()
    
    if not user_data_dict:
        print("파이프라인 중단: 데이터 로드 실패")
        return

    print("\n--- 2. AI 처리 모듈로 데이터 전달 ---")
    
    # ai_processor 모듈의 분석 함수에 딕셔너리 데이터 전달
    final_result = ai_processor.test_user_data(user_data_dict)
    
    print("\n--- 3. 최종 파이프라인 결과 ---")
    if final_result:
        print(f"최종 사용자 분석 결과: {final_result.get('analysis_result')}")
    
if __name__ == "__main__":
    # 반드시 Spring Boot 백엔드 서버가 실행 중이어야 함!
    # pip install requests (설치 필수)
    run_ai_pipeline()