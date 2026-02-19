# 1. 베이스 이미지
FROM python:3.11-slim

# 2. 작업 디렉토리 설정 (기준점을 /app으로 잡습니다)
WORKDIR /app

# 3. 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 설치 (루트의 requirements.txt를 /app으로 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 및 데이터 복사 (매우 중요!)
# 이미지에서 본 것처럼 bw-ai 폴더 '내용'을 /app에 바로 풉니다.
COPY bw-ai/ . 
COPY faiss_index/ ./faiss_index/
COPY json/ ./json/

# 6. 환경 변수 설정
# 이제 모든 파일이 /app에 있으므로 /app만 등록하면 됩니다.
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 7. 포트 노출 (Cloud Run은 8080을 사용)
EXPOSE 8080

# 8. 실행 명령어
# 이제 main.py가 /app 바로 아래에 있으므로 파일명만 적으면 됩니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]