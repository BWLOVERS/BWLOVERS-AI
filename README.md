# BWLOVERS-AI

BWLOVERS-AI는 임신부/태아 보험 상품 추천 및 약관 기반 시뮬레이션을 제공하는 FastAPI 기반 AI 프로젝트입니다.

RAG(FAISS + 보험 약관 JSON)와 여러 LLM(OpenAI/Gemini/Claude) 평가(RAGAS)를 통해 추천 결과 품질을 높입니다.
<img src="https://github.com/user-attachments/assets/114e752c-ccde-420e-9cf8-210f619ac16a" width="70%">


## 1) 프로젝트 설명

이 프로젝트는 다음 두 가지 기능을 제공합니다.

- ***보험 추천 API***: 사용자 임신/건강 정보를 입력받아 맞춤형 보험 상품 및 특약을 추천
- ***보험 시뮬레이션 API***: 특정 보험상품/특약 기준으로 사용자 질문에 대해 약관 근거 기반 답변 제공

핵심 기술:

- FastAPI API 서버
- FAISS 벡터 인덱스 기반 검색
- LangChain 기반 LLM 라우팅
- RAGAS 기반 후보 답변 평가

## 2) 저장소 구성 (Source code 설명)

```bash
├── README.md

├── requirements.txt

├── Dockerfile

├── docker-compose.yml

├── bw-ai/

│   ├── main.py                  # FastAPI 엔트리포인트 (/ai/recommend, /ai/simulation)

│   ├── insurance_recommender.py # 보험 추천 로직 (RAG + 멀티LLM + RAGAS)

│   ├── insurance_simulator.py   # 보험 시뮬레이션 로직

│   ├── rag_pipeline.py          # RAG 단일 질의 테스트/점검용

│   ├── ragas_evaluation.py      # faithfulness/relevancy 평가

│   ├── llm_router.py            # 활성 LLM 선택/병렬 실행

│   ├── llm/

│   │   ├── openai_llm.py

│   │   ├── gemini_llm.py

│   │   └── claude_llm.py

│   ├── prices.json              # 보험료 매핑 테이블

│   └── sum_insured.json         # 가입금액 매핑 테이블

├── data/                        # 원본 보험 PDF 데이터

├── json/

│   ├── Llama_json/              # 구조화된 약관 JSON

│   └── Llama_jason.ipynb        # PDF -> JSON 변환 실험/재생성 노트북

└── faiss_index/

├── index.faiss

└── index.pkl
```


## **3) How to build**

### **3-1. 로컬 Python 환경 빌드**

git clone <REPO_URL>

cd BWLOVERS-AI

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

### **3-2. Docker 이미지 빌드**

docker build -t bwlovers-ai:latest .


## **4) How to install**

### **4-1. 필수 환경변수 설정**

프로젝트 루트에 `.env` 파일을 생성하고 아래를 설정하세요.

#### 최소 1개 이상 필수

OPENAI_API_KEY=

GEMINI_API_KEY=

ANTHROPIC_API_KEY=

#### 사용 모델 (예: gemini 단독 / gemini,openai 병렬)

ENABLED_LLMS=gemini

#### 옵션

FAISS_DB_DIR=./faiss_index

INSURANCE_JSON_DIR=./json/Llama_json

REQUIRE_AUTH=false

BACKEND_URL=http://localhost:8080

> `*ENABLED_LLMS`의 첫 번째 모델이 기본 active 모델로 사용됩니다.*
> 

### **4-2. (옵션) Chroma 실행**

`docker-compose.yml`에 ChromaDB가 포함되어 있으나, 현재 핵심 추천/시뮬레이션 경로는 FAISS 파일 인덱스를 사용합니다.

docker compose up -d



## **5) How to run**

### **5-1. 로컬 실행**

cd bw-ai

uvicorn main:app --host 0.0.0.0 --port 8080 --reload

서버 확인:

- `GET http://localhost:8080/`

### **5-2. Docker 실행**

docker run --rm -p 8080:8080 --env-file .env bwlovers-ai:latest



## **6) How to test**

### **6-1. 헬스체크**

curl http://localhost:8080/

예상 응답:

{"message":"BWLOVERS AI 서버 실행 중","status":"healthy"}

### **6-2. 추천 API 테스트**

```bash
curl -X POST "http://localhost:8080/ai/recommend" \

-H "Content-Type: application/json" \

-d '{

"pregnancyInfo": {

"userId": 1,

"gestationalWeek": 20,

"isMultiplePregnancy": false,

"miscarriageHistory": 0,

"jobs": [{"jobName":"사무직"}]

},

"healthStatus": {

"pastDiseases": [],

"chronicDiseases": [],

"pregnancyComplications": []

}

}'
```

### **6-3. 시뮬레이션 API 테스트**

```bash
curl -X POST "http://localhost:8080/ai/simulation" \

-H "Content-Type: application/json" \

-d '{

"insurance_company": "교보라이프플래닛생명",

"product_name": "무배당 교보라플 아이사랑보험",

"special_contracts": [

{"contract_name":"선천이상 관련 특약", "page_number": 120}

],

"question": "임신 20주차에 해당 특약 보장 대상인지 알려줘"

}'
```


## **7) 재생성 가능성 (git clone 후 re-generate)**

채점 기준 대응을 위해, 데이터 재생성 절차를 아래처럼 명시합니다.

### **7-1. 원본 PDF -> 구조화 JSON 재생성**

- 원본 PDF 위치: `data/`
- 구조화 JSON 출력 위치: `json/Llama_json/`
- 재생성 도구: `json/Llama_jason.ipynb` (LlamaParse 기반)

노트북 실행 후 생성된 JSON이 `json/Llama_json/`에 저장되어야 합니다.

### **7-2. JSON -> FAISS 인덱스 재생성**

`faiss_index/`가 없거나 비어있으면 `insurance_recommender.py` 초기화 시 자동 생성됩니다.

수동 재생성 예시:

cd bw-ai

python -c "from insurance_recommender import InsuranceRecommender; InsuranceRecommender()"

실행 후 `faiss_index/index.faiss`, `faiss_index/index.pkl` 생성 여부 확인.


## **8) Sample data 설명**

- `data/`: 보험사 약관/상품요약서 PDF 원본 데이터
- `json/Llama_json/`: PDF 파싱 후 생성된 구조화 JSON 데이터
- `faiss_index/`: 구조화 JSON으로부터 생성된 검색 인덱스 산출물


## **9) Database / data used**

- **벡터 데이터베이스**: FAISS (로컬 파일 기반)
- **(옵션) ChromaDB**: `docker-compose.yml`에 정의되어 있으나 기본 추천 경로는 FAISS 사용
- **정적 매핑 데이터**:
    - `bw-ai/prices.json` (보험료)
    - `bw-ai/sum_insured.json` (가입금액)

## **10) Used open source**

주요 오픈소스 및 라이브러리:

- FastAPI, Uvicorn
- LangChain (`langchain`, `langchain-community`, `langchain-openai`, `langchain-google-genai`, `langchain-anthropic`, `langchain-huggingface`)
- FAISS (`faiss-cpu`)
- RAGAS, datasets
- sentence-transformers
- llama-index, llama-parse
- python-dotenv, requests, pandas, numpy

정확한 버전은 `requirements.txt` 참고.
