from datasets import Dataset
from ragas import evaluate
import math
from ragas.metrics import faithfulness, answer_relevancy

def safe_float(value, default=0.0):
    try:
        v = float(value)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def score_candidates(question: str, contexts: list[str], candidates: list[dict], judge_llm):
    # 평가 데이터셋 생성
    # 입력: 질문, 참고 문서, 추천 상품
    # candidates: LLM이 만든 후보 답변 여러 개
    # judge_llm: RAGAS 평가를 위한 판단 LLM
    rows = [
        {"question": question, 
        "answer": c["answer_text"], 
        "contexts": contexts
        } 
        for c in candidates
    ]
    ds = Dataset.from_list(rows)

    df = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        raise_exceptions=False,
    ).to_pandas()

    print("[RAGAS columns]", df.columns.tolist())
    print("[RAGAS result]")
    print(df)

    # 평가 결과 점수 계산
    # 출력: 최고 점수 후보 + 점수표 (faithfulness, answer_relevancy, total_score)
    # faithfulness: 충실도 점수 (답변이 검색된 context에 근거해서 말하고 있는지 평가)
    # answer_relevancy: 관련성 점수 (답변이 질문에 잘 맞는지 평가)

    scored = []
    for i, c in enumerate(candidates):
        f = safe_float(df.loc[i, "faithfulness"], default=0.0)
        r = safe_float(df.loc[i, "answer_relevancy"], default=0.0)
        
        total = safe_float(0.6 * f + 0.4 * r, default=0.0)
        scored.append({**c, "faithfulness": f, "answer_relevancy": r, "total_score": total})

    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return scored[0], scored