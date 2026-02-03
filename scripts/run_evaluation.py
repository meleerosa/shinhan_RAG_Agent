import json
import os
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# [중요] 우리가 만든 Agent를 import 합니다.
# 파일명이 'real_agent_final_v4.py'라고 가정합니다. 다르면 수정하세요.
try:
    from real_agent_final_v4 import app
    print(">>> Agent 로드 성공!")
except ImportError:
    print("Error: 'real_agent_final_v4.py' 파일을 찾을 수 없습니다.")
    print("평가 스크립트와 같은 폴더에 에이전트 파일을 두거나, 경로를 확인하세요.")
    exit(1)

# ==========================================
# 0. 설정
# ==========================================
TEST_DATA_PATH = "/home/wlaud/projects/shinhan/data/test_golden_20.json"
RESULT_CSV_PATH = "/home/wlaud/projects/shinhan/data/evaluation_result.csv"

# 채점관 LLM (객관성을 위해 GPT-4o 사용 권장)
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ==========================================
# 1. 평가 함수 정의
# ==========================================

def calculate_profile_score(expected, extracted):
    """프로필 추출 정확도 계산 (0.0 ~ 1.0)"""
    if not expected: return 1.0 # 기대값이 없으면 만점 처리
    
    matched = 0
    total = len(expected)
    
    for k, v in expected.items():
        extracted_val = str(extracted.get(k, ''))
        expected_val = str(v)
        
        # 단순 포함 여부나 일치 여부 확인 (유연하게)
        if expected_val in extracted_val or extracted_val in expected_val:
            matched += 1
        elif k == "age" and extracted_val: # 나이는 숫자 변환 비교
            try:
                if int(float(extracted_val)) == int(v): matched += 1
            except: pass
            
    return (matched / total) * 100

def check_retrieval_hit(target_product, candidates):
    """검색 결과에 정답 상품이 있는지 확인 (Hit@K)"""
    if not candidates: return False
    
    for cand in candidates:
        cand_name = cand.get('product_name', '')
        # 공백 제거 후 비교 or 부분 일치
        if target_product.replace(" ", "") in cand_name.replace(" ", ""):
            return True
        if cand_name in target_product:
            return True
            
    return False

def evaluate_response_quality(query, target_product, eval_points, agent_response):
    """LLM을 판사로 사용하여 답변 품질 채점 (1~10점)"""
    
    prompt = f"""
    당신은 깐깐한 AI 평가관입니다. 
    금융 AI Agent의 답변 품질을 평가해주세요.
    
    [평가 기준]
    1. **정확성**: 사용자가 원하는 상품(**{target_product}**)을 추천했는가?
    2. **필수사항**: 평가 포인트 {eval_points}를 답변에 포함했는가?
    3. **유용성**: 구체적인 수치나 행동 가이드(가입방법 등)를 제공했는가?
    
    [사용자 질문]
    {query}
    
    [Agent 답변]
    {agent_response}
    
    위 기준에 따라 1점부터 10점 사이의 점수를 매겨주세요.
    반드시 숫자만 출력하세요. (예: 8)
    """
    
    try:
        res = judge_llm.invoke([SystemMessage(content=prompt)])
        score = int(res.content.strip())
        return score
    except:
        return 5 # 에러 시 중간 점수

# ==========================================
# 2. 테스트 실행 루프
# ==========================================

if not os.path.exists(TEST_DATA_PATH):
    print("Error: 테스트 데이터가 없습니다. create_golden_testset.py를 먼저 실행하세요.")
    exit(1)

with open(TEST_DATA_PATH, 'r') as f:
    test_cases = json.load(f)

results = []
print(f"\n>>> 총 {len(test_cases)}개 케이스 평가 시작...\n")

for case in tqdm(test_cases):
    query = case['query']
    target = case['target_product']
    points = case['evaluation_points']
    exp_profile = case['expected_profile']
    
    # 1. Agent 실행
    # (상태 초기화)
    initial_state = {
        "messages": [],
        "user_query": query,
        "profile": {},
        "ask_count": 0,
        "missing_info": [],
        "candidates": []
    }
    
    try:
        output = app.invoke(initial_state)
        
        final_res = output['final_response']
        ext_profile = output['profile']
        candidates = output.get('candidates', [])
        
        # 2. 지표 계산
        # A. Profile Score
        prof_score = calculate_profile_score(exp_profile, ext_profile)
        
        # B. Retrieval Hit
        is_hit = check_retrieval_hit(target, candidates)
        
        # C. Response Quality (LLM Judge)
        qual_score = evaluate_response_quality(query, target, points, final_res)
        
        # 결과 기록
        results.append({
            "id": case['id'],
            "category": case['category'],
            "query": query,
            "target": target,
            "profile_score": prof_score,
            "retrieval_hit": is_hit,
            "quality_score": qual_score,
            "agent_response": final_res[:100] + "..." # 로그용 요약
        })
        
    except Exception as e:
        print(f"Error in Case {case['id']}: {e}")
        results.append({
            "id": case['id'],
            "error": str(e)
        })

# ==========================================
# 3. 결과 분석 및 리포트
# ==========================================
df = pd.DataFrame(results)

# 성공한 케이스만 필터링
success_df = df[df['error'].isna()].copy()

print("\n" + "="*50)
print("📊 [최종 평가 리포트]")
print("="*50)

if len(success_df) > 0:
    # 평균 점수 계산
    avg_profile = success_df['profile_score'].mean()
    hit_rate = (success_df['retrieval_hit'].sum() / len(success_df)) * 100
    avg_quality = success_df['quality_score'].mean()
    
    print(f"1. 프로필 추출 정확도 : {avg_profile:.1f}점 (100점 만점)")
    print(f"2. 상품 검색 성공률   : {hit_rate:.1f}% (Top-3 기준)")
    print(f"3. 답변 품질 점수     : {avg_quality:.1f}점 (10점 만점)")
    
    print("\n[카테고리별 검색 성공률]")
    print(success_df.groupby('category')['retrieval_hit'].mean() * 100)
    
    # CSV 저장
    df.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n상세 결과 저장 완료: {RESULT_CSV_PATH}")
    
    # 개선 가이드
    print("\n💡 [개선 포인트 제안]")
    if hit_rate < 70:
        print("- 검색 성능이 낮습니다. 'extract_sample_json.py'의 키워드 추출 로직을 보강하거나, 'retrieve_node'의 검색어 확장을 강화하세요.")
    if avg_quality < 7:
        print("- 답변 품질이 낮습니다. 'response_node'의 프롬프트를 더 구체적으로 수정하거나, Critic의 역할을 강화하세요.")
else:
    print("평가 데이터가 없거나 모든 테스트가 실패했습니다.")
