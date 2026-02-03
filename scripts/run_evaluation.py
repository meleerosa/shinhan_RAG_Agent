import json
import os
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================
# [설정] 에이전트 Import
# ==========================================
try:
    from real_agent_final_v4 import app
    print(">>> Agent 로드 성공!")
except ImportError:
    print("Error: 에이전트 파일을 찾을 수 없습니다.")
    exit(1)

# ==========================================
# 0. 설정
# ==========================================
TEST_DATA_PATH = "/home/wlaud/projects/shinhan/data/test_golden_20.json"
RESULT_CSV_PATH = "/home/wlaud/projects/shinhan/log/evaluation_result.csv"
RESULT_JSON_PATH = "/home/wlaud/projects/shinhan/log/evaluation_detail_log.json"

# 채점관 LLM
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ==========================================
# 1. 평가 함수 정의
# ==========================================
def calculate_profile_score(expected, extracted):
    """프로필 추출 정확도 계산"""
    if not expected: return 1.0
    matched = 0
    total = len(expected)
    for k, v in expected.items():
        extracted_val = str(extracted.get(k, ''))
        expected_val = str(v)
        if expected_val in extracted_val or extracted_val in expected_val:
            matched += 1
        elif k == "age" and extracted_val:
            try:
                if int(float(extracted_val)) == int(v): matched += 1
            except: pass
    return (matched / total) * 100

def check_retrieval_hit(target_product, candidates):
    """검색 결과 Hit 여부 확인"""
    if not candidates: return False
    for cand in candidates:
        cand_name = cand.get('product_name', '')
        if target_product.replace(" ", "") in cand_name.replace(" ", ""):
            return True
        if cand_name in target_product:
            return True
    return False

def evaluate_response_quality(query, target_product, eval_points, agent_response):
    """LLM 기반 답변 품질 평가"""
    prompt = f"""
    [평가 기준]
    1. 정확성: 상품({target_product}) 추천 여부
    2. 필수사항: {eval_points} 포함 여부
    3. 유용성: 구체적 가이드 제공 여부
    
    질문: {query}
    답변: {agent_response}
    
    1~10점 사이의 점수(숫자만)를 출력하세요.
    """
    try:
        res = judge_llm.invoke([SystemMessage(content=prompt)])
        return int(res.content.strip())
    except:
        return 5

# ==========================================
# 2. 테스트 실행 루프 (멀티턴 지원 & 상세 로깅)
# ==========================================
if not os.path.exists(TEST_DATA_PATH):
    print("Error: 테스트 데이터가 없습니다.")
    exit(1)

with open(TEST_DATA_PATH, 'r') as f:
    test_cases = json.load(f)

results = []
print(f"\n>>> 총 {len(test_cases)}개 케이스 평가 시작 (멀티턴 자동응답 포함)...\n")

for case in tqdm(test_cases):
    # 1. 초기 질문 설정
    query = case['query']
    target = case['target_product']
    exp_profile = case['expected_profile']
    
    # 2. 상태 초기화
    history = []
    curr_profile = {}
    curr_ask_count = 0
    
    final_res = ""
    candidates = []
    # [수정] 로그를 단순 리스트가 아니라 구조화된 딕셔너리 리스트로 저장
    log_history = [] 

    user_context_answer = f"내 정보는 {json.dumps(exp_profile, ensure_ascii=False)} 야."

    for turn in range(3): 
        
        # 턴별 시작 시간 등은 생략하지만, 입력 쿼리는 기록
        turn_input_query = query 
        
        inputs = {
            "messages": history, 
            "user_query": query, 
            "profile": curr_profile,
            "ask_count": curr_ask_count,
            "missing_info": [], 
            "candidates": [],
            "thinking_process": [] 
        }
        
        try:
            output = app.invoke(inputs)
            
            final_res = output.get('final_response', "")
            curr_profile = output.get('profile', {})
            curr_ask_count = output.get('ask_count', 0)
            candidates = output.get('candidates', [])
            turn_logs = output.get('thinking_process', [])
            
            # [상세 로깅] 턴별 상세 정보 저장
            log_history.append({
                "turn": turn + 1,
                "input_query": turn_input_query,
                "agent_response": final_res,
                "extracted_profile": curr_profile.copy(),
                "found_candidates": [c.get('product_name') for c in candidates], # 상품명만 추출
                "internal_logs": turn_logs
            })
            
            has_response_step = any("Response" in log for log in turn_logs)
            
            if has_response_step or candidates:
                break 
            
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=final_res))
            
            query = user_context_answer 
            
        except Exception as e:
            print(f"Error in Case {case['id']}: {e}")
            final_res = f"Error: {str(e)}"
            log_history.append({"turn": turn+1, "error": str(e)})
            break

    # 3. 최종 평가
    try:
        prof_score = calculate_profile_score(exp_profile, curr_profile)
        is_hit = check_retrieval_hit(target, candidates)
        qual_score = evaluate_response_quality(case['query'], target, case['evaluation_points'], final_res)
        
        results.append({
            "id": case['id'],
            "category": case['category'],
            "initial_query": case['query'],
            "target_product": target,
            "expected_profile": exp_profile,
            
            # 평가 지표
            "turns_taken": turn + 1,
            "profile_score": prof_score,
            "retrieval_hit": is_hit,
            "quality_score": qual_score,
            
            # 최종 결과물
            "final_agent_response": final_res,
            "final_extracted_profile": curr_profile,
            "final_candidates": [c.get('product_name') for c in candidates],
            
            # [Full Log] 전체 대화 및 사고 과정
            "full_interaction_log": log_history 
        })
        
    except Exception as e:
        results.append({"id": case['id'], "error": str(e)})


# ==========================================
# 3. 결과 저장
# ==========================================
df = pd.DataFrame(results)

if 'error' not in df.columns:
    df['error'] = None
    
success_df = df[df['error'].isna()].copy()

print("\n" + "="*50)
print("📊 [평가 완료]")

if len(success_df) > 0:
    # CSV는 요약 정보만 저장 (로그가 너무 길어서 CSV 포맷 깨짐 방지)
    # full_interaction_log 컬럼은 제외하거나 문자열로 변환해서 저장
    csv_df = success_df.drop(columns=['full_interaction_log'])
    csv_df.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8-sig')
    
    # JSON에는 모든 상세 정보 포함
    success_df.to_json(RESULT_JSON_PATH, orient='records', force_ascii=False, indent=2)
    
    print(f"1. 요약 결과 CSV: {RESULT_CSV_PATH}")
    print(f"2. 상세 로그 JSON: {RESULT_JSON_PATH}")
    print(f"3. 평균 점수: 품질 {success_df['quality_score'].mean():.1f}, 프로필 {success_df['profile_score'].mean():.1f}")
else:
    print("실패한 테스트만 존재합니다.")
