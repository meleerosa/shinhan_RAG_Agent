import json
import os
import random

# ==========================================
# 0. 설정
# ==========================================
INPUT_PATH = "/home/wlaud/projects/shinhan/data/sample_extracted_200.json"
OUTPUT_PATH = "/home/wlaud/projects/shinhan/data/test_golden_multiturn_20.json"
TARGET_COUNT = 20

QUOTA = {
    "Savings": 6, "Investment": 4, "Demand": 3, "Insurance": 3, "Retirement": 4
}

# ==========================================
# 1. 데이터 로드 및 그룹화 (기존과 동일)
# ==========================================
if not os.path.exists(INPUT_PATH):
    print("Error: 추출된 데이터 파일이 없습니다.")
    exit(1)

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

valid_data = [d for d in data if "error" not in d]
categorized = {cat: [] for cat in QUOTA.keys()}

for item in valid_data:
    cat = item.get('category', 'Savings')
    for key in categorized.keys():
        if key in cat:
            categorized[key].append(item)
            break

# ==========================================
# 2. 다양성 선별 (기존과 동일)
# ==========================================
selected_items = []
print(f">>> 카테고리별 다양성 선별 시작 (목표: {TARGET_COUNT}개)")

for cat, items in categorized.items():
    quota = QUOTA.get(cat, 3)
    items_by_tag = {}
    for item in items:
        tag = item.get('diversity_tag', 'General')
        if tag not in items_by_tag: items_by_tag[tag] = []
        items_by_tag[tag].append(item)
    
    picked_for_cat = []
    tags = list(items_by_tag.keys())
    random.shuffle(tags)
    
    for tag in tags:
        if len(picked_for_cat) >= quota: break
        picked_for_cat.append(random.choice(items_by_tag[tag]))
            
    if len(picked_for_cat) < quota:
        remaining = [i for i in items if i not in picked_for_cat]
        needed = quota - len(picked_for_cat)
        if len(remaining) >= needed:
            picked_for_cat.extend(random.sample(remaining, needed))
        else:
            picked_for_cat.extend(remaining)
            
    selected_items.extend(picked_for_cat)

# ==========================================
# 3. 멀티턴 시나리오 생성 (핵심 수정)
# ==========================================
final_test_set = []

def generate_multiturn_case(item):
    cat = item.get('category', '')
    name = item.get('product_name', '')
    
    # 1. 프로필 설정
    elig = item.get('eligibility', {})
    age = elig.get('age_min')
    if age is None: age = 30
    if age < 20: age = 25
    
    job = "직장인"
    if "사업자" in str(elig): job = "개인사업자"
    if "청년" in str(elig): age = 27
    
    # 2. 시나리오 타입 랜덤 결정 (단일 턴 vs 멀티 턴)
    # 30% 확률로 완벽한 질문(Single turn), 70% 확률로 정보 누락 질문(Multi turn)
    is_complete_query = random.random() < 0.3
    
    query = ""
    missing_info_answer = "" # 누락된 정보에 대한 사용자 답변
    
    # 카테고리별 템플릿
    if "Savings" in cat:
        goal = "목돈 마련"
        if is_complete_query:
            query = f"{age}살 {job}이고 월 50만원씩 저축하려는데 적금 추천해줘."
        else:
            query = "적금 하나 추천해줘." # 나이, 금액 누락
            missing_info_answer = f"{age}살이고 직장 다녀. 월 50만원 정도 생각 중이야."
            
    elif "Investment" in cat:
        goal = "자산 증식"
        asset = item.get('underlying_asset') or '펀드'
        if is_complete_query:
            query = f"{asset} 투자하고 싶은데 공격적인 성향이야. 수익률 좋은거 추천좀."
        else:
            query = f"{asset} 투자 상품 추천해줘." # 성향 누락
            missing_info_answer = "수익률 높은게 좋아. 공격적인 편이야."
            
    elif "Demand" in cat:
        goal = "비상금 관리"
        if is_complete_query:
            query = "매일 이자 주는 파킹통장 추천해줘."
        else:
            query = "입출금 통장 추천해줘." # 목적/특징 누락
            missing_info_answer = "비상금 넣어두려고 해. 이자 자주 주는 걸로."

    elif "Insurance" in cat:
        goal = "위험 대비"
        if is_complete_query:
            query = f"{age}세인데 암보험이나 건강보험 추천해줘."
        else:
            query = "보험 상품 알아보고 있어." # 나이/종류 누락
            missing_info_answer = f"{age}살이고 건강 관련 보장받고 싶어."

    elif "Retirement" in cat:
        goal = "노후 대비"
        if is_complete_query:
            query = "연말정산 세액공제용 IRP 상품 알려줘."
        else:
            query = "퇴직연금 상품 뭐 있어?" # 목적 누락
            missing_info_answer = "세액공제 혜택 받고 싶어서 그래."
    
    # 3. 평가 포인트
    eval_points = [name]
    if item.get('max_deposit'): eval_points.append("한도")
    
    # 4. 구조화된 멀티턴 데이터 반환
    return {
        "id": f"TEST_{random.randint(1000, 9999)}",
        "category": cat,
        "target_product": name,
        
        # [핵심] 시나리오 흐름 정의
        "scenario_type": "Single-turn" if is_complete_query else "Multi-turn",
        "initial_query": query,             # 1. 첫 질문
        "expected_followup": missing_info_answer, # 2. 에이전트가 되물었을 때 할 답변
        
        "expected_profile": {
            "age": age,
            "job": job,
            "financial_goal": goal
        },
        "evaluation_points": eval_points
    }

print("\n>>> 멀티턴 테스트 케이스 생성 중...")
for item in selected_items:
    test_case = generate_multiturn_case(item)
    final_test_set.append(test_case)

# ==========================================
# 4. 저장
# ==========================================
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(final_test_set, f, indent=2, ensure_ascii=False)

print(f"\n>>> 골든 테스트셋 생성 완료: {OUTPUT_PATH}")
print(f"    총 {len(final_test_set)}개 케이스 (Single/Multi 혼합)")
