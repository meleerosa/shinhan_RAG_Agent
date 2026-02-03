import json
import os
import random
import re

# ==========================================
# 0. 설정
# ==========================================
INPUT_PATH = "/home/wlaud/projects/shinhan/data/sample_extracted_200.json"
OUTPUT_PATH = "/home/wlaud/projects/shinhan/data/test_golden_20.json"
TARGET_COUNT = 20

# 카테고리별 목표 할당량 (총 20개)
# Savings(적금)은 흔하므로 조금 더, 나머지는 균등하게
QUOTA = {
    "Savings": 6,
    "Investment": 4,
    "Demand": 3,
    "Insurance": 3,
    "Retirement": 4
}

# ==========================================
# 1. 데이터 로드 및 그룹화
# ==========================================
if not os.path.exists(INPUT_PATH):
    print("Error: 추출된 데이터 파일이 없습니다.")
    exit(1)

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 에러 없는 데이터만 필터링
valid_data = [d for d in data if "error" not in d]

# 카테고리별로 데이터 분류
categorized = {cat: [] for cat in QUOTA.keys()}
for item in valid_data:
    cat = item.get('category', 'Savings') # 기본값 Savings
    # 카테고리 매칭 (부분 문자열 포함 확인)
    for key in categorized.keys():
        if key in cat:
            categorized[key].append(item)
            break

# ==========================================
# 2. 다양성 기반 선별 (Diversity Selection)
# ==========================================
selected_items = []

print(f">>> 카테고리별 다양성 선별 시작 (목표: {TARGET_COUNT}개)")

for cat, items in categorized.items():
    quota = QUOTA.get(cat, 3)
    
    # 1. 'diversity_tag' 기준으로 유니크한 것들 우선 선발
    # (diversity_tag가 없으면 product_name을 사용)
    items_by_tag = {}
    for item in items:
        tag = item.get('diversity_tag', 'General')
        if tag not in items_by_tag:
            items_by_tag[tag] = []
        items_by_tag[tag].append(item)
    
    # 각 태그에서 1개씩 뽑기
    picked_for_cat = []
    tags = list(items_by_tag.keys())
    random.shuffle(tags) # 태그 순서 섞기
    
    for tag in tags:
        if len(picked_for_cat) >= quota: break
        # 해당 태그 내에서 랜덤 1개 선택
        picked_for_cat.append(random.choice(items_by_tag[tag]))
        
    # 만약 할당량을 못 채웠으면(태그 종류가 적어서), 남은 것 중 랜덤 추가
    if len(picked_for_cat) < quota:
        remaining = [i for i in items if i not in picked_for_cat]
        needed = quota - len(picked_for_cat)
        if len(remaining) >= needed:
            picked_for_cat.extend(random.sample(remaining, needed))
        else:
            picked_for_cat.extend(remaining)
            
    selected_items.extend(picked_for_cat)
    print(f" - {cat}: {len(picked_for_cat)}개 선택 (Tags: {[i.get('diversity_tag', 'Gen') for i in picked_for_cat]})")

# ==========================================
# 3. 가상 질문 및 정답 생성 (Synthetic Query Generation)
# ==========================================
final_test_set = []

def generate_synthetic_case(item):
    """상품 정보를 바탕으로 사용자의 질문과 기대 정답을 생성"""
    cat = item.get('category', '')
    name = item.get('product_name', '')
    
    # 1. 프로필 정보 추출 (Eligibility 기반)
    elig = item.get('eligibility', {})
    age = elig.get('age_min', 30) # 기본 30세
    if age < 20: age = 25 # 너무 어리면 25세로 설정
    
    job = "직장인"
    if "사업자" in str(elig): job = "개인사업자"
    if "군인" in str(elig): job = "군인"
    if "청년" in str(elig): age = 27
    if "시니어" in str(elig): age = 60
    
    # 2. 질문 생성 (템플릿 기반)
    query = ""
    goal = ""
    
    # 카테고리별 질문 패턴
    if "Savings" in cat:
        goal = "목돈 마련"
        feature = ""
        if "자유" in item.get('accumulation_type', ''): feature = "자유롭게 넣을 수 있는"
        query = f"{age}살 {job}인데 {feature} 적금 추천해줘. {goal}하고 싶어."
        
    elif "Investment" in cat:
        goal = "자산 증식"
        risk = "공격적" if (item.get('risk_grade', 6) or 6) <= 2 else "안전"
        asset = item.get('underlying_asset', '상품')
        query = f"{asset} 같은데 투자하고 싶은데 {risk}인 성향이야. 수익률 좋은거 있어?"
        
    elif "Demand" in cat:
        goal = "비상금 관리"
        query = f"이자 매일 주는 파킹통장이나 수수료 없는 입출금 통장 추천해줘."
        
    elif "Insurance" in cat:
        goal = "위험 대비"
        query = f"{age}세인데 보험 하나 들려고 해. {item.get('insurance_type', '보장성')} 추천좀."
        
    elif "Retirement" in cat:
        goal = "노후 대비"
        query = f"연말정산 세액공제 받고 싶은데 퇴직연금 {item.get('pension_type', 'IRP')} 상품 알려줘."

    # 3. 평가 포인트 (Must Include)
    eval_points = [name] # 상품명은 무조건 나와야 함
    if item.get('max_deposit'): eval_points.append("월 한도")
    if item.get('risk_caution'): eval_points.append("유의사항")
    if item.get('join_channel', {}).get('is_online'): eval_points.append("비대면")
    
    return {
        "id": f"TEST_{random.randint(1000, 9999)}",
        "category": cat,
        "query": query,
        "target_product": name,
        "expected_profile": {
            "age": age,
            "job": job,
            "financial_goal": goal
        },
        "evaluation_points": eval_points,
        "raw_data_summary": item.get('summary', '')[:50] + "..."
    }

print("\n>>> 테스트 케이스 생성 중...")
for item in selected_items:
    test_case = generate_synthetic_case(item)
    final_test_set.append(test_case)

# ==========================================
# 4. 저장
# ==========================================
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(final_test_set, f, indent=2, ensure_ascii=False)

print(f"\n>>> 골든 테스트셋 생성 완료: {OUTPUT_PATH}")
print(f"    총 {len(final_test_set)}개 케이스")
print("    이제 'evaluate_agent.py'를 만들어 이 데이터로 평가하면 됩니다.")
