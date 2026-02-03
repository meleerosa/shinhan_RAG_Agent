import pandas as pd
import numpy as np
import os
import re
from collections import Counter

# ==========================================
# 0. 설정
# ==========================================
INPUT_PATH = "/home/wlaud/projects/shinhan/data/pdf_raw_text_data.pkl"
OUTPUT_PATH = "/home/wlaud/projects/shinhan/data/selected_diverse_200.pkl"
TARGET_COUNT = 200  # 최종 추출할 데이터 개수

# ==========================================
# 1. 데이터 로드 및 기초 분석
# ==========================================
if not os.path.exists(INPUT_PATH):
    print("Error: 파일이 없습니다.")
    exit(1)

df = pd.read_pickle(INPUT_PATH)
print(f"Initial Data Count: {len(df)}")

# 텍스트 길이 분석
df['text_len'] = df['raw_text'].apply(lambda x: len(str(x)))
print(f"Avg Length: {df['text_len'].mean():.0f}, Max Length: {df['text_len'].max()}")

# 1차 필터링: 너무 짧은 데이터(에러/목차 등) 제거
# 예: 300자 미만은 정보가 없다고 판단
valid_df = df[df['text_len'] > 300].copy()
print(f"Valid Data (>300 chars): {len(valid_df)}")

# ==========================================
# 2. 다양성 분석 (Feature Engineering)
# ==========================================
# LLM 없이, 키워드로 데이터의 성격을 규정합니다.

def get_diversity_tags(text):
    tags = []
    
    # [Target] 누구를 위한 것인가?
    if re.search(r"청년|20대|대학생|사회초년생", text): tags.append("Youth")
    elif re.search(r"시니어|고령|노후|연금수령", text): tags.append("Senior")
    elif re.search(r"군인|장병|국군", text): tags.append("Soldier")
    elif re.search(r"사업자|기업|법인|SOHO", text): tags.append("Business")
    elif re.search(r"직장인|급여", text): tags.append("Worker")
    else: tags.append("General") # 일반
    
    # [Feature] 어떤 특징이 있는가?
    if re.search(r"주택|청약|아파트", text): tags.append("Housing")
    if re.search(r"비과세|세제혜택|소득공제", text): tags.append("TaxFree")
    if re.search(r"파킹|매일이자|수시", text): tags.append("Parking")
    if re.search(r"ETF|펀드|TDF|투자", text): tags.append("Invest")
    if re.search(r"IRP|퇴직연금|DC|DB", text): tags.append("Pension")
    if re.search(r"여행|환전|외화", text): tags.append("Travel")
    
    return "_".join(sorted(tags))

print("\n>>> 태그 분석 중...")
valid_df['diversity_tag'] = valid_df['raw_text'].apply(get_diversity_tags)

# 태그 결합: 카테고리 + 태그 (예: Savings_Youth_Housing)
valid_df['cluster_id'] = valid_df['category'] + "_" + valid_df['diversity_tag']

# 분포 확인
cluster_counts = valid_df['cluster_id'].value_counts()
print(f"\n[데이터 군집 분석 결과] 총 {len(cluster_counts)}개의 고유한 상품 유형 발견")
print(cluster_counts.head(10))  # 가장 흔한 유형
print("...")
print(cluster_counts.tail(5))   # 가장 희귀한 유형

# ==========================================
# 3. 다양성 기반 추출 (Stratified Sampling)
# ==========================================
# 전략: 모든 고유한 'cluster_id'에서 최소 1개씩은 무조건 뽑는다.
# 남는 자리는 비율대로 채운다.

selected_indices = []
clusters = valid_df['cluster_id'].unique()

print(f"\n>>> 추출 전략: {len(clusters)}개 유형에서 대표 샘플 우선 확보...")

# [Step 1] 희소성 보존: 각 클러스터에서 1개씩 무작위 추출
for cluster in clusters:
    subset = valid_df[valid_df['cluster_id'] == cluster]
    # 각 유형별 대표 선수 1명 선발 (랜덤)
    picked = subset.sample(n=1, random_state=42)
    selected_indices.extend(picked.index.tolist())

current_count = len(selected_indices)
print(f" -> 1차 추출 완료: {current_count}개 (다양성 100% 보장)")

# [Step 2] 나머지 채우기 (인기 카테고리 비율 반영)
if current_count < TARGET_COUNT:
    needed = TARGET_COUNT - current_count
    
    # 이미 뽑힌거 제외
    remaining_pool = valid_df.drop(selected_indices)
    
    # 카테고리별 가중치 (투자와 연금 데이터를 좀 더 확보하고 싶다면 weight 조절)
    weights = {
        'Savings': 1.0, 
        'Investment': 1.5, # 투자 상품은 복잡하니 더 많이
        'Insurance': 1.2, 
        'Retirement': 1.5, # 연금도 복잡함
        'Demand': 0.8      # 입출금은 단순함
    }
    
    # 가중치 적용을 위한 확률 계산
    remaining_pool['weight'] = remaining_pool['category'].map(weights).fillna(1.0)
    
    # 가중치 기반 랜덤 샘플링
    extra_samples = remaining_pool.sample(
        n=min(len(remaining_pool), needed), 
        weights='weight', 
        random_state=42
    )
    selected_indices.extend(extra_samples.index.tolist())

# ==========================================
# 4. 결과 저장 및 검증
# ==========================================
final_df = valid_df.loc[selected_indices].copy()

# 셔플
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n" + "="*50)
print(f"최종 추출 데이터: {len(final_df)}개")
print("="*50)
print("[카테고리별 분포]")
print(final_df['category'].value_counts())

print("\n[주요 다양성 태그 포함 여부]")
check_tags = ['Youth', 'Senior', 'Soldier', 'Business', 'Housing', 'TaxFree', 'Invest', 'Pension']
for tag in check_tags:
    count = final_df['diversity_tag'].str.contains(tag).sum()
    print(f" - {tag}: {count}개")

# 저장
final_df.to_pickle(OUTPUT_PATH)
print(f"\n>>> 저장 완료: {OUTPUT_PATH}")
print("이제 'extract_sample_json.py'에서 이 파일을 로드해서 쓰세요.")
