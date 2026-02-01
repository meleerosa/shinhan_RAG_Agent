import os
import re
import pandas as pd

# 설정: PDF 파일 경로
base_paths = {
    "susin": "/home/wlaud/projects/shinhan/shinhan_pdfs_susin",
    "yeon": "/home/wlaud/projects/shinhan/shinhan_pdfs_yeon"
}

def classify_product_improved(filename, source_folder):
    name = filename.lower()
    
    # 1. 퇴직연금 (폴더가 yeon이거나 명확한 키워드)
    if source_folder == "yeon" or re.search(r'(퇴직|irp|db|dc)', name):
        return "Retirement_Pension (퇴직연금)"
    
    # 2. 보험 (방카슈랑스, 연금보험) - 예금자보호 문구와 혼동 주의
    if re.search(r'(생명|변액|연금보험|저축보험|방카|보험)', name) and not re.search(r'(예금자보호)', name):
        return "Insurance (보험/방카)"

    # 3. 투자/ISA/골드 (위험자산, 실적배당)
    # 골드/실버/리슈, 채권, 어음, CMA, RP, 개인종합자산관리계좌(ISA)
    if re.search(r'(isa|개인종합자산관리계좌|일임형|계약권유문서|투자|etf|펀드|신탁|채권|어음|cma|rp|환매조건부|골드|실버|금거래|금매매|리슈)', name):
        return "Investment (투자/ISA/골드)"

    # 4. 입출금 (유동성, 파킹통장)
    # 저금통, 머니박스, Box, 거래계좌 등
    # 단, '청년도약계좌' 같은 적금은 예외 처리 필요하나 아래 Savings가 먼저 잡지 못하면 여기서 잡힐 수 있음. 
    # 순서상 Savings가 뒤에 있으면 '계좌' 때문에 여기가 먼저 잡힘. 
    # -> 전략: '계좌'가 포함되더라도 '적금' 성격 키워드가 없어야 함.
    if re.search(r'(통장|입출금|수시|보통예금|저금통|머니박스|box|계좌|거래계좌)', name):
        # 예외: 이름에 계좌가 들어가는 적금 상품 방지
        if re.search(r'(청년도약|적금|정기|모아)', name):
             return "Savings (예적금)"
        return "Demand_Deposit (입출금)"

    # 5. 예적금 (저축성)
    # 프리미어세이프(ELD), 모아모아, 청년도약
    if re.search(r'(예금|적금|청약|저축|세이프|모아모아)', name):
        return "Savings (예적금)"
        
    return "Unclassified (기타)"

data = []

for folder_key, path in base_paths.items():
    if not os.path.exists(path):
        continue
        
    files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    
    for f in files:
        category = classify_product_improved(f, folder_key)
        data.append({
            "folder": folder_key,
            "filename": f,
            "category": category
        })

# 결과 확인 및 저장
df = pd.DataFrame(data)
if not df.empty:
    print("=== 개선된 분류 결과 집계 ===")
    print(df.groupby('category').size())
    df.to_csv("classified_file_list_v2.csv", index=False)
