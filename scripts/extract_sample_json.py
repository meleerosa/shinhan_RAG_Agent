import pandas as pd
import json
import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# 0. API 키 및 설정 (최우선 실행)
# ==========================================
if "OPENAI_API_KEY" not in os.environ:
    print("Error: 'OPENAI_API_KEY' 환경변수가 설정되지 않았습니다.")
    exit(1)

try:
    # 전역 LLM 객체 생성 (온도 0으로 설정하여 일관된 결과 유도)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print(">>> LLM(ChatOpenAI) 초기화 성공")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

# ==========================================
# 1. 데이터 로드 (다양성 분석이 완료된 파일 사용)
# ==========================================
# [수정] 분석 단계에서 생성한 파일 경로로 변경
pkl_path = "/home/wlaud/projects/shinhan/data/selected_diverse_200.pkl"

# 결과 저장 경로 (200개 전체 저장)
output_file = "/home/wlaud/projects/shinhan/data/sample_extracted_200.json"

if os.path.exists(pkl_path):
    df = pd.read_pickle(pkl_path)
    # [수정] 분석된 데이터는 이미 200개이므로 전체 사용
    sample_df = df.copy()
    print(f"데이터 로드 완료: 총 {len(sample_df)}개 (다양성 분석 완료 데이터)")
else:
    print(f"Error: 파일을 찾을 수 없습니다. ({pkl_path})")
    print("먼저 'analyze_and_select_data.py'를 실행하여 데이터를 생성하세요.")
    exit(1)

# ==========================================
# 2. 스키마 정의 (공통 Enum 및 서브 모델)
# ==========================================
class JoinChannel(BaseModel):
    is_online: bool = Field(False, description="비대면(앱/인터넷) 가입 가능 여부")
    is_offline: bool = Field(False, description="영업점 방문 가입 가능 여부")
    descriptions: List[str] = Field(default_factory=list, description="채널별 특이사항 (예: 앱 가입시 우대금리)")

# A. Savings (예적금)
class Eligibility(BaseModel):
    age_min: Optional[int] = Field(None, description="최소 가입 나이 (없으면 None)")
    age_max: Optional[int] = Field(None, description="최대 가입 나이 (없으면 None)")
    target_detail: Optional[str] = Field(None, description="가입대상 요약 (예: 직장인, 개인사업자, 첫거래 고객)")
    
class InterestRate(BaseModel):
    base_rate: Optional[float] = Field(None, description="기본 금리 %")
    max_rate: Optional[float] = Field(None, description="최고 우대 금리 %")
    prime_conditions: List[str] = Field(default_factory=list, description="우대 금리 조건 (예: '급여이체 0.5%')")
    
class SavingsSchema(BaseModel):
    product_name: str = Field(..., description="상품명")
    product_type: str = Field(..., description="예금(거치식), 적금(적립식), 청약 중 하나")
    accumulation_type: Optional[str] = Field(None, description="'정기적립식' 또는 '자유적립식' 구분")
    eligibility: Eligibility
    interest_rate: InterestRate
    min_month: Optional[int] = Field(None, description="최소 가입개월")
    max_month: Optional[int] = Field(None, description="최대 가입개월")
    max_deposit: Optional[int] = Field(None, description="월 납입 한도(원)")
    auto_renewal: bool = Field(False, description="만기 시 자동 재예치 가능 여부")
    partial_withdrawal: bool = Field(False, description="만기 전 일부 해지(긴급출금) 가능 여부")
    join_channel: JoinChannel
    risk_caution: Optional[str] = Field(None, description="유의사항")
    keywords: List[str] = Field(default_factory=list, description="추천 키워드 (예: #사회초년생 #목돈마련)")

# B. Investment (투자)
class InvestmentSchema(BaseModel):
    product_name: str
    category_detail: str = Field(..., description="ISA, ETF, 펀드, 골드, 채권, ELS/DLS")
    underlying_asset: Optional[str] = Field(None, description="기초자산 (예: KOSPI200, 금, 미국채)")
    payout_cycle: Optional[str] = Field(None, description="이익 지급 주기 (예: 월지급식)")
    risk_grade: Optional[int] = Field(None, description="위험등급 1~6")
    principal_protected: bool = Field(..., description="원금보장 여부")
    loss_warning: Optional[str] = Field(None, description="원금손실 경고")
    total_fee_rate: Optional[float] = Field(None, description="총 보수율 %")
    fee_detail: Optional[str] = Field(None, description="수수료 상세")
    recommended_period: Optional[str] = Field(None, description="권장 투자 기간")
    join_channel: JoinChannel

# C. Demand (입출금)
class DemandSchema(BaseModel):
    product_name: str
    interest_payment_date: Optional[str] = Field(None, description="이자 지급 시기 (예: 매월, 매일)")
    base_interest_rate: Optional[float] = Field(None, description="기본 금리 %")
    max_interest_rate: Optional[float] = Field(None, description="최고 금리 % (파킹통장용)")
    transfer_fee_waiver: Optional[str] = Field(None, description="이체 수수료 면제 조건")
    withdraw_fee_waiver: Optional[str] = Field(None, description="출금 수수료 면제 조건")
    currency_benefit: Optional[str] = Field(None, description="환율/해외 혜택")
    join_channel: JoinChannel

# D. Insurance (보험)
class InsuranceSchema(BaseModel):
    product_name: str
    insurance_type: str = Field(..., description="저축성, 보장성, 연금형")
    is_renewal: Optional[bool] = Field(None, description="갱신형 여부 (비갱신형이면 False)")
    declared_rate: Optional[float] = Field(None, description="공시이율 %")
    min_guarantee_rate: Optional[float] = Field(None, description="최저보증이율 %")
    coverage_summary: Optional[str] = Field(None, description="주요 보장 내용 요약")
    tax_benefit_cond: Optional[str] = Field(None, description="비과세 조건")
    early_termination_warning: Optional[str] = Field(None, description="해지환급금 유의사항")
    join_channel: JoinChannel

# E. Pension (연금)
class PensionSchema(BaseModel):
    product_name: str
    pension_type: str = Field(..., description="DC, DB, IRP")
    default_option_info: Optional[str] = Field(None, description="디폴트옵션 내용")
    tax_deduction_limit: Optional[str] = Field(None, description="세액공제 한도")
    pension_receipt_method: Optional[str] = Field(None, description="연금 수령 조건")
    investment_options: Optional[str] = Field(None, description="운용 가능 상품군")
    fees: Optional[str] = Field(None, description="수수료율")
    join_channel: JoinChannel

# ==========================================
# 3. LLM 체인 생성 함수
# ==========================================
def get_chain(category):
    global llm 
    
    if "Savings" in category:
        schema = SavingsSchema
        sys_msg = """
        당신은 은행 예적금 상품 분석 전문가입니다. 설명서 텍스트에서 다음 정보를 정밀하게 추출하세요.
        1. [적립방식]: '자유적립식'인지 '정기적립식'인지 구분.
        2. [만기/해지]: 자동 재예치 및 만기 전 긴급출금(일부해지) 가능 여부.
        3. [가입채널]: 영업점 방문 필수인지, 앱(SOL) 전용인지 구분하여 join_channel에 저장.
        4. [금리]: 우대조건(급여이체 등)을 상세히 적으세요.
        """
    elif "Investment" in category:
        schema = InvestmentSchema
        sys_msg = """
        당신은 투자 상품 분석가입니다. 
        1. [기초자산]: 무엇에 투자하는지(KOSPI, 채권 등).
        2. [지급방식]: 월지급식인지 만기일시지급인지.
        3. [리스크]: 위험등급(1~6)과 원금 손실 가능성 경고.
        4. [가입채널]: 비대면 가입 가능 여부 확인.
        """
    elif "Demand" in category:
        schema = DemandSchema
        sys_msg = """
        당신은 입출금/파킹통장 분석가입니다.
        1. [이자지급]: 매일 지급(파킹통장)인지 매월 지급인지 확인.
        2. [수수료]: 이체/출금 수수료 면제 조건 확인.
        3. [가입채널]: 가입 경로 확인.
        """
    elif "Insurance" in category:
        schema = InsuranceSchema
        sys_msg = """
        당신은 보험 상품 분석가입니다.
        1. [갱신여부]: 갱신형/비갱신형 구분.
        2. [해지]: 조기 해지 시 원금 손실 위험 경고.
        3. [가입채널]: 다이렉트(온라인) 상품인지 확인.
        """
    elif "Retirement" in category:
        schema = PensionSchema
        sys_msg = """
        당신은 퇴직연금 전문가입니다.
        1. [디폴트옵션]: 사전지정운용제도 관련 내용.
        2. [수령]: 연금 개시 나이 및 조건.
        3. [세제]: 세액공제 한도 확인.
        """
    else:
        schema = SavingsSchema
        sys_msg = "금융상품 정보를 추출하세요."
    
    return ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("human", "{text}"),
    ]) | llm.with_structured_output(schema)

# ==========================================
# 4. 실행 및 결과 저장
# ==========================================
results = []

print(f">>> 다양성 데이터 200개 추출 시작 (전체 텍스트 사용)...")
print(f"    입력: {pkl_path}")
print(f"    출력: {output_file}")

# [수정] 200개 전체 순회
for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    try:
        # 체인 생성
        chain = get_chain(row['category'])
        
        # 텍스트 길이 제한 (비용 관리용, 필요시 20000 등으로 조정)
        full_text = row['raw_text'][:15000]
        
        # LLM 호출
        extracted = chain.invoke({"text": full_text})
        
        # 결과 저장
        data = extracted.model_dump()
        data['filename'] = row['filename']
        data['category'] = row['category']
        # 분석 단계에서 만든 태그도 같이 저장하면 나중에 유용함
        if 'diversity_tag' in row:
            data['diversity_tag'] = row['diversity_tag']
            
        results.append(data)
        
    except Exception as e:
        print(f"Error extracting {row['filename']}: {e}")
        results.append({"filename": row['filename'], "error": str(e)})

# 결과 파일 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n>>> 추출 완료! 총 {len(results)}개 저장됨.")
print(f"    파일 위치: {output_file}")
