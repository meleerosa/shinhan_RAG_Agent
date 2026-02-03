import pandas as pd
import json
import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional

num_data = 10
# ==========================================
# 0. API 키 및 LLM 초기화 (최우선 실행)
# ==========================================
# 환경변수 확인
if "OPENAI_API_KEY" not in os.environ:
    print("Error: 'OPENAI_API_KEY' 환경변수가 설정되지 않았습니다.")
    print("export OPENAI_API_KEY='sk-...' 명령어로 설정해주세요.")
    # 테스트를 위해 코드 내에서 임시 설정하려면 아래 주석 해제 (보안 주의)
    # os.environ["OPENAI_API_KEY"] = "..."

try:
    # 전역 LLM 객체 생성 (온도 0으로 설정하여 일관된 결과 유도)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print(">>> LLM(ChatOpenAI) 초기화 성공")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

# ==========================================
# 1. 데이터 로드 (전체 텍스트 사용)
# ==========================================
pkl_path = "/home/wlaud/projects/shinhan/data/pdf_raw_text_data.pkl"

if os.path.exists(pkl_path):
    df = pd.read_pickle(pkl_path)
    sample_df = df.head(num_data).copy()
    print(f"데이터 로드 완료: 총 {len(df)}개 중 {num_data}개 샘플 추출")
else:
    print(f"파일을 찾을 수 없습니다. {pkl_path}")

# ==========================================
# 2. 스키마 정의 (프로젝트 최적화 버전)
# ==========================================



# ==========================================
# 0. 공통 Enum 및 서브 모델
# ==========================================
# 가입 채널 표준화
class JoinChannel(BaseModel):
    is_online: bool = Field(False, description="비대면(앱/인터넷) 가입 가능 여부")
    is_offline: bool = Field(False, description="영업점 방문 가입 가능 여부")
    descriptions: List[str] = Field(default_factory=list, description="채널별 특이사항 (예: 앱 가입시 우대금리)")

# ==========================================
# A. Savings (예적금) - [보강 포인트: 적립 방식 & 만기 처리]
# ==========================================
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
    
    # [Detail 1] 적립 방식 (중요: 정기 vs 자유)
    # 정기적금: 매월 같은 날 같은 돈 (강제성) / 자유적금: 아무때나 (유연성)
    accumulation_type: Optional[str] = Field(None, description="'정기적립식'(정해진 금액) 또는 '자유적립식'(자유롭게 입금) 구분")
    
    eligibility: Eligibility # (기존 Eligibility 클래스 사용)
    interest_rate: InterestRate # (기존 InterestRate 클래스 사용)
    
    min_month: Optional[int] = Field(None, description="최소 가입개월")
    max_month: Optional[int] = Field(None, description="최대 가입개월")
    max_deposit: Optional[int] = Field(None, description="월 납입 한도(원)")
    
    # [Detail 2] 만기 자동 재예치 (귀차니즘 고객용)
    auto_renewal: bool = Field(False, description="만기 시 자동 재예치/자동 연장 가능 여부")
    
    # [Detail 3] 분할 해지 (유동성 확보)
    partial_withdrawal: bool = Field(False, description="만기 전 일부 금액만 해지 가능 여부(긴급출금)")
    
    join_channel: JoinChannel
    risk_caution: Optional[str] = Field(None, description="유의사항")
    keywords: List[str] = Field(default_factory=list, description="추천 키워드 (예: #사회초년생 #목돈마련 #여행자금)")

# ==========================================
# B. Investment (투자) - [보강 포인트: 배당 주기 & 기초 자산]
# ==========================================
class InvestmentSchema(BaseModel):
    product_name: str
    category_detail: str = Field(..., description="ISA, ETF, 펀드, 골드, 채권, ELS/DLS")
    
    # [Detail 1] 기초 자산 (무엇에 투자하는가?)
    underlying_asset: Optional[str] = Field(None, description="기초자산 (예: KOSPI200, 금, 미국채, 삼성전자)")
    
    # [Detail 2] 이익 지급 방식 (월급처럼 받고 싶은가?)
    payout_cycle: Optional[str] = Field(None, description="이익 지급 주기 (예: 월지급식, 만기일시지급, 재투자)")
    
    risk_grade: Optional[int] = Field(None, description="위험등급 1~6")
    principal_protected: bool = Field(..., description="원금보장 여부")
    loss_warning: Optional[str] = Field(None, description="원금손실 경고")
    
    total_fee_rate: Optional[float] = Field(None, description="총 보수율 %")
    fee_detail: Optional[str] = Field(None, description="선취/후취/환매 수수료 상세")
    
    recommended_period: Optional[str] = Field(None, description="권장 투자 기간")

# ==========================================
# C. Demand Deposit (입출금) - [보강 포인트: 이자 지급일 & 부가 혜택]
# ==========================================
class DemandSchema(BaseModel):
    product_name: str
    
    # [Detail 1] 이자 지급 시기 (파킹통장족에게 중요)
    interest_payment_date: Optional[str] = Field(None, description="이자 결산/지급 시기 (예: 매월 셋째주 금요일, 매일)")
    
    base_interest_rate: Optional[float] = Field(None, description="기본 금리 %")
    max_interest_rate: Optional[float] = Field(None, description="조건 충족 시 최고 금리 % (파킹통장용)")
    
    transfer_fee_waiver: Optional[str] = Field(None, description="이체 수수료 면제 조건")
    withdraw_fee_waiver: Optional[str] = Field(None, description="출금 수수료 면제 조건")
    
    # [Detail 2] 환전/해외 혜택 (여행족 타겟)
    currency_benefit: Optional[str] = Field(None, description="환율 우대 또는 해외 이용 수수료 혜택")

# ==========================================
# D. Insurance (보험) - [보강 포인트: 갱신 여부 & 보장 요약]
# ==========================================
class InsuranceSchema(BaseModel):
    product_name: str
    insurance_type: str = Field(..., description="저축성, 보장성, 연금형")
    
    # [Detail 1] 갱신형 여부 (보험료 인상 폭탄 방지)
    is_renewal: Optional[bool] = Field(None, description="갱신형(보험료 변동) 여부. 비갱신형이면 False")
    
    declared_rate: Optional[float] = Field(None, description="공시이율 %")
    min_guarantee_rate: Optional[float] = Field(None, description="최저보증이율 %")
    
    # [Detail 2] 주요 보장 내용 (저축성이라도 사망보장 등이 섞여있음)
    coverage_summary: Optional[str] = Field(None, description="주요 보장 내용 요약 (예: 재해사망시 1천만원)")
    
    tax_benefit_cond: Optional[str] = Field(None, description="비과세 조건")
    early_termination_warning: Optional[str] = Field(None, description="해지환급금 유의사항")

# ==========================================
# E. Pension (퇴직연금) - [보강 포인트: 디폴트옵션 & 수령 방법]
# ==========================================
class PensionSchema(BaseModel):
    product_name: str
    pension_type: str = Field(..., description="DC, DB, IRP")
    
    # [Detail 1] 디폴트 옵션 (사전지정운용제도)
    default_option_info: Optional[str] = Field(None, description="디폴트옵션(사전지정운용) 적용 가능 여부 및 내용")
    
    tax_deduction_limit: Optional[str] = Field(None, description="세액공제 한도")
    
    # [Detail 2] 연금 수령 방식
    pension_receipt_method: Optional[str] = Field(None, description="연금 수령 조건 (예: 55세 이후, 10년 이상 가입)")
    
    investment_options: Optional[str] = Field(None, description="운용 가능 상품군")
    fees: Optional[str] = Field(None, description="운용/자산관리 수수료율")



# ==========================================
# 3. LLM 체인 생성 함수
# ==========================================
def get_chain(category):
    # 전역변수 llm 사용
    global llm 
    
    if "Savings" in category:
        schema = SavingsSchema
        sys_msg = """
        당신은 은행 예적금 상품 분석 전문가입니다. 설명서 텍스트에서 다음 정보를 정밀하게 추출하세요.
        
        1. [적립방식]: '자유적립식'인지 '정기적립식'인지 반드시 구분하세요. (매월 동일 금액 vs 자유 납입)
        2. [만기/해지]: '자동 재예치' 기능이 있는지, 만기 전 '긴급출금(일부해지)'이 가능한지 찾으세요.
        3. [가입채널]: 영업점 방문 필수인지, 앱(SOL) 전용인지 구분하여 'join_channel'에 담으세요.
        4. [금리]: 기본금리와 최고금리를 분리하고, 우대조건(급여이체 등)을 상세히 적으세요.
        """
        
    elif "Investment" in category:
        schema = InvestmentSchema
        sys_msg = """
        당신은 투자 상품 분석가입니다. 고객의 원금 손실 위험을 알리기 위해 다음을 추출하세요.
        
        1. [기초자산]: 이 상품이 무엇(KOSPI, 금, 채권 등)에 투자하는지 'underlying_asset'에 적으세요.
        2. [지급방식]: 이익을 '월지급'하는지, '만기일시지급'하는지 확인하세요.
        3. [리스크]: 위험등급(1~6)을 숫자로 추출하고, '원금 비보장' 문구를 찾아 'loss_warning'에 요약하세요.
        4. [수수료]: 총 보수율과 선취/환매 수수료 정보를 놓치지 마세요.
        """
        
    elif "Demand" in category:
        schema = DemandSchema
        sys_msg = """
        당신은 입출금/파킹통장 분석가입니다. 
        
        1. [이자지급]: 이자가 '매일' 지급되는지, '매월' 지급되는지 'interest_payment_date'에 명시하세요.
        2. [파킹기능]: 고금리 적용 한도(예: 1억원까지 연 2%)가 있다면 'max_interest_rate'와 함께 적으세요.
        3. [수수료]: 이체/출금 수수료 면제 조건(횟수제한, 무제한 등)을 정확히 추출하세요.
        """
        
    elif "Insurance" in category:
        schema = InsuranceSchema
        sys_msg = """
        당신은 보험 상품 분석가입니다. 저축성 보험으로 오해하지 않도록 다음을 추출하세요.
        
        1. [갱신여부]: 갱신형인지 비갱신형인지 확인하여 'is_renewal'에 담으세요.
        2. [사업비/해지]: '조기 해지 시 원금 손실 가능성'에 대한 경고 문구를 반드시 찾으세요.
        3. [금리]: 공시이율과 최저보증이율을 구분하세요.
        """
        
    elif "Retirement" in category:
        schema = PensionSchema
        sys_msg = """
        당신은 퇴직연금(IRP/DC) 전문가입니다.
        
        1. [디폴트옵션]: 사전지정운용제도(디폴트옵션) 관련 내용이 있으면 추출하세요.
        2. [수령방법]: 연금을 언제(55세 이후), 어떻게(분할/일시) 받을 수 있는지 확인하세요.
        3. [세제혜택]: 세액공제 한도 금액을 정확히 추출하세요.
        """
        
    else:
        # 기타 카테고리 (기본적으로 Savings로 처리하되 범용 프롬프트 사용)
        schema = SavingsSchema
        sys_msg = "금융상품 설명서입니다. 상품명, 가입조건, 금리, 리스크 정보를 상세히 추출하세요."
    
    return ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("human", "{text}"),
    ]) | llm.with_structured_output(schema)


# ==========================================
# 4. 실행 및 결과 저장
# ==========================================
results = []
output_file = f"/home/wlaud/projects/shinhan/data/sample_extracted_{num_data}.json"

print(f">>> {num_data}개 샘플 데이터 추출 시작 (전체 텍스트 사용)...")

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    try:
        # 체인 생성
        chain = get_chain(row['category'])
        
        # 텍스트 길이 제한 (토큰 비용 및 에러 방지)
        full_text = row['raw_text'][:15000]
        
        # LLM 호출
        extracted = chain.invoke({"text": full_text})
        
        # 결과 저장
        data = extracted.model_dump()
        data['filename'] = row['filename']
        data['category'] = row['category']
        results.append(data)
        
    except Exception as e:
        # 어떤 파일에서 어떤 에러가 났는지 출력
        print(f"Error extracting {row['filename']}: {e}")
        results.append({"filename": row['filename'], "error": str(e)})

# 결과 파일 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n>>> 추출 완료! '{output_file}' 파일을 확인하세요.")
