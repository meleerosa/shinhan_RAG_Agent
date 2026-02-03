import os
import json
import faiss
from typing import TypedDict, List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS

# ==========================================
# 0. API 설정
# ==========================================
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    exit(1)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# ==========================================
# 1. 데이터 로드 & FAISS 구축 (Max Info Version)
# ==========================================
json_path = "/home/wlaud/projects/shinhan/data/sample_extracted_10.json"

if not os.path.exists(json_path):
    print("Error: 데이터 파일이 없습니다. extract_sample_json.py를 먼저 실행하세요.")
    raw_data = [] # (테스트 시엔 더미 데이터 사용 권장)
else:
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

documents = []
metadatas = []

print(f">>> 총 {len(raw_data)}개 원본 데이터 로드 및 벡터화 시작")

for item in raw_data:
    if "error" in item: continue
    
    # [데이터 Flattening] 모든 상세 정보를 검색 가능한 자연어 문장으로 변환
    
    # 1. 식별 정보
    idx_info = f"상품명: {item.get('product_name')} / 카테고리: {item.get('category')}"
    
    # 2. 핵심 스펙 (금리, 대상)
    specs = []
    # 금리 처리
    if item.get('interest_rate'):
        ir = item.get('interest_rate', {})
        specs.append(f"기본금리 {ir.get('base_rate')}% 최고금리 {ir.get('max_rate')}%")
        if ir.get('prime_conditions'): specs.append(f"우대조건: {', '.join(ir.get('prime_conditions'))}")
    elif item.get('base_interest_rate'):
        specs.append(f"기본금리 {item.get('base_interest_rate')}%")
    
    # 대상 처리
    elig = item.get('eligibility', {})
    if isinstance(elig, dict):
        specs.append(f"가입대상: {elig.get('target_detail', '누구나')}")
        if elig.get('age_min'): specs.append(f"최소연령 {elig.get('age_min')}세")
        if elig.get('age_max'): specs.append(f"최대연령 {elig.get('age_max')}세")

    # 3. 상세 특징 (Features) - 검색의 핵심
    features = []
    # 가입 채널
    join = item.get('join_channel')
    if isinstance(join, dict):
        if join.get('is_online'): features.append("비대면 앱 가입 가능")
        if join.get('is_offline'): features.append("영업점 방문 필수")
        features.extend(join.get('descriptions', []))
    
    # 적금/예금 특성
    if item.get('accumulation_type'): features.append(item.get('accumulation_type')) # 자유적립식 등
    if item.get('max_deposit'): features.append(f"월납입한도 {item.get('max_deposit')}원")
    if item.get('partial_withdrawal'): features.append("긴급출금 가능")
    
    # 투자/파킹 특성
    if item.get('risk_grade'): features.append(f"위험등급 {item.get('risk_grade')}등급")
    if item.get('principal_protected') is False: features.append("원금비보장 손실위험")
    if item.get('interest_payment_date'): features.append("매일/매월 이자지급 파킹통장")
    
    # 4. 리스크 및 요약
    risk_info = item.get('risk_caution') or item.get('loss_warning') or ""
    summary = item.get('summary', '')

    # [최종 검색 텍스트 조립]
    full_text = f"""
    {idx_info}
    [스펙] {' | '.join(specs)}
    [특징] {' | '.join(features)}
    [요약] {summary}
    [주의] {risk_info}
    """
    
    documents.append(full_text.strip())
    metadatas.append(item)

if documents:
    vectorstore = FAISS.from_texts(documents, embeddings, metadatas)
    print(">>> FAISS Vector Store 구축 완료")
else:
    print("Error: 벡터 스토어 생성 실패")
    exit(1)

# ==========================================
# 2. State 정의
# ==========================================
class UserProfile(TypedDict):
    age: Optional[int]
    job: Optional[str]
    category: Optional[str]
    financial_goal: Optional[str]
    investment_period: Optional[str]
    monthly_amount: Optional[str]
    risk_tolerance: Optional[str]

class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_query: str
    profile: UserProfile
    missing_info: List[str]
    ask_count: int
    candidates: List[dict]
    critic_report: str
    final_response: str

# ==========================================
# 3. Node 구현 (High-End PB Version)
# ==========================================

# --- Node 1: Profile (심층 프로파일링) ---
def profile_node(state: AgentState):
    print("\n[1. Profiling] 심층 분석 중...")
    
    history_text = "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])
    full_context = f"{history_text}\nUser(Current): {state['user_query']}"
    current_profile = state["profile"]
    ask_count = state.get("ask_count", 0)
    
    system_prompt = f"""
    당신은 20년 경력의 베테랑 금융 PB입니다. 대화 맥락에서 아래 7가지 핵심 정보를 추출하세요.
    
    [추출 항목]
    1. age: 나이 (숫자)
    2. job: 직업
    3. category: [Savings(예적금), Investment(투자), Demand(입출금), Insurance(보험), Pension(연금)]
    4. financial_goal: 자금 목적 (결혼, 주택, 노후 등)
    5. investment_period: 투자/예치 기간
    6. monthly_amount: 월 납입/가용 금액 (예: '50만원')
    7. risk_tolerance: [안전지향, 수익지향] 중 하나
       - '수익률 높은거', '돈 불리기' -> '수익지향'
       - '안전한거', '원금보장' -> '안전지향'
    
    기존 정보: {json.dumps(current_profile, ensure_ascii=False)}
    JSON 형식으로만 답하세요.
    """
    
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=full_context)])
        extracted = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        
        for k, v in extracted.items():
            if v is not None: current_profile[k] = v
            
        missing = []
        if ask_count < 2: # 최대 2번만 되묻기
            if not current_profile.get("category"): 
                missing.append("category (상품 종류)")
            else:
                # 카테고리가 있으면 더 깊은 정보 요구
                if not current_profile.get("investment_period"): missing.append("investment_period (기간)")
                if not current_profile.get("monthly_amount"): missing.append("monthly_amount (금액)")
                if current_profile.get("category") == "Investment" and not current_profile.get("risk_tolerance"):
                    missing.append("risk_tolerance (투자성향)")
        
        return {"profile": current_profile, "missing_info": missing, "ask_count": ask_count}
    except:
        return {"missing_info": []}

# --- Node 2: AskMore (자연스러운 질문) ---
def ask_more_node(state: AgentState):
    print("\n[2. AskMore] 정보 보강 요청...")
    missing = state["missing_info"]
    prompt = f"""
    사용자에게 금융 상품 추천을 위해 필요한 정보 {missing}를 물어보세요.
    단, 기계적으로 묻지 말고 "기간을 알려주시면 이자가 가장 높은 상품을 찾을 수 있어요" 처럼 이유를 덧붙이세요.
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"final_response": response.content, "ask_count": state["ask_count"] + 1}

# --- Node 3: Retrieve (검색어 확장 & 매핑) ---
def retrieve_node(state: AgentState):
    print("\n[3. Retrieving] 정밀 매칭 검색 중...")
    profile = state["profile"]
    query = state["user_query"]
    
    # [Mapping Strategy] 프로필 정보를 DB 검색용 키워드로 변환
    keywords = []
    
    # 1. 카테고리 & 성향
    if profile.get("category"): keywords.append(profile.get("category"))
    if profile.get("risk_tolerance") == "안전지향": keywords.append("원금보장 예금자보호 안정형")
    elif profile.get("risk_tolerance") == "수익지향": keywords.append("고수익 실적배당 위험형")
    
    # 2. 사용자 발화의 의도 파악
    if "앱" in query or "폰" in query: keywords.append("비대면 온라인")
    if "지점" in query or "창구" in query: keywords.append("영업점")
    if "자유" in query or "아무때나" in query: keywords.append("자유적립식")
    if "매일" in query: keywords.append("이자지급일 매일") # 파킹통장
    
    final_query = f"{' '.join(keywords)} {query}"
    print(f" -> 검색 쿼리: {final_query}")
    
    docs_and_scores = vectorstore.similarity_search_with_score(final_query, k=30)
    candidates = []
    
    for doc, score in docs_and_scores:
        p = doc.metadata
        
        # [Strict Filter] 카테고리 불일치 제거
        if profile.get("category") and profile.get("category") not in p.get('category', ''):
            continue
            
        # [Eligibility Filter] 나이 제한 체크
        age = profile.get("age")
        elig = p.get('eligibility')
        if age and isinstance(elig, dict):
            if elig.get('age_min') and age < elig.get('age_min'): continue
            if elig.get('age_max') and age > elig.get('age_max'): continue
        
        candidates.append(p)
        if len(candidates) >= 3: break
    
    # Fallback (검색 실패 시 범위 확장)
    if not candidates and profile.get("category"):
        print(" -> [Info] 상세 조건 매칭 실패. 카테고리 기반 광범위 검색 실행.")
        for doc, score in docs_and_scores:
            if profile.get("category") in doc.metadata.get('category', ''):
                candidates.append(doc.metadata)
            if len(candidates) >= 2: break
            
    print(f" -> 최종 후보 상품: {len(candidates)}개")
    return {"candidates": candidates}

# --- Node 4: Critic (수익 시뮬레이션 & 스펙 검증) ---
def critic_node(state: AgentState):
    print("\n[4. Critical Thinking] 수익성 및 적합성 시뮬레이션...")
    candidates = state["candidates"]
    profile = state["profile"]
    
    if not candidates: return {"critic_report": "No candidates found."}

    prompt = f"""
    당신은 고객 자산 증식을 최우선으로 하는 PB입니다.
    후보 상품을 분석하여 예상 수익을 계산하고, 부적합한 요소를 찾아내세요.
    
    [고객 프로필]
    - 월 납입금: {profile.get('monthly_amount', '미정')}
    - 기간: {profile.get('investment_period', '미정')}
    - 성향: {profile.get('risk_tolerance', '미정')}
    
    [후보 상품]
    {json.dumps(candidates, ensure_ascii=False, indent=2)}
    
    [분석 요구사항]
    1. **수익 시뮬레이션 (필수)**: 
       - 고객이 월 납입금을 기간 동안 납입했을 때, **만기 예상 수령액(세후)**을 계산해 주세요. (이자소득세 15.4% 차감)
       - 정보가 부족하면 "약 00% 금리 적용 시"라고 가정하여 계산하세요.
       - 계산 과정을 자세히 작성하세요.
    2. **한도 검증**: 
       - 월 납입금이 상품의 'max_deposit'을 초과하는지 확인하세요. 초과 시 **"한도 초과! 분산 투자 필요"** 경고 필수.
    3. **가입 편의성**: 
       - 'join_channel'을 확인하여 앱 가입 가능 여부를 명시하세요.
    4. **리스크**: 
       - 원금 손실 가능성, 중도해지 불이익 등을 찾아내세요.
    
    위 내용을 포함한 상세 분석 리포트를 작성하세요.
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"critic_report": response.content}

# --- Node 5: Response (상세 정보 제공 & 행동 유도) ---
def response_node(state: AgentState):
    print("\n[5. Response] 최종 제안서 작성 중...")
    report = state.get("critic_report", "No candidates found.")
    candidates = state.get("candidates", [])
    profile = state["profile"]
    
    if report == "No candidates found.":
        return {"final_response": "죄송합니다. 조건에 맞는 상품을 찾지 못했습니다."}

    # [Data Injection] 원본 데이터의 디테일을 보기 좋게 가공
    product_details = ""
    for idx, item in enumerate(candidates, 1):
        # 가입채널
        join = item.get('join_channel')
        if isinstance(join, dict): 
            join_str = ", ".join(join.get('descriptions', [])) or ("앱 가입 가능" if join.get('is_online') else "영업점 방문")
        else: join_str = str(join)
        
        # 금리 표시
        rate_info = item.get('interest_rate')
        if isinstance(rate_info, dict):
            rate_str = f"최고 연 {rate_info.get('max_rate')}% (기본 {rate_info.get('base_rate')}%)"
            rate_cond = f"\n   └ 우대조건: {', '.join(rate_info.get('prime_conditions', []))}" if rate_info.get('prime_conditions') else ""
        else:
            rate_str = str(rate_info or item.get('base_interest_rate', '상세설명 참조'))
            rate_cond = ""

        product_details += f"""
        [{idx}. {item.get('product_name')}]
        ──────────────────────────────
        💰 금리/수익: {rate_str}{rate_cond}
        📅 기간/한도: {item.get('min_month', 0)}~{item.get('max_month', 0)}개월 / 월 {item.get('max_deposit', '제한없음')}원
        📱 가입방법: {join_str}
        📝 특징: {item.get('accumulation_type', '')} {item.get('summary', '')}
        ⚠️ 주의: {item.get('risk_caution', item.get('loss_warning', '없음'))}
        """

    prompt = f"""
    당신은 고객의 성공적인 자산 관리를 돕는 AI PB입니다.
    전문적인 분석 결과와 상세 상품 정보를 바탕으로 설득력 있는 제안서를 작성하세요.
    
    [Critic 분석 결과]
    {report}
    
    [상품 상세 스펙]
    {product_details}
    
    [답변 구조 가이드]
    
    ### 1. 🏆 AI PB의 원픽 추천
    *   "고객님의 상황(목적, 금액)에는 **[상품명]**이 가장 유리합니다." 로 시작.
    *   추천 이유를 핵심만 요약.
    
    ### 2. 💸 머니 시뮬레이션 (예상 수익)
    *   "매월 {profile.get('monthly_amount', '00')}원씩 납입 시..."
    *   Critic이 계산한 **계산 과정과 만기 예상 수령액**을 명시. (예: "얼마의 이자로 어떻게 계산했더니 세후 약 6,150,000원을 받으실 수 있습니다.")
    
    ### 3. 🔍 상품 상세 정보
    *   위 [상품 상세 스펙] 내용을 바탕으로 고객이 알아야 할 핵심 정보(우대금리 조건, 가입방법 등)를 정리.
    
    ### 4. 💡 솔직한 조언 (Risk & Tip)
    *   한도가 부족하다면 **"풍차돌리기"**나 **"분산 투자"** 제안.
    *   리스크나 까다로운 우대조건이 있다면 미리 고지.
    
    ### 5. 🚀 가입 바로가기
    *   가입 채널(앱/영업점)에 따른 구체적 행동 지침 제시.
    *   예: "지금 신한 SOL 앱 > 상품몰 > 적금 메뉴에서 가입하세요!"
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"final_response": response.content}

# ==========================================
# 4. Graph 구성
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("profile", profile_node)
workflow.add_node("ask_more", ask_more_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("critic", critic_node)
workflow.add_node("response", response_node)

def route_after_profile(state):
    # 정보가 부족하고 && 질문 횟수가 2회 미만일 때만 ask_more
    if state["missing_info"] and state["ask_count"] < 2: return "ask_more"
    return "retrieve"
def route_after_retrieve(state):
    if not state["candidates"]: return "response"
    return "critic"

workflow.set_entry_point("profile")
workflow.add_conditional_edges("profile", route_after_profile)
workflow.add_edge("ask_more", END)
workflow.add_conditional_edges("retrieve", route_after_retrieve)
workflow.add_edge("critic", "response")
workflow.add_edge("response", END)

app = workflow.compile()

# ==========================================
# 5. 실행 루프
# ==========================================
def start_chat():
    print("="*60)
    print("🤖 AI PB와 상담을 시작합니다. (종료: q)")
    print("   Tip: 나이, 직업, 월 저축액, 자금 목적을 알려주시면 정확도가 올라갑니다.")
    print("="*60)
    
    history = []
    curr_profile = {"age": None, "job": None, "category": None, 
                   "financial_goal": None, "investment_period": None, 
                   "monthly_amount": None, "risk_tolerance": None}
    curr_ask_count = 0 
    
    while True:
        q = input("\nUser: ").strip()
        if q.lower() == 'q': break
        
        inputs = {
            "messages": history, "user_query": q, "profile": curr_profile,
            "ask_count": curr_ask_count, "missing_info": [], "candidates": []
        }
        res = app.invoke(inputs)
        ans = res['final_response']
        print(f"\nAI: {ans}")
        
        history.append(HumanMessage(content=q))
        history.append(SystemMessage(content=ans))
        if "profile" in res: curr_profile = res["profile"]
        if "ask_count" in res: curr_ask_count = res["ask_count"]

if __name__ == "__main__":
    start_chat()
