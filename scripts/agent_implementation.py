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
# 1. 데이터 로드 & FAISS 구축
# ==========================================
# (기존 데이터 로드 로직 유지 - 생략 가능하지만 실행을 위해 포함)
json_path = "/home/wlaud/projects/shinhan/data/sample_extracted_10.json"
if not os.path.exists(json_path):
    print("Dummy Data Used")
    raw_data = [{"product_name": "신한 청년우대 적금", "category": "Savings", "eligibility": {"age_min": 19}, "interest_rate": {"max_rate": 5.5}}]
else:
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

documents = []
metadatas = []
for item in raw_data:
    if "error" in item: continue
    # 검색용 텍스트 풍부하게 구성
    desc = [f"상품명: {item.get('product_name')}", f"카테고리: {item.get('category')}"]
    if item.get('interest_rate'): desc.append(f"금리: {item.get('interest_rate')}")
    if item.get('risk_caution'): desc.append(f"위험: {item.get('risk_caution')}")
    documents.append("\n".join(desc))
    metadatas.append(item)

vectorstore = FAISS.from_texts(documents, embeddings, metadatas)
print(">>> FAISS 구축 완료")

# ==========================================
# 2. State 정의 (UserProfile 확장 + Ask Count)
# ==========================================
class UserProfile(TypedDict):
    # [신원]
    age: Optional[int]
    job: Optional[str]
    # [니즈]
    category: Optional[str]     # Savings, Investment, Demand, Insurance, Pension
    financial_goal: Optional[str]
    # [제약]
    investment_period: Optional[str] # 기간
    monthly_amount: Optional[str]    # 금액
    # [성향]
    risk_tolerance: Optional[str]    # 안전/수익

class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_query: str
    profile: UserProfile
    missing_info: List[str]
    ask_count: int              # [New] 추가 질문 횟수 카운터
    candidates: List[dict]
    critic_report: str
    final_response: str

# ==========================================
# 3. Node 구현
# ==========================================

# --- Node 1: Profile (PB급 분석) ---
def profile_node(state: AgentState):
    print("\n[1. Profiling] 심층 분석 중...")
    
    # 히스토리 병합
    history_text = "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])
    current_input = state["user_query"]
    full_context = f"{history_text}\nUser(Current): {current_input}"
    
    current_profile = state["profile"]
    ask_count = state.get("ask_count", 0) # 현재까지 물어본 횟수
    
    system_prompt = f"""
    당신은 전문 PB(Private Banker) 수준의 금융 프로파일러입니다.
    대화 맥락을 분석하여 아래 항목을 추출/업데이트하세요.
    
    [추출 항목 정의]
    1. age: 나이 (숫자)
    2. job: 직업
    3. category: [Savings, Investment, Demand, Insurance, Pension] 중 택1 (명확하지 않으면 null)
    4. financial_goal: 자금 목적 (결혼, 주택, 여행 등)
    5. investment_period: 기간 (6개월, 3년 등)
    6. monthly_amount: 월 납입/예치 금액
    7. risk_tolerance: [안전지향, 수익지향] 중 하나
    
    [규칙]
    - '수익률 높은거' -> risk_tolerance: '수익지향' (category는 null일 수 있음)
    - '안전한거' -> risk_tolerance: '안전지향'
    
    기존 정보: {json.dumps(current_profile, ensure_ascii=False)}
    JSON 형식으로만 답하세요.
    """
    
    try:
        msg = [SystemMessage(content=system_prompt), HumanMessage(content=full_context)]
        response = llm.invoke(msg)
        
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean_json)
        
        for k, v in extracted.items():
            if v is not None: current_profile[k] = v
            
        # [Multi-turn 전략: 2회 제한]
        missing = []
        
        # 이미 3번 물어봤으면 더 이상 안 물어봄 (부족한 대로 진행)
        if ask_count < 3:
            # 1. 카테고리 필수
            if not current_profile.get("category"):
                missing.append("category (상품 종류)")
            
            # 2. 카테고리 있으면 디테일 질문
            else:
                # 기간/금액은 중요
                if not current_profile.get("investment_period"): missing.append("investment_period (기간)")
                if not current_profile.get("monthly_amount"): missing.append("monthly_amount (금액)")
                # 투자인 경우 성향 필수
                if current_profile.get("category") == "Investment" and not current_profile.get("risk_tolerance"):
                    missing.append("risk_tolerance (위험감수 성향)")
        else:
            print(" -> [Info] 질문 횟수 초과로 정보 수집 중단 (Soft Mode)")

        return {"profile": current_profile, "missing_info": missing, "ask_count": ask_count}
        
    except Exception as e:
        print(f"Profiling Error: {e}")
        return {"missing_info": []}

# --- Node 2: AskMore (세련된 질문) ---
def ask_more_node(state: AgentState):
    print("\n[2. AskMore] 추가 정보 요청...")
    missing = state["missing_info"]
    current_count = state["ask_count"]
    
    prompt = f"""
    당신은 꼼꼼한 PB입니다. 
    사용자에게 부족한 정보 [{', '.join(missing)}]를 물어봐야 합니다.
    
    [작성 규칙]
    1. 단순히 나열하지 말고, **"왜 필요한지"**를 설명하며 자연스럽게 물어보세요.
    2. 예: "기간" -> "돈을 얼마나 묶어두실 수 있는지 알려주시면, 그 기간에 이자가 제일 센 상품을 찾아드릴게요!"
    3. 필요한 정보를 명확하게 정리하여 전달하세요.
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    
    # 질문 횟수 증가
    return {"final_response": response.content, "ask_count": current_count + 1}

# --- Node 3: Retrieve (보강된 검색) ---
def retrieve_node(state: AgentState):
    print("\n[3. Retrieving] 정밀 검색 중...")
    profile = state["profile"]
    
    # 검색어 조합 (모든 프로필 정보 활용)
    search_parts = []
    if profile.get("category"): search_parts.append(profile.get("category"))
    else: search_parts.append("금융 상품") # 카테고리 없으면 범용 검색
    
    if profile.get("financial_goal"): search_parts.append(f"{profile.get('financial_goal')} 목적")
    if profile.get("risk_tolerance") == "수익지향": search_parts.append("고수익")
    if profile.get("risk_tolerance") == "안전지향": search_parts.append("원금보장 안전")
    
    query = state["user_query"]
    search_query = f"{' '.join(search_parts)} {query}"
    
    docs_and_scores = vectorstore.similarity_search_with_score(search_query, k=30)
    candidates = []
    
    for doc, score in docs_and_scores:
        p = doc.metadata
        
        # [Filter] 카테고리 (있으면 필수 체크)
        if profile.get("category") and profile.get("category") not in p.get('category', ''):
            continue
            
        # [Filter] 나이 (있으면 체크)
        age = profile.get("age")
        eligibility = p.get('eligibility')
        if age and isinstance(eligibility, dict):
            if eligibility.get('age_min') and age < eligibility.get('age_min'): continue
            if eligibility.get('age_max') and age > eligibility.get('age_max'): continue
        
        candidates.append(p)
        if len(candidates) >= 3: break
    
    # Fallback
    if not candidates and profile.get("category"):
        for doc, score in docs_and_scores:
            p = doc.metadata
            if profile.get("category") in p.get('category', ''):
                candidates.append(p)
            if len(candidates) >= 2: break
            
    print(f" -> 검색 결과: {len(candidates)}개")
    return {"candidates": candidates}

# --- Node 4: Critic (상세 프로필 기반 검증) ---
def critic_node(state: AgentState):
    print("\n[4. Critical Thinking] 상품 검증 중...")
    candidates = state["candidates"]
    profile = state["profile"]
    
    if not candidates: return {"critic_report": "No candidates found."}

    candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2)

    prompt = f"""
    당신은 고객 전담 PB(Private Banker)입니다.
    고객 프로필과 후보 상품을 대조하여 냉철하게 분석하세요.
    
    [고객 프로필]
    {profile}
    
    [후보 상품]
    {candidates_str}
    
    [분석 포인트]
    1. **기간/금액 매칭**: 고객이 원하는 기간({profile.get('investment_period')})과 금액({profile.get('monthly_amount')})이 상품 조건(한도, 만기)과 맞는가? 불일치 시 경고.
    2. **조건 매칭**: 상품 가입 조건에 부합하는가? 또 우대 조건이 고객에게 해당되는가?
    3. **성향 적합성**: {profile.get('risk_tolerance')} 성향인데 상품 위험도(risk_grade/원금보장)가 맞는가?
    4. **현실성**: 우대금리 조건이 까다로운가? (함정 탐지)
    5. **목적 적합성**: {profile.get('financial_goal')} 목적에 부합하는가? (예: 결혼자금인데 중도해지 리스크 큼)
    6. **수익성**: 후보 상품을 통해 고객이 얻는 수익이 최대화 되는가?
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"critic_report": response.content}

# --- Node 5: Response (컨설팅형 답변) ---
def response_node(state: AgentState):
    print("\n[5. Response] 답변 생성 중...")
    report = state.get("critic_report", "No candidates found.")
    profile = state["profile"]
    
    if report == "No candidates found.":
        return {"final_response": "죄송합니다. 조건에 딱 맞는 상품을 찾지 못했습니다."}
    
    # 정보 부족 시 안내 멘트
    missing_notice = ""
    if not profile.get("age"): missing_notice += "- 정확한 나이\n"
    if not profile.get("monthly_amount"): missing_notice += "- 월 납입 가능 금액\n"
    
    prompt = f"""
    당신은 신뢰받는 AI PB입니다. Critic 보고서를 바탕으로 컨설팅해주세요.
    
    [Critic Report]
    {report}
    
    [작성 가이드]
    1. "고객님의 상황(기간, 목적 등)을 고려할 때..." 라는 멘트로 시작하여 추천 상품을 제안하세요.
    2. 상품에 대한 자세한 설명을 제시하세요. (가입 조건, 우대 조건 등)
    3. 상품을 추천한 이유를 자세히 제시하세요. (최소, 최대 예상 수익 등)
    4. Critic이 지적한 **'기간/금액 불일치'**나 **'리스크'**가 있다면, 숨기지 말고 **"솔직한 조언"** 섹션에서 강조하세요.
    5. {f"참고로 아래 정보를 더 알려주시면 정확도가 올라갑니다:\n{missing_notice}" if missing_notice else ""}
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
    if state["missing_info"] and state["ask_count"] < 2:
        return "ask_more"
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
# 5. Interactive Chat Loop
# ==========================================
def start_chat():
    print(
        """>>> AI PB와 상담을 시작합니다. (종료: q)\n\n \n안녕하세요. 개인 맞춤형 금융 상품 추천 서비스입니다.\n
원하시는 상품 종류(예: 적금, IRP)가 무엇인가요?
나이와 자금 사용 목적 등 자세한 정보를 알려주시면, 광고성 추천은 빼고 고객님께 진짜 필요한 상품만 비교해 드립니다."""
)
    history = []
    # 프로필 및 ask_count 초기화
    curr_profile = {"age": None, "job": None, "category": None, 
                   "financial_goal": None, "investment_period": None, 
                   "monthly_amount": None, "risk_tolerance": None}
    curr_ask_count = 0 
    
    while True:
        q = input("\nUser: ").strip()
        if q.lower() == 'q': break
        
        inputs = {
            "messages": history,
            "user_query": q,
            "profile": curr_profile,
            "ask_count": curr_ask_count, # 카운트 전달
            "missing_info": [], "candidates": []
        }
        
        res = app.invoke(inputs)
        ans = res['final_response']
        print(f"\nAI: {ans}")
        
        # 상태 업데이트
        history.append(HumanMessage(content=q))
        history.append(SystemMessage(content=ans))
        if "profile" in res: curr_profile = res["profile"]
        if "ask_count" in res: curr_ask_count = res["ask_count"] # 카운트 업데이트

if __name__ == "__main__":
    start_chat()
