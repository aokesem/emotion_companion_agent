import json
import os
from datetime import datetime, timezone
from typing import Annotated, TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from supabase import create_client, Client

# 加载环境变量
load_dotenv()

#定义系统状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] #聊天记录
    emotion_data: Dict[str, Any] #存储情感分析结果
    openness: float #开放度
    candidate_memories: List[Dict[str, Any]] #从数据库检索出来的历史对话片段
    strategy_pack: Dict[str, Any] #下次对话的心理学策略
    user_id: str #用户的唯一标识

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def _safe_json_parse(raw_text: str) -> Dict[str, Any]:
    try:
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {}

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

def emotion_decoder_node(state: AgentState):
    """模块①：隐性情感解码器。"""
    last_message = state["messages"][-1].content #用户最新的输入内容
    recent_context = [m.content for m in state["messages"][-4:-1]] #拿取最近4条对话作为上下文

    prompt = f"""
你是情感解码器。根据用户当前发言和最近上下文，输出严格JSON：
{{
  "surface_content": "表层意思",
  "inferred_emotion": "如孤独/怀旧/防御",
  "confidence": "高/中/低",
  "intensity": "低/中/高",
  "recommended_tone": "试探性共情/温和倾听/稳定陪伴"
}}
最近上下文: {recent_context}
用户原始文本: "{last_message}"
只输出JSON，不要额外说明。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    emotion_info = _safe_json_parse(response.content) #解析LLM的响应，提取情感分析结果
    if not emotion_info: #如果解析失败，则使用默认值
        emotion_info = {
            "surface_content": last_message,
            "inferred_emotion": "未知",
            "confidence": "低",
            "intensity": "低",
            "recommended_tone": "试探性共情",
        }
    print(f"--- 情感分析结果: {emotion_info['inferred_emotion']} ({emotion_info['intensity']}) ---") #打印情感分析结果
    return {"emotion_data": emotion_info}

def openness_tracker_node(state: AgentState):
    """模块②：情绪开放度追踪器。"""
    last_user_text = state["messages"][-1].content #用户最新的输入内容
    openness = state.get("openness", 0.3) #从状态面板读取当前开放度，初始值为0.3
    delta = 0.0 #开放度变化量

    #初版内容，仍需完善
    if "我其实" in last_user_text: #如果用户表达了对自我不满或不自信，则增加开放度
        delta += 0.3
    if len(last_user_text) > 30: #如果用户输入的内容较长，则增加开放度
        delta += 0.2
    if len(last_user_text) < 8: #如果用户输入的内容较短，则减少开放度
        delta -= 0.1
    if "哈哈没事" in last_user_text or "算了不说了" in last_user_text: #如果用户表达了对对话的结束或不感兴趣，则减少开放度
        delta -= 0.1

    new_openness = _clamp(openness + delta) #将开放度限制在0-1之间
    return {"openness": new_openness} #更新开放度

def memory_retrieval_node(state: AgentState):
    """模块③：记忆检索（Supabase Top3）。"""
    if not supabase:
        return {"candidate_memories": []} #如果数据库连接失败，则返回空列表
    try:
        resp = (
            supabase.table("conversation_turns") #从数据库中检索最近3条对话
            .select("user_text,emotion_label,intensity,created_at") #选择需要的回忆字段（用户原话、当时的情绪、强度、时间）
            .eq("user_id", state["user_id"]) #根据用户ID过滤
            .order("created_at", desc=True) #按创建时间降序排序
            .limit(3)
            .execute() #执行查询
        )
        return {"candidate_memories": resp.data or []}
    except Exception:
        return {"candidate_memories": []}

def strategy_controller_node(state: AgentState):
    """模块④：策略控制器。"""
    openness = state.get("openness", 0.3)  #提取判定的用户开放度
    confidence = state.get("emotion_data", {}).get("confidence", "低") #提取情感分析结果的置信度
    inferred = state.get("emotion_data", {}).get("inferred_emotion", "未知") #提取情感分析结果的情感类型
    intensity = state.get("emotion_data", {}).get("intensity", "低") #提取情感分析结果的强度

    memory_allowed = openness >= 0.5 and len(state.get("candidate_memories", [])) > 0 #只有当用户开放度达到一定程度时才提及历史记忆
    if openness < 0.3 or confidence == "低": #如果用户开放度低于0.3或情感分析结果的置信度为低，则使用主动倾听策略
        therapy = "active_listening"
        posture = "深度示弱"
    elif "过去" in state["messages"][-1].content and openness > 0.5: #如果用户最近说过“过去”，则使用回忆疗法
        therapy = "reminiscence"
        posture = "软过渡延伸"
    elif inferred == "存在性焦虑" and intensity == "高": #如果情感分析结果的情感类型为存在性焦虑且强度为高，则使用存在性疗法
        therapy = "logotherapy"
        posture = "陪伴倾听"
    else: #其他情况下使用默认策略
        therapy = "default"
        posture = "软过渡延伸"

    strategy_pack = { #下次对话的心理学策略，包括疗法类型、姿态、是否允许提及历史记忆、语气倾向
        "therapy": therapy,
        "posture": posture,
        "memory_allowed": memory_allowed,
        "tone": state.get("emotion_data", {}).get("recommended_tone", "温和倾听"),
    }
    return {"strategy_pack": strategy_pack}

def generation_node(state: AgentState):
    """模块⑤：对话生成。"""
    strategy = state.get("strategy_pack", {})
    memory_allowed = strategy.get("memory_allowed", False)
    memory_text = ""
    if memory_allowed:
        memory_text = f"可引用历史记忆（最多1条，软过渡）：{state.get('candidate_memories', [])[:1]}"
    
    #默认系统提示词
    system_prompt = f"""
你是独居老人陪伴助手“小安”，人设是谦逊好奇的晚辈。
要求：
1) 口语化、慢节奏、句子短；
2) 不说教，不给直接行动建议；
3) 优先共情，再轻微延伸；
4) 当前策略：{strategy}
5) {memory_text}
请输出一段自然中文回复。
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

#将记忆写入数据库
def memory_writer_node(state: AgentState):
    """模块⑥：记忆写入（Supabase）。"""
    if not supabase:
        return {}
    last_user = state["messages"][-2].content if len(state["messages"]) >= 2 else ""
    last_ai = state["messages"][-1].content if state["messages"] else ""
    payload = {
        "user_id": state["user_id"],
        "user_text": last_user,
        "ai_text": last_ai,
        "emotion_label": state["emotion_data"].get("inferred_emotion", "未知"),
        "intensity": state["emotion_data"].get("intensity", "低"),
        "openness": state.get("openness", 0.3),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        supabase.table("conversation_turns").insert(payload).execute()
    except Exception:
        pass
    return {}

workflow = StateGraph(AgentState)
workflow.add_node("decoder", emotion_decoder_node)
workflow.add_node("openness_tracker", openness_tracker_node)
workflow.add_node("memory_retrieval", memory_retrieval_node)
workflow.add_node("strategy_controller", strategy_controller_node)
workflow.add_node("generator", generation_node)
workflow.add_node("memory_writer", memory_writer_node)

workflow.set_entry_point("decoder")
workflow.add_edge("decoder", "openness_tracker")
workflow.add_edge("openness_tracker", "memory_retrieval")
workflow.add_edge("memory_retrieval", "strategy_controller")
workflow.add_edge("strategy_controller", "generator")
workflow.add_edge("generator", "memory_writer")
workflow.add_edge("memory_writer", END)

app = workflow.compile()

if __name__ == "__main__":
    print("AI精神慰藉助手已启动 (输入 'quit' 退出)")
    initial_state = {
        "messages": [],
        "emotion_data": {},
        "openness": 0.3,
        "candidate_memories": [],
        "strategy_pack": {},
        "user_id": os.getenv("DEFAULT_USER_ID", "demo_user"),
    }

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        initial_state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(initial_state)
        initial_state = result
        print(f"小安: {result['messages'][-1].content}")
