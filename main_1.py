import copy
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from supabase import create_client, Client

os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897" #不要删除这两行
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"

# 加载环境变量
load_dotenv()


# 定义系统状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]  # 聊天记录
    emotion_data: Dict[str, Any]  # 情感分析结果
    openness: float  # 开放度
    candidate_memories: List[Dict[str, Any]]  # 检索到的历史片段
    strategy_pack: Dict[str, Any]  # 本轮策略
    user_id: str
    session_id: str  # 当前会话（chat_sessions.id，UUID 字符串）
    user_profile: Dict[str, Any]  # 用户角色画像（与 user_profiles.profile JSON 对齐）


def _default_user_profile() -> Dict[str, Any]:
    return {
        "gender": None,
        "age": None,
        "former_occupation": None,
        "education": None,
        "health": {"summary": None, "history": []},
        "family_social": {"summary": None, "history": []},
        "hobbies": {"summary": None, "history": []},
        "personality": {"summary": None, "history": []},
        "habits": {"summary": None, "history": []},
    }


def _ensure_profile_shape(p: Any) -> Dict[str, Any]:
    base = _default_user_profile()
    if not isinstance(p, dict):
        return copy.deepcopy(base)
    out = copy.deepcopy(base)
    for k in base:
        if k in p:
            if k in (
                "health",
                "family_social",
                "hobbies",
                "personality",
                "habits",
            ) and isinstance(p[k], dict):
                out[k] = {**out[k], **p[k]}
                if not isinstance(out[k].get("history"), list):
                    out[k]["history"] = []
                if "summary" not in out[k]:
                    out[k]["summary"] = None
            else:
                out[k] = p[k]
    return out


def _merge_user_profile(
    current: Dict[str, Any],
    raw_patch: Dict[str, Any],
    source_turn_id: Optional[int],
) -> Dict[str, Any]:
    """谨慎合并：标量仅填空；长叙述 summary 仅填空；history 只追加。"""
    out = _ensure_profile_shape(current)
    p = (raw_patch or {}).get("profile_patch") or raw_patch
    if not isinstance(p, dict):
        return out

    ts = datetime.now(timezone.utc).isoformat()
    conf_fill_empty = ("高", "中")

    for key in ("gender", "age", "former_occupation", "education"):
        item = p.get(key)
        if not isinstance(item, dict):
            continue
        if item.get("confidence") not in conf_fill_empty:
            continue
        if out.get(key) is not None and out.get(key) != "":
            continue
        val = item.get("value")
        if val is not None and val != "":
            out[key] = val

    for key in ("health", "family_social", "hobbies", "personality", "habits"):
        block = p.get(key)
        if not isinstance(block, dict):
            continue
        if key not in out:
            out[key] = {"summary": None, "history": []}
        summ = block.get("summary")
        if isinstance(summ, dict) and out[key].get("summary") in (None, ""):
            if summ.get("confidence") in conf_fill_empty:
                v = summ.get("value")
                if v is not None and v != "":
                    out[key]["summary"] = v
        for text in block.get("append_history", []):
            if not text or not str(text).strip():
                continue
            out[key].setdefault("history", []).append(
                {
                    "text": str(text).strip(),
                    "recorded_at": ts,
                    "source_turn_id": source_turn_id,
                }
            )
    return out


llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_embeddings: Optional[OpenAIEmbeddings] = None

EMOTION_LABEL_SET = {
    "孤独",
    "怀旧",
    "防御",
    "焦虑",
    "悲伤",
    "平静",
    "压抑",
    "被需要感",
    "无明显情感",
}


def _get_embeddings() -> Optional[OpenAIEmbeddings]:
    """智谱等 OpenAI 兼容 embedding（维度与 DB vector(1536) 一致）。"""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    base = os.getenv("EMBEDDING_BASE_URL")
    key = os.getenv("EMBEDDING_API_KEY")
    if not base or not key:
        return None
    _embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "embedding-3"),
        openai_api_base=base,
        openai_api_key=key,
        dimensions=int(os.getenv("EMBEDDING_DIMS", "1536")),
    )
    return _embeddings


def _safe_json_parse(raw_text: str) -> Dict[str, Any]:
    try:
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _build_role_labeled_context(messages: List[BaseMessage], keep_last: int = 6) -> List[Dict[str, str]]:
    """将最近上下文转换为带角色标签的结构，供模块①使用。"""
    ctx: List[Dict[str, str]] = []
    for msg in messages[-keep_last:]:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "system"
        ctx.append({"role": role, "text": str(msg.content)})
    return ctx


def _get_profile_for_decoder(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """只抽取对情感解码有帮助的画像片段，避免噪声过大。"""
    p = _ensure_profile_shape(user_profile or {})
    return {
        "age": p.get("age"),
        "health": p.get("health", {}).get("summary"),
        "family_social": p.get("family_social", {}).get("summary"),
        "hobbies": p.get("hobbies", {}).get("summary"),
        "personality": p.get("personality", {}).get("summary"),
        "habits": p.get("habits", {}).get("summary"),
    }


def _get_high_freq_emotions(user_id: str) -> List[Dict[str, Any]]:
    """历史高频情感：仅统计 confidence 为中/高的记录。"""
    if not supabase:
        return []
    try:
        resp = (
            supabase.table("conversation_turns")
            .select("emotion_data_json")
            .eq("user_id", user_id)
            .order("id", desc=True)
            .limit(500)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        print(f"[decoder] 高频情感检索失败: {e}")
        return []

    counts: Dict[str, int] = {}
    total_valid = 0
    for row in rows:
        info = row.get("emotion_data_json") if isinstance(row, dict) else None
        if isinstance(info, str):
            info = _safe_json_parse(info)
        if not isinstance(info, dict):
            continue
        conf = str(info.get("confidence", "")).strip()
        emo = str(info.get("inferred_emotion", "")).strip()
        if conf not in ("中", "高"):
            continue
        if emo not in EMOTION_LABEL_SET:
            continue
        total_valid += 1
        counts[emo] = counts.get(emo, 0) + 1

    if total_valid <= 0:
        return []
    # 冷启动与成熟期采用不同阈值，避免早期完全无可用信号
    if total_valid < 20:
        min_count, min_ratio = 3, 0.30
    else:
        min_count, min_ratio = 10, 0.25

    out: List[Dict[str, Any]] = []
    for emo, cnt in counts.items():
        ratio = cnt / total_valid
        if cnt >= min_count and ratio >= min_ratio:
            out.append({"emotion": emo, "count": cnt, "ratio": round(ratio, 3)})
    out.sort(key=lambda x: (-x["count"], -x["ratio"]))
    return out


def _normalize_emotion_info(raw: Dict[str, Any], last_message: str) -> Dict[str, Any]:
    """字段级容错：解析成功但字段缺失时逐项补默认并校验枚举。"""
    info = raw if isinstance(raw, dict) else {}
    surface = str(info.get("surface_content", "")).strip() or last_message
    hidden = str(info.get("hidden_subtext", "")).strip() or "未能稳定推断潜台词"
    inferred = str(info.get("inferred_emotion", "")).strip()
    if inferred not in EMOTION_LABEL_SET:
        inferred = "无明显情感"
    confidence = str(info.get("confidence", "")).strip()
    if confidence not in ("高", "中", "低"):
        confidence = "低"
    intensity = str(info.get("intensity", "")).strip()
    if intensity not in ("低", "中", "高"):
        intensity = "低"
    return {
        "surface_content": surface,
        "hidden_subtext": hidden,
        "inferred_emotion": inferred,
        "confidence": confidence,
        "intensity": intensity,
    }


def ensure_session_node(state: AgentState):
    """入口：为当前 run 准备 session_id；无则向 chat_sessions 新建一行。"""
    existing = (state.get("session_id") or "").strip()
    if existing:
        return {}
    uid = state["user_id"]
    if not supabase:
        return {"session_id": str(uuid.uuid4())}
    try:
        supabase.table("chat_sessions").insert(
            {
                "user_id": uid,
                "title": os.getenv("DEFAULT_SESSION_TITLE", "新会话"),
            }
        ).execute()
        got = (
            supabase.table("chat_sessions")
            .select("id")
            .eq("user_id", uid)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if got.data:
            return {"session_id": str(got.data[0]["id"])}
    except Exception as e:
        print(f"[ensure_session] 创建/读取 chat_sessions 失败: {e}")
    return {"session_id": str(uuid.uuid4())}


def load_profile_node(state: AgentState):
    """进入主流程前：按 user_id 加载 user_profiles，无则使用默认空画像。"""
    if not supabase:
        return {"user_profile": _default_user_profile()}
    try:
        resp = (
            supabase.table("user_profiles")
            .select("profile")
            .eq("user_id", state["user_id"])
            .limit(1)
            .execute()
        )
        if resp.data and resp.data[0].get("profile") is not None:
            return {
                "user_profile": _ensure_profile_shape(resp.data[0]["profile"]),
            }
    except Exception as e:
        print(f"[load_profile] 读取 user_profiles 失败: {e}")
    return {"user_profile": _default_user_profile()}


def emotion_decoder_node(state: AgentState):
    """模块①：隐性情感解码器。"""
    last_message = state["messages"][-1].content
    recent_context = _build_role_labeled_context(state["messages"][:-1], keep_last=6)
    profile_for_decoder = _get_profile_for_decoder(state.get("user_profile", {}))
    high_freq_emotions = _get_high_freq_emotions(state["user_id"])

    prompt = f"""
    你是情感解码器，服务对象是独居，缺乏社交，情感表达能力较差的中老年人。你的核心任务不是简单分类情感，
    而是先判断这句话的潜台词，再给出情感结论。

    你可在内部进行推断，但不要输出推理过程，只输出最终JSON。
    情感标签必须从以下集合中选择一个：
    ["孤独","怀旧","防御","焦虑","悲伤","平静","压抑","被需要感","无明显情感"]

    请输出严格JSON：
    {{
    "surface_content": "表层意思",
    "hidden_subtext": "推断的用户潜台词（一句话）",
    "inferred_emotion": "判断的用户情感类型，从情感标签集合中选择一个",
    "confidence": "判断的用户情感的置信度，从高/中/低中选择一个",
    "intensity": "判断的用户情感的强度，从低/中/高中选择一个"
    }}

    若最近上下文为空，则仅根据当前发言判断，不要臆造历史。

    最近上下文(带角色): {recent_context}
    用户画像(情感解码相关片段): {profile_for_decoder}
    历史高频情感(仅中/高置信统计结果，可能为空): {high_freq_emotions}
    用户原始文本: "{last_message}"
    只输出JSON，不要额外说明。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    emotion_info = _normalize_emotion_info(_safe_json_parse(response.content), last_message)
    print(
        f"--- 情感分析结果: {emotion_info['inferred_emotion']} ({emotion_info['intensity']}) ---"
    )
    return {"emotion_data": emotion_info}


def openness_tracker_node(state: AgentState):
    """模块②：情绪开放度追踪器。"""
    last_user_text = state["messages"][-1].content
    openness = state.get("openness", 0.3)
    delta = 0.0

    if "我其实" in last_user_text:
        delta += 0.3
    if len(last_user_text) > 30:
        delta += 0.2
    if len(last_user_text) < 8:
        delta -= 0.1
    if "哈哈没事" in last_user_text or "算了不说了" in last_user_text:
        delta -= 0.1

    new_openness = _clamp(openness + delta)
    return {"openness": new_openness}


def memory_retrieval_node(state: AgentState):
    """模块③：本会话内向量检索，失败时回退为按时间取最近 N 条。"""
    if not supabase:
        return {"candidate_memories": []}
    sid = (state.get("session_id") or "").strip()
    uid = state["user_id"]
    if not sid:
        return {"candidate_memories": []}
    last_user = state["messages"][-1].content if state.get("messages") else ""
    match_count = int(os.getenv("RAG_MATCH_COUNT", "5"))
    emb = _get_embeddings()
    if emb and last_user:
        try:
            query_vec = emb.embed_query(last_user)
            r = supabase.rpc(
                "match_conversation_turns",
                {
                    "query_embedding": query_vec,
                    "match_count": match_count,
                    "p_user_id": uid,
                    "p_session_id": sid,
                },
            ).execute()
            if r.data is not None:
                rows: List[Dict[str, Any]] = (
                    list(r.data) if isinstance(r.data, list) else []
                )
                if rows:
                    return {"candidate_memories": rows}
        except Exception as e:
            print(f"[memory_retrieval] 向量/RPC 失败: {e}")
    try:
        resp = (
            supabase.table("conversation_turns")
            .select("user_text,ai_text,emotion_label,intensity,created_at")
            .eq("user_id", uid)
            .eq("session_id", sid)
            .order("created_at", desc=True)
            .limit(3)
            .execute()
        )
        return {"candidate_memories": resp.data or []}
    except Exception as e:
        print(f"[memory_retrieval] 回退时间序失败: {e}")
        return {"candidate_memories": []}


def strategy_controller_node(state: AgentState):
    """模块④：策略控制器。"""
    openness = state.get("openness", 0.3)
    confidence = state.get("emotion_data", {}).get("confidence", "低")
    inferred = state.get("emotion_data", {}).get("inferred_emotion", "未知")
    intensity = state.get("emotion_data", {}).get("intensity", "低")

    memory_allowed = openness >= 0.5 and len(
        state.get("candidate_memories", [])
    ) > 0
    if openness < 0.3 or confidence == "低":
        therapy = "active_listening"
        posture = "深度示弱"
    elif "过去" in state["messages"][-1].content and openness > 0.5:
        therapy = "reminiscence"
        posture = "软过渡延伸"
    elif inferred == "焦虑" and intensity == "高":
        therapy = "logotherapy"
        posture = "陪伴倾听"
    else:
        therapy = "default"
        posture = "软过渡延伸"

    # 语气由策略控制器统一决策，不再由模块①直接给出。
    if confidence == "低":
        tone = "试探性共情"
    elif intensity == "高" and inferred in ("孤独", "悲伤", "焦虑", "压抑"):
        tone = "稳定陪伴"
    else:
        tone = "温和倾听"

    strategy_pack = {
        "therapy": therapy,
        "posture": posture,
        "memory_allowed": memory_allowed,
        "tone": tone,
    }
    return {"strategy_pack": strategy_pack}


def generation_node(state: AgentState):
    """模块⑤：对话生成（注入当前用户画像）。"""
    strategy = state.get("strategy_pack", {})
    memory_allowed = strategy.get("memory_allowed", False)
    memory_text = ""
    if memory_allowed:
        mem = state.get("candidate_memories", []) or []
        top = mem[0] if mem else None
        memory_text = f"可引用本会话内相关记忆（最多1条，软过渡）：{top}"

    profile = state.get("user_profile") or _default_user_profile()
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)

    system_prompt = f"""
你是独居老人陪伴助手“小安”，人设是谦逊好奇的晚辈。

【已知的用户长期画像（可能不完整，请自然使用、不要像列表一样全念出来，也不要编造未出现的信息）】
{profile_json}

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


def _extract_profile_patch_llm(
    user_text: str,
    ai_text: str,
    emotion: Dict[str, Any],
    current_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """6A-2：对本轮用户与助手话轮做画像增量提取。"""
    extract_llm = llm if not hasattr(llm, "bind") else llm.bind(temperature=0.2)
    emo = json.dumps(emotion, ensure_ascii=False)
    cur = json.dumps(current_profile, ensure_ascii=False)[:8000]
    prompt = f"""你是信息抽取器，从本轮对话中判断是否可以补充到「用户长期角色画像」。
要求：只输出**严格 JSON**（不要 markdown、不要说明），结构如下。无法判断或没有依据的字段不要填或填空对象。
- 标量字段 gender/age/former_occupation/education：每个可为 {{"value": ..., "confidence": "高/中/低"}}
- 带叙述块 health/family_social/hobbies/personality/habits：每个可为
  {{"summary": {{"value": "string", "confidence": "高/中/低"}}, "append_history": ["可追加的一句客观事实，来自用户本话，不要评价"]}}
规则：
1) 仅使用用户/助手明确可推断的信息，宁缺毋错。
2) append_history 只追加对用户本人事实/经历的陈述，不要重复已有 summary。
3) confidence 为「低」时，对应字段在合并时会被忽略。

当前画像（节选/截断仅供参考）:
{cur}

本轮用户原话: {user_text}
本轮助手回复: {ai_text}
参考情感结果: {emo}

只输出一个 JSON 对象，键名: gender, age, former_occupation, education, health, family_social, hobbies, personality, habits
"""
    response = extract_llm.invoke([HumanMessage(content=prompt)])
    return _safe_json_parse(response.content)


def memory_writer_node(state: AgentState):
    """模块⑥：记忆与画像持久化。
    6A — Supabase：对话行 + 用户画像 upsert
    6B — 知识图谱（未实现，仅占位）
    """
    current_profile = _ensure_profile_shape(state.get("user_profile"))

    if not supabase:
        print("[memory_writer] 未配置 Supabase，跳过 6A。")
        return {"user_profile": current_profile}

    last_user = state["messages"][-2].content if len(state["messages"]) >= 2 else ""
    last_ai = state["messages"][-1].content if state["messages"] else ""
    uid = state["user_id"]
    sid = (state.get("session_id") or "").strip()
    if not last_user or not last_ai:
        print("[memory_writer] 消息不完整，跳过写入。")
        return {"user_profile": current_profile}
    if not sid:
        print("[memory_writer] 无 session_id，跳过 conversation / 画像落库。")
        return {"user_profile": current_profile}

    embed_text = f"用户：{last_user}\n助手：{last_ai}"

    # --- 6A-1: 写入 conversation_turns（向量化在写入后通过 update 回写）---
    turn_id: Optional[int] = None
    conv_payload: Dict[str, Any] = {
        "user_id": uid,
        "session_id": sid,
        "user_text": last_user,
        "ai_text": last_ai,
        "embed_text": embed_text,
        "emotion_label": state["emotion_data"].get("inferred_emotion", "未知"),
        "emotion_data_json": state.get("emotion_data", {}),
        "intensity": state["emotion_data"].get("intensity", "低"),
        "openness": state.get("openness", 0.3),
        "strategy_pack_json": state.get("strategy_pack", {}),
        "rag_hit_count": len(state.get("candidate_memories", []) or []),
        "memory_used": bool(state.get("strategy_pack", {}).get("memory_allowed", False)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        supabase.table("conversation_turns").insert(conv_payload).execute()
    except Exception as e:
        # 兼容旧表结构：若新增列尚未迁移，回退到基础字段写入
        fallback = {
            "user_id": uid,
            "session_id": sid,
            "user_text": last_user,
            "ai_text": last_ai,
            "embed_text": embed_text,
            "emotion_label": state["emotion_data"].get("inferred_emotion", "未知"),
            "intensity": state["emotion_data"].get("intensity", "低"),
            "openness": state.get("openness", 0.3),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            supabase.table("conversation_turns").insert(fallback).execute()
            print(f"[6A-1] 使用基础字段回退写入: {e}")
        except Exception as e2:
            print(f"[6A-1 conversation_turns 写入失败] {e2}")
            return {"user_profile": current_profile}
    try:
        q = (
            supabase.table("conversation_turns")
            .select("id")
            .eq("user_id", uid)
            .eq("session_id", sid)
            .order("id", desc=True)
            .limit(1)
            .execute()
        )
        if q.data:
            turn_id = q.data[0].get("id")
    except Exception as e:
        print(f"[6A-1 取 turn id 失败] {e}")

    # 6A-1b: 对 embed_text 生成向量并回写本行
    efn = _get_embeddings()
    if efn and turn_id is not None:
        try:
            vec = efn.embed_query(embed_text)
            supabase.table("conversation_turns").update(
                {"embedding": vec}
            ).eq("id", turn_id).execute()
        except Exception as e:
            print(f"[6A-1b 向量回写失败] {e}")
    try:
        supabase.table("chat_sessions").update(
            {"last_message_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", sid).execute()
    except Exception as e:
        print(f"[chat_sessions 更新时间] {e}")

    # --- 6A-2: LLM 提取画像 patch ---
    patch: Dict[str, Any] = {}
    try:
        patch = _extract_profile_patch_llm(
            last_user,
            last_ai,
            state.get("emotion_data", {}),
            current_profile,
        )
    except Exception as e:
        print(f"[6A-2 用户画像 LLM 提取失败] {e}")

    # --- 6A-3: 合并到内存结构 ---
    merged = _merge_user_profile(current_profile, patch, turn_id)

    # --- 6A-4: upsert user_profiles ---
    try:
        supabase.table("user_profiles").upsert(
            {
                "user_id": uid,
                "profile": merged,
            },
            on_conflict="user_id",
        ).execute()
    except Exception as e:
        print(f"[6A-4 user_profiles upsert 失败] {e}")
        return {"user_profile": current_profile}

    # --- 6B: 知识图谱（Neo4j 等）---
    # TODO: 从本轮与 merged 中抽取实体/关系并写入图库

    return {"user_profile": merged}


workflow = StateGraph(AgentState)
workflow.add_node("ensure_session", ensure_session_node)
workflow.add_node("load_profile", load_profile_node)
workflow.add_node("decoder", emotion_decoder_node)
workflow.add_node("openness_tracker", openness_tracker_node)
workflow.add_node("memory_retrieval", memory_retrieval_node)
workflow.add_node("strategy_controller", strategy_controller_node)
workflow.add_node("generator", generation_node)
workflow.add_node("memory_writer", memory_writer_node)

workflow.set_entry_point("ensure_session")
workflow.add_edge("ensure_session", "load_profile")
workflow.add_edge("load_profile", "decoder")
workflow.add_edge("decoder", "openness_tracker")
workflow.add_edge("openness_tracker", "memory_retrieval")
workflow.add_edge("memory_retrieval", "strategy_controller")
workflow.add_edge("strategy_controller", "generator")
workflow.add_edge("generator", "memory_writer")
workflow.add_edge("memory_writer", END)

app = workflow.compile()

if __name__ == "__main__":
    print("AI精神慰藉助手已启动 (输入 'quit' 退出)")
    initial_state: AgentState = {
        "messages": [],
        "emotion_data": {},
        "openness": 0.3,
        "candidate_memories": [],
        "strategy_pack": {},
        "user_id": os.getenv("DEFAULT_USER_ID", "demo_user"),
        "session_id": os.getenv("DEFAULT_SESSION_ID", "").strip(),
        "user_profile": _default_user_profile(),
    }

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        initial_state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(initial_state)
        initial_state = result
        print(f"小安: {result['messages'][-1].content}")
