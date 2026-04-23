import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from main_1 import app as graph_app, supabase, _default_user_profile


class ChatSendRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    debug: bool = True


class SessionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    title: Optional[str] = None


class SessionRenameRequest(BaseModel):
    title: str = Field(..., min_length=1)


class SessionState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    emotion_data: Dict[str, Any] = Field(default_factory=dict)
    openness: float = 0.3
    candidate_memories: List[Dict[str, Any]] = Field(default_factory=list)
    strategy_pack: Dict[str, Any] = Field(default_factory=dict)
    user_id: str
    session_id: str
    user_profile: Dict[str, Any] = Field(default_factory=_default_user_profile)


app = FastAPI(title="Emotion Companion API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 进程内会话状态缓存（阶段1：开发/小范围内测足够）
STATE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _build_initial_state(user_id: str, session_id: str) -> Dict[str, Any]:
    return {
        "messages": [],
        "emotion_data": {},
        "openness": 0.3,
        "candidate_memories": [],
        "strategy_pack": {},
        "user_id": user_id,
        "session_id": session_id,
        "user_profile": _default_user_profile(),
    }


def _default_session_title() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_json_value(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _restore_state_from_db(user_id: str, session_id: str) -> Dict[str, Any]:
    state = _build_initial_state(user_id, session_id)
    if not supabase:
        return state
    try:
        resp = (
            supabase.table("conversation_turns")
            .select(
                "id,user_text,ai_text,openness,emotion_data_json,strategy_pack_json"
            )
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .order("id", desc=False)
            .limit(200)
            .execute()
        )
        rows = resp.data or []
        for row in rows:
            ut = row.get("user_text")
            at = row.get("ai_text")
            if ut:
                state["messages"].append(HumanMessage(content=ut))
            if at:
                state["messages"].append(AIMessage(content=at))
        if rows:
            last = rows[-1]
            state["openness"] = float(last.get("openness") or 0.3)
            state["emotion_data"] = _safe_json_value(last.get("emotion_data_json"))
            state["strategy_pack"] = _safe_json_value(last.get("strategy_pack_json"))
    except Exception:
        pass
    return state


def _ensure_session_for_user(user_id: str, session_id: Optional[str]) -> str:
    sid = (session_id or "").strip()
    if sid:
        return sid
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase 未配置，无法自动创建会话")
    title = os.getenv("DEFAULT_SESSION_TITLE", "").strip() or _default_session_title()
    try:
        supabase.table("chat_sessions").insert(
            {"user_id": user_id, "title": title}
        ).execute()
        got = (
            supabase.table("chat_sessions")
            .select("id")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if got.data:
            return str(got.data[0]["id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {e}")
    raise HTTPException(status_code=500, detail="创建会话失败：未获取到 session_id")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/users")
def list_users():
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase 未配置")
    users: set[str] = set()
    try:
        cs = (
            supabase.table("chat_sessions")
            .select("user_id")
            .order("created_at", desc=True)
            .limit(1000)
            .execute()
        )
        for row in cs.data or []:
            uid = (row.get("user_id") or "").strip()
            if uid:
                users.add(uid)
    except Exception:
        pass
    try:
        up = supabase.table("user_profiles").select("user_id").limit(1000).execute()
        for row in up.data or []:
            uid = (row.get("user_id") or "").strip()
            if uid:
                users.add(uid)
    except Exception:
        pass
    return {"items": sorted(users)}


@app.get("/chat/sessions")
def list_sessions(user_id: str, include_archived: bool = False):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase 未配置")
    try:
        query = (
            supabase.table("chat_sessions")
            .select("id,title,last_message_at,created_at,is_archived")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(100)
        )
        if not include_archived:
            query = query.eq("is_archived", False)
        resp = query.execute()
        return {"items": resp.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"拉取会话失败: {e}")


@app.post("/chat/sessions")
def create_session(payload: SessionCreateRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase 未配置")
    try:
        title = payload.title or os.getenv("DEFAULT_SESSION_TITLE", "").strip() or _default_session_title()
        supabase.table("chat_sessions").insert(
            {"user_id": payload.user_id, "title": title}
        ).execute()
        got = (
            supabase.table("chat_sessions")
            .select("id,title,last_message_at,created_at,is_archived")
            .eq("user_id", payload.user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if got.data:
            item = got.data[0]
            return {"session_id": item["id"], "session": item}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {e}")
    raise HTTPException(status_code=500, detail="创建会话失败：无返回数据")


@app.patch("/chat/sessions/{session_id}")
def rename_session(session_id: str, payload: SessionRenameRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase 未配置")
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="会话名称不能为空")
    try:
        supabase.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()
        got = (
            supabase.table("chat_sessions")
            .select("id,title,last_message_at,created_at,is_archived")
            .eq("id", session_id)
            .limit(1)
            .execute()
        )
        if got.data:
            return {"session": got.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重命名会话失败: {e}")
    raise HTTPException(status_code=404, detail="会话不存在")


@app.delete("/chat/sessions/{session_id}")
def delete_session(session_id: str):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase 未配置")
    try:
        supabase.table("chat_sessions").update({"is_archived": True}).eq("id", session_id).execute()
        drop_keys = [k for k in STATE_CACHE.keys() if k[1] == session_id]
        for key in drop_keys:
            STATE_CACHE.pop(key, None)
        return {"ok": True, "session_id": session_id, "archived": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {e}")


@app.get("/chat/history")
def get_history(user_id: str, session_id: str, limit: int = 50):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase 未配置")
    try:
        n = max(1, min(limit, 200))
        resp = (
            supabase.table("conversation_turns")
            .select("id,user_text,ai_text,emotion_label,intensity,openness,created_at")
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .order("id", desc=False)
            .limit(n)
            .execute()
        )
        return {"items": resp.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"拉取历史失败: {e}")


@app.get("/chat/timeline")
def get_timeline(user_id: str, session_id: str, limit: int = 100):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase 未配置")
    try:
        n = max(1, min(limit, 500))
        resp = (
            supabase.table("conversation_turns")
            .select("id,created_at,openness,strategy_pack_json")
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .order("id", desc=False)
            .limit(n)
            .execute()
        )
        items: List[Dict[str, Any]] = []
        for row in resp.data or []:
            items.append(
                {
                    "id": row.get("id"),
                    "created_at": row.get("created_at"),
                    "openness": row.get("openness"),
                    "strategy_pack": _safe_json_value(row.get("strategy_pack_json")),
                }
            )
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"拉取时间线失败: {e}")


@app.post("/chat/send")
def send_chat(payload: ChatSendRequest):
    session_id = _ensure_session_for_user(payload.user_id, payload.session_id)
    cache_key = (payload.user_id, session_id)
    state = STATE_CACHE.get(cache_key)
    if not state:
        state = _restore_state_from_db(payload.user_id, session_id)

    state["messages"].append(HumanMessage(content=payload.message))
    result = graph_app.invoke(state)
    STATE_CACHE[cache_key] = result

    ai_text = ""
    if result.get("messages"):
        ai_text = result["messages"][-1].content

    response: Dict[str, Any] = {
        "user_id": payload.user_id,
        "session_id": session_id,
        "reply_text": ai_text,
    }
    if payload.debug:
        response["debug"] = {
            "openness": result.get("openness"),
            "emotion_data": result.get("emotion_data"),
            "strategy_pack": result.get("strategy_pack"),
            "candidate_memories": result.get("candidate_memories"),
            "user_profile": result.get("user_profile"),
        }
    return response

