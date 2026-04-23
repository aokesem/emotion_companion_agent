"use client";

import { useEffect, useMemo, useState } from "react";

type SessionItem = {
  id: string;
  title?: string | null;
  last_message_at?: string | null;
  created_at?: string | null;
};

type HistoryItem = {
  id: number;
  user_text: string;
  ai_text: string;
  created_at?: string;
};

type ChatDebug = {
  openness?: number;
  emotion_data?: unknown;
  strategy_pack?: unknown;
  candidate_memories?: unknown;
  user_profile?: unknown;
};

type TimelineItem = {
  id: number;
  created_at?: string;
  openness?: number;
  strategy_pack?: unknown;
};

type Message = {
  role: "user" | "assistant";
  content: string;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

export default function Home() {
  const [userId, setUserId] = useState("demo_user");
  const [users, setUsers] = useState<string[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [input, setInput] = useState("");
  const [debugData, setDebugData] = useState<ChatDebug | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [renamingId, setRenamingId] = useState("");
  const [renamingTitle, setRenamingTitle] = useState("");

  const sessionLabel = useMemo(
    () => sessions.find((s) => s.id === sessionId)?.title || sessionId,
    [sessions, sessionId]
  );

  async function loadUsers() {
    const resp = await fetch(`${API_BASE}/users`);
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    const data = await resp.json();
    const items: string[] = data.items || [];
    setUsers(items);
    if ((!userId || !items.includes(userId)) && items.length > 0) {
      setUserId(items[0]);
    }
  }

  async function loadSessions(targetUserId: string) {
    setError("");
    const resp = await fetch(
      `${API_BASE}/chat/sessions?user_id=${encodeURIComponent(targetUserId)}`
    );
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    const data = await resp.json();
    const items: SessionItem[] = data.items || [];
    setSessions(items);
    if (!sessionId && items.length > 0) {
      setSessionId(items[0].id);
    }
  }

  async function loadHistory(targetUserId: string, targetSessionId: string) {
    const resp = await fetch(
      `${API_BASE}/chat/history?user_id=${encodeURIComponent(
        targetUserId
      )}&session_id=${encodeURIComponent(targetSessionId)}`
    );
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    const data = await resp.json();
    const rows: HistoryItem[] = data.items || [];
    const built: Message[] = [];
    for (const row of rows) {
      built.push({ role: "user", content: row.user_text });
      built.push({ role: "assistant", content: row.ai_text });
    }
    setMessages(built);
  }

  async function loadTimeline(targetUserId: string, targetSessionId: string) {
    const resp = await fetch(
      `${API_BASE}/chat/timeline?user_id=${encodeURIComponent(
        targetUserId
      )}&session_id=${encodeURIComponent(targetSessionId)}`
    );
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    const data = await resp.json();
    setTimeline(data.items || []);
  }

  async function createSession() {
    setError("");
    const resp = await fetch(`${API_BASE}/chat/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
      }),
    });
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    const data = await resp.json();
    const newSessionId: string = data.session_id;
    await loadSessions(userId);
    setSessionId(newSessionId);
    setMessages([]);
    setTimeline([]);
    setDebugData(null);
  }

  async function renameSession(targetId: string, title: string) {
    const nextTitle = title.trim();
    if (!nextTitle) {
      setError("会话名称不能为空");
      return;
    }
    const resp = await fetch(`${API_BASE}/chat/sessions/${encodeURIComponent(targetId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: nextTitle }),
    });
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    await loadSessions(userId);
    setRenamingId("");
    setRenamingTitle("");
  }

  async function deleteSession(targetId: string) {
    const ok = window.confirm("确定要删除（归档）这个会话吗？");
    if (!ok) return;
    const resp = await fetch(`${API_BASE}/chat/sessions/${encodeURIComponent(targetId)}`, {
      method: "DELETE",
    });
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    const remaining = sessions.filter((s) => s.id !== targetId);
    await loadSessions(userId);
    if (sessionId === targetId) {
      const next = remaining[0];
      setSessionId(next?.id || "");
      if (!next) {
        setMessages([]);
        setTimeline([]);
        setDebugData(null);
      }
    }
  }

  async function sendMessage() {
    const text = input.trim();
    if (!text || busy) return;
    setBusy(true);
    setError("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    try {
      const resp = await fetch(`${API_BASE}/chat/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          session_id: sessionId || undefined,
          message: text,
          debug: true,
        }),
      });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      const data = await resp.json();
      setSessionId(data.session_id);
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply_text }]);
      setDebugData(data.debug ?? null);
      await loadSessions(userId);
      await loadTimeline(userId, data.session_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "发送失败");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadUsers().catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadSessions(userId).catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  useEffect(() => {
    if (!sessionId) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadHistory(userId, sessionId).catch((e) => setError(e.message));
    loadTimeline(userId, sessionId).catch((e) => setError(e.message));
  }, [sessionId, userId]);

  return (
    <div className="min-h-screen bg-zinc-100 text-zinc-900">
      <div className="mx-auto grid max-w-[1500px] grid-cols-12 gap-4 p-4">
        <aside className="col-span-3 rounded-xl bg-white p-4 shadow">
          <h2 className="mb-3 text-lg font-semibold">用户与会话</h2>
          <label className="mb-1 block text-sm text-zinc-600">用户（下拉）</label>
          <select
            className="mb-2 w-full rounded border px-2 py-1"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          >
            {users.map((u) => (
              <option key={u} value={u}>
                {u}
              </option>
            ))}
          </select>
          <button
            className="mb-3 w-full rounded border border-zinc-300 px-3 py-1 text-sm"
            onClick={() => loadUsers().catch((e) => setError(e.message))}
          >
            刷新用户列表
          </button>

          <label className="mb-1 block text-sm text-zinc-600">或手动输入用户 ID</label>
          <input
            className="mb-3 w-full rounded border px-2 py-1"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
          <button
            className="mb-3 w-full rounded bg-blue-600 px-3 py-2 text-white"
            onClick={() => createSession().catch((e) => setError(e.message))}
          >
            新建会话
          </button>
          <div className="space-y-2">
            {sessions.map((s) => (
              <div
                key={s.id}
                className={`rounded border px-2 py-2 text-left text-sm ${
                  sessionId === s.id ? "border-blue-500 bg-blue-50" : "border-zinc-200"
                }`}
              >
                <button className="w-full text-left" onClick={() => setSessionId(s.id)}>
                  <div className="font-medium">{s.title || s.id.slice(0, 8)}</div>
                  <div className="text-xs text-zinc-500">{s.last_message_at || s.created_at}</div>
                </button>
                {renamingId === s.id ? (
                  <div className="mt-2 flex gap-1">
                    <input
                      className="min-w-0 flex-1 rounded border px-2 py-1 text-xs"
                      value={renamingTitle}
                      onChange={(e) => setRenamingTitle(e.target.value)}
                    />
                    <button
                      className="rounded bg-green-600 px-2 py-1 text-xs text-white"
                      onClick={() =>
                        renameSession(s.id, renamingTitle).catch((e) =>
                          setError(e instanceof Error ? e.message : "重命名失败")
                        )
                      }
                    >
                      保存
                    </button>
                    <button
                      className="rounded border px-2 py-1 text-xs"
                      onClick={() => {
                        setRenamingId("");
                        setRenamingTitle("");
                      }}
                    >
                      取消
                    </button>
                  </div>
                ) : (
                  <div className="mt-2 flex gap-1">
                    <button
                      className="rounded border px-2 py-1 text-xs"
                      onClick={() => {
                        setRenamingId(s.id);
                        setRenamingTitle(s.title || "");
                      }}
                    >
                      重命名
                    </button>
                    <button
                      className="rounded border border-red-300 px-2 py-1 text-xs text-red-600"
                      onClick={() =>
                        deleteSession(s.id).catch((e) =>
                          setError(e instanceof Error ? e.message : "删除失败")
                        )
                      }
                    >
                      删除
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </aside>

        <main className="col-span-6 rounded-xl bg-white p-4 shadow">
          <h2 className="mb-2 text-lg font-semibold">对话区</h2>
          <div className="mb-2 text-sm text-zinc-600">当前会话：{sessionLabel || "未选择"}</div>
          <div className="h-[70vh] space-y-3 overflow-y-auto rounded border p-3">
            {messages.map((m, i) => (
              <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
                <div
                  className={`inline-block max-w-[80%] rounded px-3 py-2 text-sm ${
                    m.role === "user" ? "bg-blue-600 text-white" : "bg-zinc-200 text-zinc-900"
                  }`}
                >
                  {m.content}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 flex gap-2">
            <input
              className="flex-1 rounded border px-3 py-2"
              placeholder="输入消息..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") sendMessage().catch((err) => setError(err.message));
              }}
            />
            <button
              className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
              disabled={busy}
              onClick={() => sendMessage().catch((e) => setError(e.message))}
            >
              {busy ? "发送中..." : "发送"}
            </button>
          </div>
          {error && <div className="mt-2 text-sm text-red-600">{error}</div>}
        </main>

        <aside className="col-span-3 rounded-xl bg-white p-4 shadow">
          <h2 className="mb-3 text-lg font-semibold">调试信息</h2>
          <div className="space-y-3 text-sm">
            <div>
              <div className="font-medium">Openness</div>
              <div>{debugData?.openness ?? "-"}</div>
            </div>
            <div>
              <div className="font-medium">策略</div>
              <pre className="max-h-32 overflow-auto rounded bg-zinc-100 p-2 text-xs">
                {JSON.stringify(debugData?.strategy_pack ?? {}, null, 2)}
              </pre>
            </div>
            <div>
              <div className="font-medium">RAG 命中</div>
              <pre className="max-h-40 overflow-auto rounded bg-zinc-100 p-2 text-xs">
                {JSON.stringify(debugData?.candidate_memories ?? [], null, 2)}
              </pre>
            </div>
            <div>
              <div className="font-medium">用户画像</div>
              <pre className="max-h-56 overflow-auto rounded bg-zinc-100 p-2 text-xs">
                {JSON.stringify(debugData?.user_profile ?? {}, null, 2)}
              </pre>
            </div>
            <div>
              <div className="font-medium">会话级时间线（Openness / Strategy）</div>
              <pre className="max-h-56 overflow-auto rounded bg-zinc-100 p-2 text-xs">
                {JSON.stringify(timeline, null, 2)}
              </pre>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
