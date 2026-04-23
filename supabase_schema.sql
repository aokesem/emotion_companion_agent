-- 参考用 SQL（你已在 Supabase 执行过 则无需重复执行，以下便于新环境与对照）
-- 1) 扩展: create extension if not exists vector with schema extensions;

-- 2) 会话
create table if not exists public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  title text,
  is_archived boolean not null default false,
  last_message_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists idx_chat_sessions_user_time
  on public.chat_sessions (user_id, created_at desc);

-- 3) 若已有 conversation_turns 仅为旧版，可用手动执行补充列，例如:
-- alter table public.conversation_turns add column if not exists session_id uuid;
-- alter table public.conversation_turns add column if not exists embed_text text;
-- alter table public.conversation_turns add column if not exists embedding vector(1536);
-- alter table public.conversation_turns add column if not exists emotion_data_json jsonb;
-- alter table public.conversation_turns add column if not exists strategy_pack_json jsonb;
-- alter table public.conversation_turns add column if not exists rag_hit_count int;
-- alter table public.conversation_turns add column if not exists memory_used boolean;
-- 再补外键/非空/索引与 RPC，见项目 Supabase 控制台已执行的脚本。

-- 4) 用户画像
create table if not exists public.user_profiles (
  user_id text primary key,
  profile jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 5) 向量 RPC：名称须为 match_conversation_turns，与 main_1 中 .rpc 调用一致（实现见自建 SQL）
