create table if not exists conversation_turns (
  id bigint generated always as identity primary key,
  user_id text not null,
  user_text text not null,
  ai_text text not null,
  emotion_label text,
  intensity text,
  openness double precision default 0.3,
  created_at timestamptz default now()
);

create index if not exists idx_conversation_turns_user_time
on conversation_turns (user_id, created_at desc);


-- 用户画像：一行一用户，profile 为 JSONB（与 main_1.py 中结构一致；若你已在控制台执行过可忽略）
create table if not exists public.user_profiles (
  user_id text primary key,
  profile jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 与 Python 中 set_updated_at 一致时可用；若表已存在且无触发器，可在控制台单独执行
-- create trigger trg_user_profiles_updated_at
--   before update on public.user_profiles
--   for each row execute function public.set_updated_at();
