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
