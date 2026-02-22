"""
llm/memory.py — Per-session sliding window conversation memory.

Stores the last N turns of conversation per session_id.
In-process (dict) storage — survives the request lifecycle but not restarts.

Why in-process, not Redis?
For a local dev/assignment setup, in-process is simpler and has zero infra cost.
The migration path to Redis is exactly one function swap:
  get_history()  -> redis.lrange(session_id, 0, -1)
  save_turn()    -> redis.rpush(session_id, ...) + redis.expire(...)

Trade-off: Memory is lost on server restart. For a demo this is fine.
For production, add Redis (already in the AWS architecture doc).

Context window management:
The Llama 3.2 3B model has a 128K context window but we cap history at
3 turns (6 messages) to keep total prompt tokens predictable and fast.
At 3 turns: ~600 tokens for history + ~400 system + ~2400 context = ~3400 total.
"""
from collections import defaultdict, deque

# session_id -> deque of {"role": ..., "content": ...} dicts
_SESSIONS: dict[str, deque] = defaultdict(lambda: deque(maxlen=6))  # 3 turns = 6 messages

MAX_TURNS = 3  # Number of (user, assistant) pairs to keep


def get_history(session_id: str) -> list[dict]:
    """
    Return conversation history for a session as a list of message dicts.
    Returns empty list for new sessions.
    """
    return list(_SESSIONS[session_id])


def save_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    """
    Append a user+assistant turn to the session history.
    The deque's maxlen automatically evicts oldest messages.
    """
    _SESSIONS[session_id].append({"role": "user", "content": user_message})
    _SESSIONS[session_id].append({"role": "assistant", "content": assistant_message})


def clear_session(session_id: str) -> None:
    """Clear history for a session (e.g. when user uploads new documents)."""
    if session_id in _SESSIONS:
        _SESSIONS[session_id].clear()


def get_session_count() -> int:
    """Number of active sessions — for observability."""
    return len(_SESSIONS)
