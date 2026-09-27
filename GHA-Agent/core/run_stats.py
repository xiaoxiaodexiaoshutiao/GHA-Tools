from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import local
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if cleaned.isdigit():
            return int(cleaned)
    return None


class RunStats:
    def __init__(self) -> None:
        self._local = local()

    def _new_state(self, started_at: Optional[str] = None) -> Dict[str, Any]:
        return {
            "started_at": started_at or _utc_now(),
            "completed_at": None,
            "parser_elapsed_seconds": 0.0,
            "llm_call_count": 0,
            "llm_elapsed_seconds": 0.0,
            "total_tokens": 0,
            "token_count_available": False,
            "llm_calls": [],
        }

    def reset(self, started_at: Optional[str] = None) -> None:
        self._local.state = self._new_state(started_at)

    def _state(self) -> Dict[str, Any]:
        state = getattr(self._local, "state", None)
        if state is None:
            state = self._new_state()
            self._local.state = state
        return state

    def record_llm_call(
        self,
        *,
        provider: str,
        model: Optional[str],
        elapsed_seconds: Optional[float] = None,
        total_tokens: Any = None,
        prompt_tokens: Any = None,
        completion_tokens: Any = None,
        raw_usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self._state()
        total_token_count = _to_int(total_tokens)
        prompt_token_count = _to_int(prompt_tokens)
        completion_token_count = _to_int(completion_tokens)
        elapsed = float(elapsed_seconds or 0.0)

        state["llm_call_count"] += 1
        state["llm_elapsed_seconds"] += elapsed
        if total_token_count is not None:
            state["total_tokens"] += total_token_count
            state["token_count_available"] = True

        call_record: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "elapsed_seconds": round(elapsed, 3),
            "total_tokens": total_token_count,
        }
        if prompt_token_count is not None:
            call_record["prompt_tokens"] = prompt_token_count
        if completion_token_count is not None:
            call_record["completion_tokens"] = completion_token_count
        if raw_usage:
            call_record["raw_usage"] = raw_usage

        state["llm_calls"].append(call_record)

    def snapshot(
        self,
        *,
        parser_elapsed_seconds: Optional[float] = None,
        completed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = deepcopy(self._state())
        if parser_elapsed_seconds is not None:
            state["parser_elapsed_seconds"] = round(float(parser_elapsed_seconds), 3)
        else:
            state["parser_elapsed_seconds"] = round(float(state["parser_elapsed_seconds"]), 3)

        state["completed_at"] = completed_at or state.get("completed_at") or _utc_now()
        state["llm_elapsed_seconds"] = round(float(state["llm_elapsed_seconds"]), 3)
        return state


run_stats = RunStats()
