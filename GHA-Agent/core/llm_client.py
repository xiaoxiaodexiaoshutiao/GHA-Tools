from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from core.config_loader import config_loader
from core.run_stats import run_stats
from core.tools import ToolCollection


class LLMClient:
    def __init__(
        self,
        agent_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = agent_config.get("model") or "gpt-5.4"
        self.logger = logger
        self.platform_name = agent_config.get("platform", "default")
        self.platform_config = config_loader.get_platform_config(self.platform_name)
        if self.platform_config.get("provider") != "openai":
            raise ValueError("The model provider must be openai")
        self.reasoning_effort = agent_config.get("reasoning_effort", "low")
        self.verbosity = agent_config.get("verbosity", "low")
        self.max_completion_tokens = agent_config.get("max_completion_tokens")
        self.client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self.client is None:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key or api_key == "your_openai_api_key":
                raise RuntimeError("Set OPENAI_API_KEY in the environment or GHA-Agent/.env")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.openai.com/v1",
                timeout=float(self.platform_config.get("timeout_seconds", 600)),
                max_retries=int(self.platform_config.get("max_retries", 2)),
            )
        return self.client

    def _create_completion(self, messages: List[Dict[str, Any]], **kwargs: Any):
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "reasoning_effort": self.reasoning_effort,
            "verbosity": self.verbosity,
            "store": False,
        }
        if self.max_completion_tokens is not None:
            request["max_completion_tokens"] = self.max_completion_tokens
        request.update(kwargs)
        if request.get("stream"):
            raise ValueError("Streaming is not supported by the parser")
        client = self._get_client()
        started = time.perf_counter()
        response = client.chat.completions.create(**request)
        usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
        run_stats.record_llm_call(
            provider="openai",
            model=response.model,
            elapsed_seconds=time.perf_counter() - started,
            total_tokens=usage.get("total_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_usage=usage,
        )
        if not response.choices:
            raise RuntimeError("OpenAI returned no completion choices")
        choice = response.choices[0]
        if choice.finish_reason not in {"stop", "tool_calls"}:
            raise RuntimeError(f"OpenAI completion did not finish: {choice.finish_reason}")
        if choice.message.refusal:
            raise RuntimeError(f"OpenAI declined the request: {choice.message.refusal}")
        return choice.message

    @staticmethod
    def _text(message) -> str:
        if not message.content or not message.content.strip():
            raise RuntimeError("OpenAI returned an empty final response")
        return message.content.strip()

    def call(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        message = self._create_completion(messages, **kwargs)
        if message.tool_calls:
            raise RuntimeError("Use call_with_tools to execute function calls")
        return self._text(message)

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Union[List[Dict[str, Any]], ToolCollection],
        function_executor,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> str:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        schemas = tools.to_openai_tools() if isinstance(tools, ToolCollection) else tools
        if not schemas:
            return self.call(messages, **kwargs)
        allowed_names = {tool["function"]["name"] for tool in schemas}
        conversation = deepcopy(messages)
        for iteration in range(max_iterations):
            message = self._create_completion(
                conversation,
                tools=schemas,
                tool_choice="auto",
                parallel_tool_calls=False,
                **kwargs,
            )
            if not message.tool_calls:
                return self._text(message)
            conversation.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [call.model_dump(exclude_none=True) for call in message.tool_calls],
            })
            for tool_call in message.tool_calls:
                if tool_call.type != "function":
                    raise ValueError("Only function tool calls are supported")
                name = tool_call.function.name
                if name not in allowed_names:
                    raise ValueError(f"Unknown tool call: {name}")
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Tool call JSON parsing failed for {name}") from exc
                if not isinstance(arguments, dict):
                    raise ValueError(f"Tool call arguments must be an object: {name}")
                if self.logger:
                    self.logger.debug("Tool iteration %s: %s", iteration + 1, name)
                result = function_executor.execute(name, arguments)
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                })
        raise RuntimeError(f"Tool calling exceeded {max_iterations} iterations without a final answer")
