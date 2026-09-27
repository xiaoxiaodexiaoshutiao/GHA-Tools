from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from core.llm_client import LLMClient
from core.config_loader import config_loader
from core.checkpoint_manager import CheckpointManager
import time
import logging


class BaseAgent(ABC):
    _shared_logger: Optional[logging.Logger] = None

    _shared_checkpoint_manager: Optional[CheckpointManager] = None

    @classmethod
    def set_logger(cls, logger: logging.Logger):
        cls._shared_logger = logger

    @classmethod
    def get_logger(cls) -> Optional[logging.Logger]:
        return cls._shared_logger

    @classmethod
    def set_checkpoint_manager(cls, checkpoint_manager: CheckpointManager):
        cls._shared_checkpoint_manager = checkpoint_manager

    @classmethod
    def get_checkpoint_manager(cls) -> Optional[CheckpointManager]:
        return cls._shared_checkpoint_manager

    def __init__(self, agent_name: str, agent_config: Optional[Dict[str, Any]] = None):
        self.agent_name = agent_name

        if agent_config is None:
            agent_config = config_loader.get_agent_config(agent_name)

        self.config = agent_config

        try:
            self.prompts = config_loader.get_agent_prompt(agent_name)
        except KeyError:
            self.prompts = {}

        if 'model' in agent_config:
            self.llm_client = LLMClient(
                agent_config,
                logger=self._shared_logger,
            )
        else:
            self.llm_client = None

        if self._shared_checkpoint_manager is None:
            self._shared_checkpoint_manager = CheckpointManager()

        self.log(f"Initialized Agent: {agent_name}")

    def call_llm(self, messages: List[Dict[str, str]],
                 log_metadata: Dict[str, str] = None,
                 call_index: int = 0,
                 extra_info: Dict[str, Any] = None,
                 **kwargs) -> str:
        if self.llm_client is None:
            raise RuntimeError(f"Agent {self.agent_name} has no LLM client configured")

        start_time = time.time()
        response = self.llm_client.call(messages, **kwargs)
        elapsed_time = time.time() - start_time

        self.log(f"LLM call completed, elapsed: {elapsed_time:.2f}s")

        if log_metadata and self._shared_checkpoint_manager:
            try:
                self._shared_checkpoint_manager.save_llm_response(
                    agent_name=self.agent_name,
                    response=response,
                    log_metadata=log_metadata,
                    call_index=call_index,
                    extra_info=extra_info
                )
            except Exception as e:
                self.log(f"Failed to save LLM response: {e}", "warning")

        return response

    def call_llm_with_tools(self, messages: List[Dict[str, str]],
                            tools,
                            function_executor,
                            log_metadata: Dict[str, str] = None,
                            call_index: int = 0,
                            extra_info: Dict[str, Any] = None,
                            max_iterations: int = 10,
                            **kwargs) -> str:
        if self.llm_client is None:
            raise RuntimeError(f"Agent {self.agent_name} has no LLM client configured")

        start_time = time.time()
        response = self.llm_client.call_with_tools(
            messages=messages,
            tools=tools,
            function_executor=function_executor,
            max_iterations=max_iterations,
            **kwargs
        )
        elapsed_time = time.time() - start_time

        self.log(f"LLM call completed (with tools), elapsed: {elapsed_time:.2f}s")

        if log_metadata and self._shared_checkpoint_manager:
            try:
                self._shared_checkpoint_manager.save_llm_response(
                    agent_name=self.agent_name,
                    response=response,
                    log_metadata=log_metadata,
                    call_index=call_index,
                    extra_info=extra_info
                )
            except Exception as e:
                self.log(f"Failed to save LLM response: {e}", "warning")

        return response

    def format_prompt(self, template_name: str = 'user_prompt_template', **kwargs) -> str:
        if template_name not in self.prompts:
            raise KeyError(f"Prompt template not found: {template_name}")

        template = self.prompts[template_name]
        return template.format(**kwargs)

    def get_system_prompt(self) -> Optional[str]:
        return self.prompts.get('system_prompt')

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict)

    def retry_on_parse_failure(self, llm_call_func, parse_func, max_retries: int = 3):
        last_response = None
        last_error = None

        for attempt in range(max_retries):
            try:
                response = llm_call_func()
                last_response = response

                result = parse_func(response)

                if attempt > 0:
                    self.log(f"Retry successful (attempt {attempt + 1})", "success")
                return True, result, response

            except ValueError as e:
                if self._is_non_retryable_llm_error(e):
                    self.log(f"Non-retryable LLM error: {str(e)}", "error")
                    raise

                last_error = e
                error_msg = str(e)

                if "tool call" in error_msg.lower() or "json" in error_msg.lower():
                    if attempt < max_retries - 1:
                        self.log(f"Tool call parsing failed (attempt {attempt + 1}), will retry: {error_msg}", "warning")
                    else:
                        self.log(f"Tool call parsing failed (reached max retries {max_retries}): {error_msg}", "error")
                else:
                    if attempt < max_retries - 1:
                        self.log(f"Response parsing failed (attempt {attempt + 1}), will retry: {error_msg}", "warning")
                    else:
                        self.log(f"Response parsing failed (reached max retries {max_retries}): {error_msg}", "error")
                        if last_response:
                            self.log(f"Last LLM response: {last_response[:500]}...", "error")

            except Exception as e:
                if self._is_non_retryable_llm_error(e):
                    self.log(f"Non-retryable LLM error: {str(e)}", "error")
                    raise RuntimeError(f"Non-retryable LLM error: {str(e)}") from e

                last_error = e
                error_type = type(e).__name__
                if attempt < max_retries - 1:
                    self.log(f"{error_type} exception (attempt {attempt + 1}), will retry: {str(e)}", "warning")
                else:
                    self.log(f"{error_type} exception (reached max retries {max_retries}): {str(e)}", "error")
                    if last_response:
                        self.log(f"Last LLM response: {last_response[:500]}...", "error")

        return False, None, last_response

    def _is_non_retryable_llm_error(self, error: Exception) -> bool:
        error_type = type(error).__name__.lower()
        message = str(error).lower()
        non_retryable_markers = (
            "openai_api_key",
            "authenticationerror",
            "permissiondeniederror",
            "invalid api key",
            "invalid key",
            "unauthorized",
            "401",
        )
        return any(marker in error_type or marker in message for marker in non_retryable_markers)

    def log(self, message: str, level: str = "info"):
        prefix = f"[{self.agent_name}]"
        full_message = f"{prefix} {message}"

        if self._shared_logger:
            if level == "error":
                self._shared_logger.error(full_message)
            elif level == "warning":
                self._shared_logger.warning(full_message)
            elif level == "debug":
                self._shared_logger.debug(full_message)
            elif level == "success":
                self._shared_logger.info(f"✅ {full_message}")
            else:
                self._shared_logger.info(full_message)

        if level == "error":
            print(f"❌ {prefix} {message}")
        elif level == "warning":
            print(f"⚠️  {prefix} {message}")
        elif level == "success":
            print(f"✅ {prefix} {message}")
        elif level == "debug":
            import os
            if os.environ.get('DEBUG'):
                print(f"🔍 {prefix} {message}")
        else:
            print(f"ℹ️  {prefix} {message}")
