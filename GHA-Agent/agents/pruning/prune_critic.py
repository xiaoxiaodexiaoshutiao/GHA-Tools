import json
import re
from typing import List, Set, Dict, Optional, Tuple
from .types import LogLine, RCAResult, ProtectResult


class PruneCritic:
    def __init__(self, llm_client=None, validation_mode: str = "evidence"):
        self.llm_client = llm_client
        self.validation_mode = validation_mode

    def validate(self, log_lines: List[LogLine],
                delete_ids: Set[int],
                protect: ProtectResult) -> Set[int]:
        if not self.llm_client:
            return self._rule_based_validation(log_lines, delete_ids, protect)

        line_dict = {line.no: line for line in log_lines}

        if self.validation_mode == "rca":
            return self._rca_validation(log_lines, delete_ids, line_dict)
        else:
            return self._evidence_validation(log_lines, delete_ids, line_dict)

    def _rule_based_validation(self, log_lines: List[LogLine],
                               delete_ids: Set[int],
                               protect: ProtectResult) -> Set[int]:
        restore_ids = set()

        hard_deleted = delete_ids & protect.hard_protect_set
        restore_ids.update(hard_deleted)

        remaining_lines = set(line.no for line in log_lines) - delete_ids

        has_error_evidence = False
        for line in log_lines:
            if line.no in remaining_lines:
                if self._contains_error_keyword(line.text):
                    has_error_evidence = True
                    break

        if not has_error_evidence:
            soft_deleted = delete_ids & protect.soft_protect_set
            restore_ids.update(soft_deleted)

        return restore_ids

    def _evidence_validation(self, log_lines: List[LogLine],
                            delete_ids: Set[int],
                            line_dict: Dict[int, LogLine],
                            max_retries: int = 3) -> Set[int]:
        pruned_content = self._build_pruned_content(log_lines, delete_ids)

        original_content = "\n".join(line.text for line in log_lines)

        prompt = f"""Please validate whether the following pruned log still contains sufficient error evidence.

## Pruned Log
{pruned_content}

## Validation Questions
Please answer the following questions (answer YES or NO only):

1. Can you identify the failed command or step from the pruned log?
2. Can you find the first/primary error message?
3. Can you find the exit code or exception type?

If the answer to any of the above questions is NO, please list the line numbers that need to be restored.

## Output Format (strict JSON)
```json
{{
  "all_evidence_preserved": true/false,
  "restore_line_nos": []
}}
```

If evidence is sufficient, restore_line_nos should be an empty array.
If evidence is insufficient, please infer which lines need to be restored based on the original log."""

        messages = [
            {"role": "system", "content": "You are a professional log analysis expert, responsible for validating whether pruned logs retain sufficient error evidence."},
            {"role": "user", "content": prompt}
        ]

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm_client.call(messages)

                restore_ids, parse_success = self._parse_validation_response_with_status(response)

                if parse_success:
                    if attempt > 1:
                        print(f"PruneCritic: Successfully parsed on attempt {attempt}")
                    return restore_ids
                else:
                    last_error = "Unable to extract valid JSON from response"
                    if attempt < max_retries:
                        print(f"PruneCritic: Failed to parse response (attempt {attempt}/{max_retries}): {last_error}")
                        print(f"  Response preview: {response[:200]}...")
                        continue

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    print(f"PruneCritic: LLM call or parsing failed (attempt {attempt}/{max_retries}): {e}")
                    continue

        error_msg = f"PruneCritic: Maximum retries ({max_retries}) reached, unable to parse LLM response: {last_error}"
        print(f"X {error_msg}")
        raise RuntimeError(error_msg)

    def _rca_validation(self, log_lines: List[LogLine],
                       delete_ids: Set[int],
                       line_dict: Dict[int, LogLine]) -> Set[int]:
        original_content = "\n".join(line.text for line in log_lines)
        original_rca = self._extract_rca(original_content)

        pruned_content = self._build_pruned_content(log_lines, delete_ids)
        pruned_rca = self._extract_rca(pruned_content)

        if original_rca.core_fields_match(pruned_rca):
            return set()

        return self._find_lines_to_restore(
            log_lines, delete_ids, original_rca, pruned_rca
        )

    def _extract_rca(self, log_content: str, max_retries: int = 3) -> RCAResult:
        if not self.llm_client:
            return RCAResult()

        prompt = f"""Please extract Root Cause Analysis (RCA) information from the following log.

## Log Content
{log_content[:5000]}  # Length limited

## Output Format (strict JSON)
```json
{{
  "failed_step": "Name of the failed step",
  "failed_command": "Failed command",
  "primary_error_message": "Primary error message",
  "exit_code": 1,
  "exception_type": "Exception type (if any)",
  "file_path": "Related file path (if any)",
  "line_number": 10
}}
```

Use null or empty string if a field cannot be determined."""

        messages = [
            {"role": "system", "content": "You are a professional log analysis expert."},
            {"role": "user", "content": prompt}
        ]

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm_client.call(messages)

                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())

                    if any(key in data for key in ['failed_step', 'failed_command', 'primary_error_message', 'exit_code']):
                        if attempt > 1:
                            print(f"PruneCritic RCA: Successfully parsed on attempt {attempt}")
                        return RCAResult(
                            failed_step=data.get("failed_step", ""),
                            failed_command=data.get("failed_command", ""),
                            primary_error_message=data.get("primary_error_message", ""),
                            exit_code=data.get("exit_code"),
                            exception_type=data.get("exception_type", ""),
                            file_path=data.get("file_path", ""),
                            line_number=data.get("line_number")
                        )

                last_error = "Unable to extract valid RCA JSON from response"
                if attempt < max_retries:
                    print(f"PruneCritic RCA: Failed to parse response (attempt {attempt}/{max_retries}): {last_error}")
                    print(f"  Response preview: {response[:200]}...")
                    continue

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                if attempt < max_retries:
                    print(f"PruneCritic RCA: JSON parsing failed (attempt {attempt}/{max_retries}): {e}")
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    print(f"PruneCritic RCA: LLM call failed (attempt {attempt}/{max_retries}): {e}")
                    continue

        error_msg = f"PruneCritic RCA: Maximum retries ({max_retries}) reached, unable to parse LLM response: {last_error}"
        print(f"X {error_msg}")
        raise RuntimeError(error_msg)

    def _find_lines_to_restore(self, log_lines: List[LogLine],
                               delete_ids: Set[int],
                               original_rca: RCAResult,
                               pruned_rca: RCAResult) -> Set[int]:
        restore_ids = set()

        for line in log_lines:
            if line.no not in delete_ids:
                continue

            should_restore = False

            if original_rca.primary_error_message and \
               original_rca.primary_error_message in line.text:
                should_restore = True

            if original_rca.exit_code is not None:
                if f"exit code {original_rca.exit_code}" in line.text.lower() or \
                   f"exitcode={original_rca.exit_code}" in line.text.lower():
                    should_restore = True

            if original_rca.failed_command and \
               original_rca.failed_command in line.text:
                should_restore = True

            if original_rca.exception_type and \
               original_rca.exception_type in line.text:
                should_restore = True

            if should_restore:
                restore_ids.add(line.no)

        return restore_ids

    def _build_pruned_content(self, log_lines: List[LogLine],
                             delete_ids: Set[int]) -> str:
        pruned_lines = []
        for line in log_lines:
            if line.no not in delete_ids:
                pruned_lines.append(f"[{line.no}] {line.text}")

        return "\n".join(pruned_lines)

    def _parse_validation_response(self, response: str) -> Set[int]:
        restore_ids, _ = self._parse_validation_response_with_status(response)
        return restore_ids

    def _parse_validation_response_with_status(self, response: str) -> Tuple[Set[int], bool]:
        restore_ids = set()

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return restore_ids, False

            data = json.loads(json_match.group())

            if "all_evidence_preserved" not in data:
                return restore_ids, False

            if not data.get("all_evidence_preserved", True):
                restore_line_nos = data.get("restore_line_nos", [])
                for line_no in restore_line_nos:
                    if isinstance(line_no, int):
                        restore_ids.add(line_no)

            return restore_ids, True

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON validation response: {e}")
            return restore_ids, False
        except Exception as e:
            print(f"Error parsing validation response: {e}")
            return restore_ids, False

    def _contains_error_keyword(self, text: str) -> bool:
        error_keywords = [
            'error', 'Error', 'ERROR',
            'fail', 'Fail', 'FAIL',
            'exception', 'Exception',
            'traceback', 'Traceback',
            'panic', 'Panic',
            'fatal', 'Fatal',
            'exit code',
        ]
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in error_keywords)


class PruneCriticAgent:
    def __init__(self, instance_id: int = 1, validation_mode: str = "evidence"):
        from core.config_loader import config_loader

        try:
            pipeline_config = config_loader.get_agent_config('pruning_pipeline')
            agent_config = pipeline_config.get('prune_critic', {})
        except Exception:
            extraction_config = config_loader.get_agent_config('extraction_agents')
            agent_config = extraction_config.get('pruning_agent_1', {})

        from core.base_agent import BaseAgent

        class _CriticAgent(BaseAgent):
            def __init__(self, config):
                super().__init__('prune_critic', config)
                self.agent_name = f"prune_critic_{instance_id}"

            def process(self, input_data):
                pass

        self._agent = _CriticAgent(agent_config)
        self._critic = PruneCritic(
            llm_client=self._agent.llm_client,
            validation_mode=validation_mode
        )

    def validate(self, log_lines: List[LogLine],
                delete_ids: Set[int],
                protect: ProtectResult,
                log_metadata: Dict[str, str] = None) -> Set[int]:
        return self._critic.validate(log_lines, delete_ids, protect)
