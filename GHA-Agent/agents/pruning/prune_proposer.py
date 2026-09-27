import json
import re
from typing import List, Set, Dict, Any, Optional, Tuple
from .types import LogLine, Anchor, AnchorResult, ProtectResult, DeleteProposal


class PruneProposer:
    def __init__(self, llm_client=None, max_candidate_lines: int = 300):
        self.llm_client = llm_client
        self.max_candidate_lines = max_candidate_lines

    def propose(self, log_lines: List[LogLine],
               anchors: AnchorResult,
               protect: ProtectResult,
               max_retries: int = 3) -> Set[int]:
        if not self.llm_client:
            return set()

        line_dict = {line.no: line for line in log_lines}

        prompt = self._build_prompt(log_lines, anchors, protect, line_dict)

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm_client.call(messages)

                delete_ids, parse_success = self._parse_response_with_status(response, protect)

                if parse_success:
                    if attempt > 1:
                        print(f"PruneProposer: Successfully parsed on attempt {attempt}")
                    return delete_ids
                else:
                    last_error = "Unable to extract valid JSON from response"
                    if attempt < max_retries:
                        print(f"PruneProposer: Failed to parse response (attempt {attempt}/{max_retries}): {last_error}")
                        print(f"  Response preview: {response[:200]}...")
                        continue

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    print(f"PruneProposer: LLM call or parsing failed (attempt {attempt}/{max_retries}): {e}")
                    continue

        error_msg = f"PruneProposer: Maximum retries ({max_retries}) reached, unable to parse LLM response: {last_error}"
        print(f"X {error_msg}")
        raise RuntimeError(error_msg)

    def _get_system_prompt(self) -> str:
        return """You are a log pruning agent. Your task: Output only the line numbers of log lines to delete, without affecting root cause identification.

## Hard Rules (Must Follow)
1. Hard-Protect lines **must NEVER be deleted**
2. When uncertain, do NOT delete
3. Output must be valid JSON, containing only a delete array (list of line numbers)
4. Do NOT output any explanations, summaries, or retained lines

## Types of Lines That Can Be Deleted
- Successful download/installation pipelines (far from error anchors)
- Environment variable enumerations, PATH displays, runner info unrelated to failure
- Lines in large successful compilation output that are "pure info/no warning/error"
- Repeated INFO logs (keep first and last)

## Never Delete
- Lines in the Hard-Protect set
- Lines containing signals like error|failed|exception|traceback|panic|exit code|caused by|assert
- Command lines like ##[command] / Run ...
- Any lines you're uncertain about

## Output Format (Strict JSON, only line numbers)
```json
{"delete": [128, 129, 130, 145, 146]}
```

If there are no lines to delete, output:
```json
{"delete": []}
```"""

    def _build_prompt(self, log_lines: List[LogLine],
                     anchors: AnchorResult,
                     protect: ProtectResult,
                     line_dict: Dict[int, LogLine]) -> str:
        parts = []

        hard_protect_list = sorted(protect.hard_protect_set)
        parts.append(f"Hard-Protect line_nos (must never delete): {hard_protect_list}")

        soft_protect_list = sorted(protect.soft_protect_set)
        parts.append(f"Soft-Protect line_nos (delete with caution): {soft_protect_list}")

        parts.append("\nFailure anchors (for reference):")
        for anchor in anchors.anchors_hard[:10]:
            parts.append(f"[no={anchor.line_no}] {anchor.text[:200]}")

        parts.append("\nCandidate lines (you may choose deletions ONLY from these):")

        candidate_lines = []
        for line in log_lines:
            if line.no in protect.candidate_set:
                tag = "soft" if line.no in protect.soft_protect_set else "candidate"
                candidate_lines.append((line.no, tag, line.text))

        if len(candidate_lines) > self.max_candidate_lines:
            half = self.max_candidate_lines // 2
            candidate_lines = candidate_lines[:half] + candidate_lines[-half:]
            parts.append(f"(Showing {self.max_candidate_lines} of {len(candidate_lines)} candidates)")

        for line_no, tag, text in candidate_lines:
            display_text = text[:300] + "..." if len(text) > 300 else text
            parts.append(f"[no={line_no}][tag={tag}] {display_text}")

        parts.append("""
Task:
Return JSON with line numbers only (no text field):
{"delete": [128, 129, 130]}
Only include lines that are safe to delete and do not remove any evidence needed to explain why the run failed.
If unsure, output an empty delete list: {"delete": []}
""")

        return "\n".join(parts)

    def _parse_response(self, response: str, protect: ProtectResult) -> Set[int]:
        delete_ids, _ = self._parse_response_with_status(response, protect)
        return delete_ids

    def _parse_response_with_status(self, response: str, protect: ProtectResult) -> Tuple[Set[int], bool]:
        delete_ids = set()

        json_str = None

        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if code_block_match:
            json_str = code_block_match.group(1)

        if json_str is None:
            simple_match = re.search(r'\{\s*"delete"\s*:\s*\[[\d\s,]*\]\s*\}', response)
            if simple_match:
                json_str = simple_match.group()

        if json_str is None:
            obj_match = re.search(r'\{\s*"delete"\s*:\s*\[.*?\]\s*\}', response, re.DOTALL)
            if obj_match:
                json_str = obj_match.group()

        if json_str is None:
            greedy_match = re.search(r'\{[\s\S]*"delete"[\s\S]*\}', response)
            if greedy_match:
                json_str = greedy_match.group()

        if json_str is None:
            return delete_ids, False

        try:
            data = json.loads(json_str)

            if "delete" in data and isinstance(data["delete"], list):
                for item in data["delete"]:
                    line_no = None
                    if isinstance(item, dict) and "line_no" in item:
                        line_no = int(item["line_no"])
                    elif isinstance(item, (int, float)):
                        line_no = int(item)

                    if line_no is not None:
                        if line_no not in protect.hard_protect_set:
                            delete_ids.add(line_no)

                return delete_ids, True
            else:
                return delete_ids, False

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return delete_ids, False
        except Exception as e:
            print(f"Error parsing response: {e}")
            return delete_ids, False

    def create_proposals(self, delete_ids: Set[int],
                        line_dict: Dict[int, LogLine]) -> List[DeleteProposal]:
        proposals = []
        for line_no in sorted(delete_ids):
            line = line_dict.get(line_no)
            if line:
                proposals.append(DeleteProposal(
                    line_no=line_no,
                    text=line.text,
                    reason="llm_proposed",
                    source="llm"
                ))
        return proposals


class PruneProposerAgent:
    def __init__(self, instance_id: int = 1):
        from core.config_loader import config_loader

        try:
            pipeline_config = config_loader.get_agent_config('pruning_pipeline')
            agent_config = pipeline_config.get('prune_proposer', {})
        except Exception:
            extraction_config = config_loader.get_agent_config('extraction_agents')
            agent_config = extraction_config.get('pruning_agent_1', {})

        from core.base_agent import BaseAgent

        class _ProposerAgent(BaseAgent):
            def __init__(self, config):
                super().__init__('prune_proposer', config)
                self.agent_name = f"prune_proposer_{instance_id}"

            def process(self, input_data):
                pass

        self._agent = _ProposerAgent(agent_config)
        self._proposer = PruneProposer(
            llm_client=self._agent.llm_client
        )

    def propose(self, log_lines: List[LogLine],
               anchors: AnchorResult,
               protect: ProtectResult,
               log_metadata: Dict[str, str] = None) -> Set[int]:
        return self._proposer.propose(log_lines, anchors, protect)


class StrictPruneProposer(PruneProposer):
    def _get_system_prompt(self) -> str:
        return """You are a log pruning agent (strict mode). Your task: Delete ONLY **completely irrelevant** log lines, using an extremely conservative strategy.

## Core Principle (Most Important)
**Better to keep 100 irrelevant log lines than to mistakenly delete 1 potentially relevant line**

## Hard Rules (Must Follow)
1. Hard-Protect lines **must NEVER be deleted**
2. If there's ANY uncertainty, do **NOT delete**
3. Output must be valid JSON, containing only a delete array (list of line numbers)
4. Do NOT output any explanations, summaries, or retained lines

## Lines That Can Be Deleted (Must satisfy ALL of the following conditions)
1. **Pure numeric progress bars**: e.g., "32%", "64%", "100%", containing only percentage numbers
2. **Consecutive repeated decorative characters**: e.g., "====", "----", "****" pure separator lines
3. **Completely blank lines**: containing only spaces or tabs
4. **More than 50 lines away from error anchors**
5. **You are 100% certain they are completely unrelated to any error information**

## Never Delete (Strictly Follow)
- All lines in the Hard-Protect set
- Lines containing any English words (even "success", "ok", "done")
- Lines containing error|failed|exception|traceback|panic|exit code|caused by|assert|warning
- Command lines like ##[command] / Run ...
- Lines containing file paths (e.g., /path/to/file or C:\\path)
- Lines containing timestamps (e.g., 2024-01-01, 12:34:56)
- Lines containing version numbers (e.g., v1.2.3, 1.0.0)
- Lines containing any number+letter combinations
- All lines within 50 lines before/after error anchors
- Any lines you are even 1% uncertain about

## Output Format (Strict JSON, only line numbers)
```json
{"delete": [128, 129, 130]}
```

If there are no lines to delete (this is normal), output:
```json
{"delete": []}
```

## Important Reminder
In strict mode, outputting an empty delete list is **normal and recommended** behavior.
Do not delete any potentially relevant lines just to "do something"."""
