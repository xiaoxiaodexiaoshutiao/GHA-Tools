from typing import Dict, Any, List, Tuple, Optional
from core.base_agent import BaseAgent
import re
import time


class ExtractionAgent(BaseAgent):
    def __init__(self, agent_type: str, instance_id: int = 1):
        from core.config_loader import config_loader
        extraction_agents_config = config_loader.get_agent_config('extraction_agents')

        if agent_type == 'pruning':
            agent_config_key = f'pruning_agent_{instance_id}'
            prompt_key = 'extraction_agent_pruning'
        elif agent_type == 'verification':
            agent_config_key = 'verification_agent'
            prompt_key = 'extraction_agent_verification'
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        if agent_config_key not in extraction_agents_config:
            raise ValueError(f"Extraction agent config not found: {agent_config_key}")

        agent_config = extraction_agents_config[agent_config_key]

        super().__init__(prompt_key, agent_config)
        self.agent_type = agent_type
        self.instance_id = instance_id
        self.agent_name = f"extraction_agent_{agent_type}_{instance_id}"


class PruningAgent(ExtractionAgent):
    def __init__(self, instance_id: int = 1):
        super().__init__('pruning', instance_id)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            log_content = input_data['log_content']
            block_start = input_data['block_start']
            block_end = input_data['block_end']
            log_metadata = input_data.get('log_metadata', None)
            block_index = input_data.get('block_index', 0)

            self.log(f"Identifying irrelevant log lines [{block_start}-{block_end}]")

            user_prompt = self.format_prompt(
                'user_prompt_template',
                log_content=log_content,
                block_start=block_start,
                block_end=block_end
            )

            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]

            def llm_call():
                return self.call_llm(
                    messages,
                    log_metadata=log_metadata,
                    call_index=block_index,
                    extra_info={"block_range": [block_start, block_end]}
                )

            def parse_response(resp):
                ranges = self._parse_response(resp, block_start, block_end)

                return ranges

            success, irrelevant_ranges, response = self.retry_on_parse_failure(llm_call, parse_response)

            if not success:
                irrelevant_ranges = []
                response = ""

            self.log(f"Identified {len(irrelevant_ranges)} irrelevant ranges", "success")

            return {
                "status": "success",
                "irrelevant_ranges": irrelevant_ranges,
                "raw_response": response
            }

        except Exception as e:
            self.log(f"Failed to identify irrelevant logs: {e}", "error")
            return {
                "status": "failed",
                "error": str(e),
                "irrelevant_ranges": []
            }

    def _parse_response(self, response: str, block_start: int, block_end: int) -> List[Tuple[int, int]]:
        response = response.strip()

        if "#ALL#" in response.upper():
            return [(block_start, block_end)]

        if "#NONE#" in response.upper():
            return []

        pattern = r'#(\d+)-(\d+)#'
        matches = re.findall(pattern, response)

        ranges = []
        for match in matches:
            start = int(match[0])
            end = int(match[1])

            if start >= block_start and end <= block_end and start <= end:
                ranges.append((start, end))
            else:
                self.log(f"Invalid range: {start}-{end}, block range: {block_start}-{block_end}", "warning")

        return ranges


class SlidingWindowExtractionAgent(BaseAgent):
    def __init__(self, instance_id: int = 1):
        from core.config_loader import config_loader

        try:
            sw_extraction_config = config_loader.get_agent_config('sliding_window_extraction_agents')
            agent_config_key = f'agent_{instance_id}'

            if agent_config_key in sw_extraction_config:
                agent_config = sw_extraction_config[agent_config_key]
            else:
                agent_config = sw_extraction_config.get('default_agent', {})
                if not agent_config:
                    extraction_config = config_loader.get_agent_config('extraction_agents')
                    agent_config = extraction_config.get('pruning_agent_1', {})

            self.max_forward_expansion = sw_extraction_config.get('max_forward_expansion', 1)
            self.max_backward_expansion = sw_extraction_config.get('max_backward_expansion', 3)
            self.max_view_previous = sw_extraction_config.get('max_view_previous', 10)

        except Exception:
            extraction_config = config_loader.get_agent_config('extraction_agents')
            agent_config = extraction_config.get('pruning_agent_1', {})
            self.max_forward_expansion = 1
            self.max_backward_expansion = 3
            self.max_view_previous = 10

        super().__init__('extraction_agent_sliding_window', agent_config)
        self.instance_id = instance_id
        self.agent_name = f"sliding_window_extraction_agent_{instance_id}"

    def _check_can_expand_forward(
        self,
        previous_extractions: List[Dict[str, Any]],
        window_start: int
    ) -> bool:
        if not previous_extractions:
            return True

        last_extraction = previous_extractions[-1]
        for range_item in last_extraction.get('ranges', []):
            start, end = range_item[0], range_item[1]
            if start <= window_start <= end:
                return False

        return True

    def _merge_overlapping_ranges(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not ranges:
            return []

        sorted_ranges = sorted(ranges, key=lambda x: x[0])

        merged = [sorted_ranges[0]]

        for current in sorted_ranges[1:]:
            last = merged[-1]

            if current[0] <= last[1] + 1:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)

        return merged

    def _remove_duplicate_ranges(
        self,
        new_ranges: List[Tuple[int, int]],
        previous_ranges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        if not new_ranges or not previous_ranges:
            return new_ranges

        result = []
        for new_start, new_end in new_ranges:
            is_duplicate = False
            for prev_start, prev_end in previous_ranges:
                if prev_start <= new_start and new_end <= prev_end:
                    is_duplicate = True
                    break

                elif prev_start <= new_start <= prev_end < new_end:
                    new_start = prev_end + 1
                elif new_start < prev_start <= new_end <= prev_end:
                    new_end = prev_start - 1

            if not is_duplicate and new_start <= new_end:
                result.append((new_start, new_end))

        return result

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            log_content = input_data['log_content']
            window_start = input_data['window_start']
            window_end = input_data['window_end']
            window_index = input_data.get('window_index', 0)
            total_windows = input_data.get('total_windows', 1)
            log_lines = input_data.get('log_lines', [])
            total_lines = input_data.get('total_lines', len(log_lines))
            window_size = input_data.get('window_size', 200)
            previous_extractions = input_data.get('previous_extractions', [])
            log_metadata = input_data.get('log_metadata', None)

            self.log(f"Sliding window extraction [{window_start}-{window_end}] (window {window_index + 1}/{total_windows})")

            can_expand_forward = self._check_can_expand_forward(previous_extractions, window_start)
            has_previous_extraction = len(previous_extractions) > 0

            if not can_expand_forward:
                self.log("Forward expansion blocked: Previous window extraction result covers current window start line", "info")

            user_prompt = self.format_prompt(
                'user_prompt_template',
                log_content=log_content,
                window_start=window_start,
                window_end=window_end,
                window_index=window_index + 1,
                total_windows=total_windows,
                has_previous_extraction="Yes" if has_previous_extraction else "No",
                can_expand_forward="Yes" if can_expand_forward else "No (Previous window extraction result covers current start line)",
                max_forward_expansion=self.max_forward_expansion,
                max_backward_expansion=self.max_backward_expansion,
                forward_expansions=0,
                backward_expansions=0
            )

            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]

            from core.function_executor import create_sliding_window_extraction_executor
            from core.function_schemas import get_sliding_window_extraction_tool_collection

            executor = create_sliding_window_extraction_executor()

            executor.set_context({
                'window_start': window_start,
                'window_end': window_end,
                'window_index': window_index,
                'total_lines': total_lines,
                'log_lines': log_lines,
                'window_size': window_size,
                'forward_expansions': 0,
                'backward_expansions': 0,
                'max_forward_expansion': self.max_forward_expansion,
                'max_backward_expansion': self.max_backward_expansion,
                'can_expand_forward': can_expand_forward,
                'previous_extractions': previous_extractions,
                'current_window_index': window_index,
                'has_modifications': False,
                'modification_log': []
            })

            tool_collection = get_sliding_window_extraction_tool_collection()

            def llm_call():
                return self.call_llm_with_tools(
                    messages=messages,
                    tools=tool_collection,
                    function_executor=executor,
                    log_metadata=log_metadata,
                    call_index=window_index,
                    extra_info={"window_range": [window_start, window_end]},
                    max_iterations=8
                )

            def parse_response(resp):
                return self._parse_response(resp, window_start, window_end)

            success, extracted_ranges, response = self.retry_on_parse_failure(llm_call, parse_response)

            if not success:
                extracted_ranges = []
                response = ""

            forward_expansions = executor.context.get('forward_expansions', 0)
            backward_expansions = executor.context.get('backward_expansions', 0)
            modifications = executor.context.get('modification_log', [])

            all_previous_ranges = []
            for extraction in previous_extractions:
                all_previous_ranges.extend([(r[0], r[1]) for r in extraction.get('ranges', [])])

            if all_previous_ranges:
                original_count = len(extracted_ranges)
                extracted_ranges = self._remove_duplicate_ranges(extracted_ranges, all_previous_ranges)
                if len(extracted_ranges) < original_count:
                    self.log(f"Deduplication: Removed {original_count - len(extracted_ranges)} duplicate ranges from previous results", "info")

            extracted_ranges = self._merge_overlapping_ranges(extracted_ranges)

            self.log(f"Extracted {len(extracted_ranges)} ranges", "success")
            if forward_expansions > 0 or backward_expansions > 0:
                self.log(f"Expansion info: Forward {forward_expansions} times, Backward {backward_expansions} times")

            return {
                "status": "success",
                "extracted_ranges": extracted_ranges,
                "raw_response": response,
                "forward_expansions": forward_expansions,
                "backward_expansions": backward_expansions,
                "modifications": modifications
            }

        except Exception as e:
            self.log(f"Sliding window extraction failed: {e}", "error")
            import traceback
            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e),
                "extracted_ranges": [],
                "forward_expansions": 0,
                "backward_expansions": 0,
                "modifications": []
            }

    def _parse_response(self, response: str, window_start: int, window_end: int) -> List[Tuple[int, int]]:
        response = response.strip()

        if "NO_ERROR" in response.upper():
            return []

        if "#ALL#" in response.upper():
            return [(window_start, window_end)]

        pattern = r'#(\d+)-(\d+)#'
        matches = re.findall(pattern, response)

        ranges = []
        for match in matches:
            start = int(match[0])
            end = int(match[1])

            if start <= end:
                ranges.append((start, end))
            else:
                self.log(f"Invalid range: {start}-{end} (start line greater than end line)", "warning")

        return ranges
