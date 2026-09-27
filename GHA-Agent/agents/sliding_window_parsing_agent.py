from typing import Dict, Any, List, Tuple, Optional
from core.base_agent import BaseAgent
import time


class SlidingWindowParsingAgent(BaseAgent):
    def __init__(
        self,
        instance_id: int = 1,
        window_size: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
        use_react_extraction: bool = True
    ):
        self.instance_id = instance_id
        self.use_react_extraction = use_react_extraction

        from core.config_loader import config_loader

        try:
            sliding_agents_config = config_loader.get_agent_config('sliding_window_agents')
            agent_config_key = f'agent_{instance_id}'

            if agent_config_key in sliding_agents_config:
                agent_config = sliding_agents_config[agent_config_key]
                prompt_key = 'extraction_agent_pruning'
                super().__init__(prompt_key, agent_config)
            else:
                super().__init__('sliding_window_agent')
        except Exception:
            super().__init__('sliding_window_agent')

        self.default_window_size = self.config.get('window_size', 300)
        self.default_overlap_ratio = self.config.get('overlap_ratio', 0.3)
        self.max_windows = self.config.get('max_windows', 10000)

        if window_size is not None:
            self.default_window_size = window_size
        if overlap_ratio is not None:
            self.default_overlap_ratio = overlap_ratio

        self.default_window_stride = int(self.default_window_size * (1 - self.default_overlap_ratio))

        self.agent_name = f"sliding_window_agent_{instance_id}"

        self.extraction_agent = None
        if self.use_react_extraction:
            try:
                from agents.extraction_agent import SlidingWindowExtractionAgent
                self.extraction_agent = SlidingWindowExtractionAgent(instance_id=instance_id)
                print(f"Initialized {self.agent_name} (ReAct mode), default window size: {self.default_window_size} lines, "
                      f"default overlap ratio: {self.default_overlap_ratio*100}%, default stride: {self.default_window_stride} lines")
            except Exception as e:
                print(f"Warning: Unable to initialize ReAct extraction agent, using simple mode: {e}")
                self.use_react_extraction = False
                print(f"Initialized {self.agent_name} (simple mode), default window size: {self.default_window_size} lines, "
                      f"default overlap ratio: {self.default_overlap_ratio*100}%, default stride: {self.default_window_stride} lines")
        else:
            print(f"Initialized {self.agent_name} (simple mode), default window size: {self.default_window_size} lines, "
                  f"default overlap ratio: {self.default_overlap_ratio*100}%, default stride: {self.default_window_stride} lines")

    def _generate_windows(
        self,
        total_lines: int,
        existing_blocks: List[Tuple[int, int]] = None,
        window_size: Optional[int] = None,
        window_stride: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        ws = window_size or self.default_window_size
        stride = window_stride or self.default_window_stride

        windows = []
        existing_blocks = existing_blocks or []

        sorted_blocks = sorted(existing_blocks, key=lambda x: x[0])

        current_start = 1

        while current_start <= total_lines and len(windows) < self.max_windows:
            window_start = current_start
            window_end = min(total_lines, current_start + ws - 1)

            has_overlap = False
            for block_start, block_end in sorted_blocks:
                if not (window_end < block_start or window_start > block_end):
                    has_overlap = True

                    current_start = block_end + 1
                    break

            if not has_overlap:
                windows.append((window_start, window_end))

                current_start += stride

                if window_end == total_lines:
                    break

        overlap_ratio = 1.0 - (stride / ws) if ws > 0 else 0.0
        print(f"Generated {len(windows)} sliding windows ({ws} lines per window, {overlap_ratio*100:.1f}% overlap), "
              f"skipped {len(sorted_blocks)} verified code blocks")
        return windows

    def _analyze_window(self, logs: List[str], start: int, end: int) -> Dict[str, Any]:
        window_content = "\n".join(logs[start-1:end])

        prompt = f"""Please analyze the following GitHub Action log snippet (lines {start}-{end}) to determine if it contains error information causing the run failure.

If errors are found, return the line number range (format: #start_line-end_line#).
If no obvious errors are found, return: NO_ERROR

Log content:
{window_content}

Please only return line number range or NO_ERROR, do not include other explanations.
"""

        messages = [{"role": "user", "content": prompt}]

        try:
            def llm_call():
                return self.call_llm(messages)

            def parse_response(resp):
                return self._parse_response(resp)

            success, blocks, response = self.retry_on_parse_failure(llm_call, parse_response)

            if not success:
                response = ""

            return {
                "status": "success",
                "response": response,
                "window": (start, end)
            }
        except Exception as e:
            print(f"Window [{start}-{end}] analysis failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "window": (start, end)
            }

    def _analyze_window_react(
        self,
        logs: List[str],
        start: int,
        end: int,
        window_index: int,
        total_windows: int,
        window_size: int,
        previous_extractions: List[Dict[str, Any]],
        log_metadata: Dict[str, str] = None
    ) -> Dict[str, Any]:
        if not self.extraction_agent:
            return self._analyze_window(logs, start, end)

        window_content = "\n".join(logs[start-1:end])

        input_data = {
            'log_content': window_content,
            'window_start': start,
            'window_end': end,
            'window_index': window_index,
            'total_windows': total_windows,
            'log_lines': logs,
            'total_lines': len(logs),
            'window_size': window_size,
            'previous_extractions': previous_extractions,
            'log_metadata': log_metadata
        }

        try:
            result = self.extraction_agent.process(input_data)

            if result['status'] == 'success':
                return {
                    "status": "success",
                    "extracted_ranges": result.get('extracted_ranges', []),
                    "window": (start, end),
                    "forward_expansions": result.get('forward_expansions', 0),
                    "backward_expansions": result.get('backward_expansions', 0),
                    "modifications": result.get('modifications', []),
                    "raw_response": result.get('raw_response', '')
                }
            else:
                return {
                    "status": "failed",
                    "error": result.get('error', 'Unknown error'),
                    "window": (start, end)
                }

        except Exception as e:
            print(f"Window [{start}-{end}] ReAct analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e),
                "window": (start, end)
            }

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

    def _parse_response(self, response: str) -> List[Tuple[int, int]]:
        import re

        if "NO_ERROR" in response.upper():
            return []

        pattern = r'#(\d+)(?:-(\d+))?#'
        matches = re.findall(pattern, response)

        blocks = []
        for match in matches:
            start = int(match[0])
            end = int(match[1]) if match[1] else start
            blocks.append((start, end))

        return blocks

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw_logs = input_data['raw_logs']
            existing_blocks = input_data.get('existing_blocks', [])
            line_offset = input_data.get('line_offset', 0)
            use_react = input_data.get('use_react', self.use_react_extraction)
            log_metadata = input_data.get('log_metadata', None)

            window_size = input_data.get('window_size', self.default_window_size)
            overlap_ratio = input_data.get('overlap_ratio', self.default_overlap_ratio)
            window_stride = int(window_size * (1 - overlap_ratio))

            mode_str = "ReAct mode" if use_react and self.extraction_agent else "simple mode"
            print(f"[{self.agent_name}] Starting sliding window parsing ({mode_str}), log lines: {len(raw_logs)}, "
                  f"existing blocks: {len(existing_blocks)}, line offset: {line_offset}")
            print(f"  Window config: size={window_size} lines, overlap={overlap_ratio*100:.1f}%, stride={window_stride} lines")

            windows = self._generate_windows(
                len(raw_logs),
                existing_blocks,
                window_size=window_size,
                window_stride=window_stride
            )

            all_extracted_blocks = []
            previous_extractions = []

            react_stats = {
                "total_forward_expansions": 0,
                "total_backward_expansions": 0,
                "total_modifications": 0,
                "windows_with_errors": 0
            }

            for window_idx, (start, end) in enumerate(windows):
                if use_react and self.extraction_agent:
                    result = self._analyze_window_react(
                        raw_logs, start, end,
                        window_index=window_idx,
                        total_windows=len(windows),
                        window_size=window_size,
                        previous_extractions=previous_extractions,
                        log_metadata=log_metadata
                    )

                    if result['status'] == 'success':
                        blocks = result.get('extracted_ranges', [])

                        react_stats["total_forward_expansions"] += result.get('forward_expansions', 0)
                        react_stats["total_backward_expansions"] += result.get('backward_expansions', 0)
                        react_stats["total_modifications"] += len(result.get('modifications', []))

                        if blocks:
                            react_stats["windows_with_errors"] += 1
                            all_extracted_blocks.extend(blocks)
                            print(f"Window [{start}-{end}] found {len(blocks)} error blocks")

                            modifications = result.get('modifications', [])
                            if modifications:
                                print(f"  Warning: Window modified {len(modifications)} previous extraction results")

                        if blocks:
                            previous_extractions.append({
                                "window_index": window_idx,
                                "window_range": [start, end],
                                "ranges": blocks,
                                "timestamp": time.time()
                            })
                else:
                    result = self._analyze_window(raw_logs, start, end)

                    if result['status'] == 'success':
                        blocks = self._parse_response(result['response'])
                        all_extracted_blocks.extend(blocks)

                        if blocks:
                            print(f"Window [{start}-{end}] found {len(blocks)} error blocks")

            merged_blocks = self._merge_overlapping_ranges(all_extracted_blocks)

            final_blocks = []
            for new_start, new_end in merged_blocks:
                is_covered = False
                for exist_start, exist_end in existing_blocks:
                    if exist_start <= new_start and new_end <= exist_end:
                        is_covered = True
                        break

                if not is_covered:
                    final_blocks.append((new_start, new_end))

            if line_offset > 0:
                absolute_blocks = [(start + line_offset, end + line_offset)
                                  for start, end in final_blocks]
                print(f"Applied offset {line_offset}, converted relative line numbers to absolute line numbers")
            else:
                absolute_blocks = final_blocks

            result = {
                "status": "success",
                "extracted_blocks": absolute_blocks,
                "windows_analyzed": len(windows),
                "block_count": len(absolute_blocks),
                "window_size_used": window_size,
                "overlap_ratio_used": overlap_ratio,
                "react_stats": react_stats if use_react and self.extraction_agent else None
            }

            print(f"[{self.agent_name}] Sliding window parsing complete, analyzed {len(windows)} windows, "
                  f"extracted {len(absolute_blocks)} new code blocks (absolute line numbers)")

            if use_react and self.extraction_agent:
                print(f"  ReAct stats: Forward expansions {react_stats['total_forward_expansions']}, "
                      f"Backward expansions {react_stats['total_backward_expansions']}, "
                      f"History modifications {react_stats['total_modifications']}")

            return result

        except Exception as e:
            print(f"Sliding window parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e),
                "extracted_blocks": []
            }

    def validate_input(self, input_data: Any) -> bool:
        return (isinstance(input_data, dict) and
                "raw_logs" in input_data and
                isinstance(input_data["raw_logs"], list))
