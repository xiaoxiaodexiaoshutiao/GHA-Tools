import re
import json
import os
import time
from typing import List, Dict, Any, Set, Tuple
from core.base_agent import BaseAgent
from core.config_loader import config_loader
from utils.timestamp_segmenter import TimestampSegmenter
from pathlib import Path


class PreprocessingAgent(BaseAgent):
    def __init__(self):
        super().__init__('preprocessing_agent')
        self.keywords = self._load_keywords()
        self.timestamp_segmenter = TimestampSegmenter()

        print(f"Initialized {self.agent_name}, rule-based processing, no LLM needed")

    def _load_keywords(self) -> Dict[str, List[str]]:
        try:
            keywords_path = self.config.get('keywords_path', 'resources/keywords.json')
            if not os.path.isabs(keywords_path):
                base_dir = Path(__file__).parent.parent
                keywords_path = base_dir / keywords_path

            with open(keywords_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    "critical_keywords": data["critical_keywords"],
                    "warning_keywords": data["warning_keywords"]
                }
        except Exception as e:
            print(f"Failed to load keywords file: {e}")
            return {"critical_keywords": [], "warning_keywords": []}

    def _match_keywords(self, logs: List[str], keywords: List[str]) -> Set[int]:
        matched_lines = set()

        for line_num, log_line in enumerate(logs, 1):
            log_lower = log_line.lower()

            for keyword in keywords:
                keyword_lower = keyword.lower()
                pattern = r'\b' + re.escape(keyword_lower) + r"(?:'s)?\b"

                if re.search(pattern, log_lower):
                    matched_lines.add(line_num)

        return matched_lines

    def _match_ansi_red_codes(self, logs: List[str]) -> Set[int]:
        matched_lines = set()

        red_patterns = [
            r'\033\[31m',
            r'\033\[91m',
            r'\033\[1;31m',
            r'\033\[0;31m',
            r'\033\[41m',
            r'\033\[101m',
            r'\033\[31;1m',
            r'\033\[1;91m',
            r'\x1b\[31m',
            r'\x1b\[91m',
            r'\x1b\[1;31m',
        ]

        combined_pattern = '|'.join(red_patterns)

        for line_num, log_line in enumerate(logs, 1):
            if re.search(combined_pattern, log_line):
                matched_lines.add(line_num)

        return matched_lines

    def _group_consecutive_lines(self, line_numbers: List[int]) -> List[List[int]]:
        if not line_numbers:
            return []

        groups = []
        current_group = [line_numbers[0]]

        for i in range(1, len(line_numbers)):
            if line_numbers[i] == line_numbers[i-1] + 1:
                current_group.append(line_numbers[i])
            else:
                groups.append(current_group)
                current_group = [line_numbers[i]]

        groups.append(current_group)
        return groups

    def _extract_error_lines(self, logs: List[str]) -> List[int]:
        start_time = time.time()

        print("Matching using critical_keywords...")
        critical_matched = self._match_keywords(logs, self.keywords["critical_keywords"])
        print(f"critical_keywords matched {len(critical_matched)} lines")

        print("Matching red ANSI codes...")
        ansi_matched = self._match_ansi_red_codes(logs)
        print(f"Red ANSI codes matched {len(ansi_matched)} lines")

        all_matched = critical_matched.union(ansi_matched)
        print(f"Merged total: {len(all_matched)} lines")

        if len(all_matched) < 2:
            print("Too few matches, using warning_keywords...")
            warning_matched = self._match_keywords(logs, self.keywords["warning_keywords"])
            print(f"warning_keywords matched {len(warning_matched)} lines")
            all_matched = all_matched.union(warning_matched)
            print(f"Final total: {len(all_matched)} lines")

        sorted_lines = sorted(list(all_matched))

        processing_time = time.time() - start_time
        print(f"Error line extraction completed in {processing_time:.2f} seconds")

        return sorted_lines

    def _filter_timestamp_blocks(
        self,
        timestamp_blocks: List[Tuple[int, int]],
        error_line_blocks: List[List[int]]
    ) -> List[Tuple[int, int]]:
        error_lines = set()
        for block in error_line_blocks:
            error_lines.update(block)

        filtered = []
        for block_start, block_end in timestamp_blocks:
            for line_num in range(block_start, block_end + 1):
                if line_num in error_lines:
                    filtered.append((block_start, block_end))
                    break

        return filtered

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            start_time = time.time()
            logs = input_data['logs']
            line_offset = input_data.get('line_offset', 0)

            print(f"Starting log preprocessing, total {len(logs)} lines")

            print("\n=== Timestamp Segmentation ===")
            timestamp_success = True
            try:
                timestamps, segmentation_points, segments, statistics = \
                    self.timestamp_segmenter.segment_logs(logs, line_offset=line_offset)

                all_blocks = [
                    (start + line_offset, end + line_offset)
                    for start, end in segments
                ]

                print(f"Timestamp segmentation completed, {len(all_blocks)} blocks")

            except Exception as e:
                print(f"Timestamp segmentation failed: {e}")
                all_blocks = [(1 + line_offset, len(logs) + line_offset)]
                timestamp_success = False

            print("\n=== Keyword Filtering ===")
            error_line_numbers = self._extract_error_lines(logs)
            error_line_blocks = self._group_consecutive_lines(error_line_numbers)

            absolute_error_line_blocks = [
                [line + line_offset for line in block]
                for block in error_line_blocks
            ]

            print("\n=== Filter Timestamp Blocks ===")
            filtered_blocks = self._filter_timestamp_blocks(all_blocks, absolute_error_line_blocks)

            print(f"Filtered: {len(filtered_blocks)} blocks remaining (total {len(all_blocks)})")

            processing_time = time.time() - start_time

            segmenter_params = {
                "k_value": self.timestamp_segmenter.k_value,
                "window_size": self.timestamp_segmenter.window_size
            }

            result = {
                "status": "success",
                "timestamp_blocks": {
                    "total_count": len(all_blocks),
                    "filtered_count": len(filtered_blocks),
                    "segmenter_params": segmenter_params,
                    "timestamp_success": timestamp_success
                },
                "filtered_blocks": filtered_blocks,
                "all_blocks": all_blocks,
                "processing_time": processing_time
            }

            print(f"\nPreprocessing completed, timestamp blocks: {len(all_blocks)} total / {len(filtered_blocks)} filtered")
            print(f"Segmenter params: k_value={segmenter_params['k_value']}, window_size={segmenter_params['window_size']}")
            print(f"Processing time: {processing_time:.2f} seconds")

            return result

        except Exception as e:
            print(f"Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e)
            }

    def validate_input(self, input_data: Any) -> bool:
        return (isinstance(input_data, dict) and
                "logs" in input_data and
                isinstance(input_data["logs"], list) and
                len(input_data["logs"]) > 0)
