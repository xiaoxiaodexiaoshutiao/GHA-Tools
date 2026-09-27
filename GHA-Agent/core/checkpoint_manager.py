import os
from typing import Dict, Any, List, Tuple, Optional


class BlockStatistics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_blocks_unfiltered = 0
        self.planned_blocks = 0
        self.visited_blocks = []
        self.visited_count = 0

    def set_total_blocks(self, count: int):
        self.total_blocks_unfiltered = count

    def set_planned_blocks(self, count: int):
        self.planned_blocks = count

    def add_visited_block(self, block_range: Tuple[int, int]):
        self.visited_blocks.append(block_range)
        self.visited_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_timestamp_blocks_unfiltered": self.total_blocks_unfiltered,
            "planned_timestamp_blocks": self.planned_blocks,
            "visited_timestamp_blocks": self.visited_count,
            "visited_blocks_detail": self.visited_blocks
        }


class CheckpointManager:
    def __init__(self, result_base_path: str = None):
        self.result_base_path = result_base_path

    def save_llm_response(self, agent_name: str, response: str,
                         log_metadata: Dict[str, str] = None,
                         call_index: int = 0,
                         extra_info: Dict[str, Any] = None):
        pass

    def check_agent_completed(self, agent_name: str, log_metadata: Dict[str, str]) -> bool:
        return False

    def load_agent_result(self, agent_name: str, log_metadata: Dict[str, str]) -> Optional[Dict[str, Any]]:
        return None

    def save_agent_result(self, agent_name: str, result: Dict[str, Any],
                         log_metadata: Dict[str, str]) -> str:
        return ""


class ResultSaver:
    def __init__(self, result_base_path: str = None):
        self.result_base_path = result_base_path or "results"

    def save_judgment_result(self, blocks_to_extract: List = None,
                            log_metadata: Dict[str, str] = None,
                            status: str = "success") -> str:
        return ""

    def save_extraction_result(self, final_ranges: List = None,
                              log_metadata: Dict[str, str] = None) -> str:
        return ""

    def save_final_result(self, result: Dict[str, Any],
                         log_metadata: Dict[str, str]):
        if not log_metadata:
            return

        save_dir = self.result_base_path
        os.makedirs(save_dir, exist_ok=True)

        import json
        result_file = os.path.join(save_dir, "final_summary.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
