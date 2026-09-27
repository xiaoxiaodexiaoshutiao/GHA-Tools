from typing import List, Set, Dict, Tuple
from .types import LogLine, ProtectResult, PruningResult


class Emitter:
    def merge(self,
             predelete_ids: Set[int],
             llm_delete_ids: Set[int],
             restore_ids: Set[int],
             hard_protect_set: Set[int],
             all_line_nos: Set[int] = None) -> PruningResult:
        delete_ids = predelete_ids | llm_delete_ids

        delete_ids = delete_ids - restore_ids

        delete_ids = delete_ids - hard_protect_set

        if all_line_nos:
            kept_lines = sorted(all_line_nos - delete_ids)
        else:
            kept_lines = []

        statistics = {
            "predelete_count": len(predelete_ids),
            "llm_delete_count": len(llm_delete_ids),
            "restore_count": len(restore_ids),
            "hard_protect_count": len(hard_protect_set),
            "final_delete_count": len(delete_ids),
            "final_kept_count": len(kept_lines) if kept_lines else None,
        }

        return PruningResult(
            delete_lines=sorted(delete_ids),
            kept_lines=kept_lines,
            statistics=statistics
        )

    def to_ranges(self, delete_lines: List[int]) -> List[Tuple[int, int]]:
        if not delete_lines:
            return []

        ranges = []
        start = delete_lines[0]
        end = delete_lines[0]

        for line_no in delete_lines[1:]:
            if line_no == end + 1:
                end = line_no
            else:
                ranges.append((start, end))
                start = line_no
                end = line_no

        ranges.append((start, end))

        return ranges

    def calculate_kept_ranges(self,
                             block_start: int,
                             block_end: int,
                             delete_lines: List[int]) -> List[Tuple[int, int]]:
        if not delete_lines:
            return [(block_start, block_end)]

        delete_set = set(delete_lines)
        kept_ranges = []
        current_start = None

        for line_no in range(block_start, block_end + 1):
            if line_no not in delete_set:
                if current_start is None:
                    current_start = line_no
            else:
                if current_start is not None:
                    kept_ranges.append((current_start, line_no - 1))
                    current_start = None

        if current_start is not None:
            kept_ranges.append((current_start, block_end))

        return kept_ranges

    def format_output(self, result: PruningResult,
                     include_text: bool = False,
                     line_dict: Dict[int, LogLine] = None) -> Dict:
        output = {
            "delete_lines": result.delete_lines,
            "delete_count": result.delete_count,
            "kept_count": result.kept_count,
            "statistics": result.statistics,
        }

        if include_text and line_dict:
            output["delete_details"] = [
                {
                    "line_no": line_no,
                    "text": line_dict[line_no].text if line_no in line_dict else ""
                }
                for line_no in result.delete_lines
            ]

        return output

    def format_as_irrelevant_ranges(self,
                                   delete_lines: List[int]) -> List[Tuple[int, int]]:
        return self.to_ranges(delete_lines)


def subtract_ranges(block_range: Tuple[int, int],
                   irrelevant_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not irrelevant_ranges:
        return [block_range]

    sorted_irrelevant = sorted(irrelevant_ranges, key=lambda x: x[0])
    merged_irrelevant = []

    for curr_start, curr_end in sorted_irrelevant:
        if merged_irrelevant and curr_start <= merged_irrelevant[-1][1] + 1:
            merged_irrelevant[-1] = (merged_irrelevant[-1][0], max(merged_irrelevant[-1][1], curr_end))
        else:
            merged_irrelevant.append((curr_start, curr_end))

    block_start, block_end = block_range
    kept_ranges = []
    current_pos = block_start

    for irrel_start, irrel_end in merged_irrelevant:
        if irrel_start > current_pos:
            kept_ranges.append((current_pos, min(irrel_start - 1, block_end)))

        current_pos = max(current_pos, irrel_end + 1)

        if current_pos > block_end:
            break

    if current_pos <= block_end:
        kept_ranges.append((current_pos, block_end))

    return kept_ranges
