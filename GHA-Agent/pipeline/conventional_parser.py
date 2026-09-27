from typing import Dict, Any, List, Tuple, Optional, Set
from agents.judgment_agent import JudgmentAgent
from agents.extraction_agent import PruningAgent
from agents.early_termination_agent import EarlyTerminationJudgmentAgent
from agents.sliding_window_parsing_agent import SlidingWindowParsingAgent
from core.checkpoint_manager import CheckpointManager, BlockStatistics, ResultSaver


class ConventionalParser:
    CONSECUTIVE_NO_ERROR_THRESHOLD = 10
    HARD_CONSECUTIVE_NO_ERROR_CUTOFF = 20

    @staticmethod
    def _subtract_ranges(block_range: Tuple[int, int],
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

    def __init__(self, result_base_path: str = None):
        self.judgment_agent = JudgmentAgent(instance_id=1)
        self.pruning_agent = PruningAgent(instance_id=1)
        self.early_termination_agent = EarlyTerminationJudgmentAgent(instance_id=1)

        self.checkpoint_manager = CheckpointManager(result_base_path)
        self.result_saver = ResultSaver(result_base_path)

        self.block_stats = BlockStatistics()

        self.consecutive_no_error_count = 0
        self.recent_no_error_blocks = []
        self.early_termination_triggered = False
        self.early_termination_result = None
        self.error_blocks_found_count = 0

        try:
            from core.config_loader import config_loader
            early_termination_config = config_loader.get_agent_config('early_termination_agents')
            self.CONSECUTIVE_NO_ERROR_THRESHOLD = early_termination_config.get(
                'consecutive_no_error_threshold',
                self.CONSECUTIVE_NO_ERROR_THRESHOLD
            )
            self.HARD_CONSECUTIVE_NO_ERROR_CUTOFF = early_termination_config.get(
                'hard_consecutive_no_error_cutoff',
                self.HARD_CONSECUTIVE_NO_ERROR_CUTOFF
            )
        except Exception:
            pass

        print("Conventional parser initialized (Single Agent Mode)")
        print(
            "Early termination strategy: Start checking after 5 consecutive no-error blocks, "
            "then invoke at 6th, 8th, 10th... (every 2 blocks)"
        )
        if self.HARD_CONSECUTIVE_NO_ERROR_CUTOFF:
            print(
                "Hard early termination cutoff: "
                f"{self.HARD_CONSECUTIVE_NO_ERROR_CUTOFF} consecutive no-error blocks after first error"
            )

    def parse(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            log_lines = input_data.get('log_lines')
            preprocessing_result = input_data.get('preprocessing_result')
            log_metadata = input_data.get('log_metadata', {})
            phase = input_data.get('phase', None)
            blocks_to_extract_input = input_data.get('blocks_to_extract', None)

            if not log_lines:
                raise ValueError("Missing required parameter: log_lines")
            if not preprocessing_result:
                raise ValueError("Missing required parameter: preprocessing_result")

            timestamp_blocks_info = preprocessing_result['timestamp_blocks']
            total_timestamp_blocks = timestamp_blocks_info['total_count']
            filtered_blocks = preprocessing_result['filtered_blocks']
            all_blocks = preprocessing_result['all_blocks']

            print(f"Starting conventional parsing")
            print(f"Timestamp blocks: total {total_timestamp_blocks}, filtered {len(filtered_blocks)}")

            self.block_stats.reset()
            self.block_stats.set_total_blocks(total_timestamp_blocks)

            self.consecutive_no_error_count = 0
            self.recent_no_error_blocks = []
            self.early_termination_triggered = False
            self.early_termination_result = None
            self.error_blocks_found_count = 0

            self.block_stats.set_planned_blocks(len(filtered_blocks))

            blocks_to_extract = []
            skip_judgment = False

            if phase == 'pruning_only':
                if blocks_to_extract_input is None:
                    raise ValueError("pruning_only phase requires blocks_to_extract parameter")
                blocks_to_extract = blocks_to_extract_input
                skip_judgment = True
                print("Skipping judgment phase (pruning_only mode)")

            elif log_metadata and self.checkpoint_manager.check_agent_completed('judgment_agents', log_metadata):
                saved_judgment = self.checkpoint_manager.load_agent_result('judgment_agents', log_metadata)
                if saved_judgment:
                    print("Detected judgment_agents.json exists, skipping judgment phase")
                    blocks_to_extract_raw = saved_judgment.get('blocks_to_extract', [])

                    blocks_to_extract = [
                        {'range': tuple(block), 'expanded_blocks_up': 0, 'expanded_blocks_down': 0}
                        for block in blocks_to_extract_raw
                    ]
                    skip_judgment = True

            if not skip_judgment:
                print("\n=== Step 1: Judgment Phase (Reverse traversal: back to front) ===")
                verified_blocks: Set[Tuple[int, int]] = set()
                error_blocks: Set[Tuple[int, int]] = set()

                judgment_stats = {
                    "blocks_processed": 0,
                    "blocks_with_error": 0,
                    "blocks_no_error": 0
                }

                for reverse_idx, (block_start, block_end) in enumerate(reversed(filtered_blocks)):
                    block_idx = len(filtered_blocks) - 1 - reverse_idx

                    print(f"\nProcessing block {reverse_idx + 1}/{len(filtered_blocks)} (original index {block_idx}): [{block_start}-{block_end}]")

                    self.block_stats.add_visited_block((block_start, block_end))
                    judgment_stats["blocks_processed"] += 1

                    if self.early_termination_triggered:
                        print("Early termination triggered, skipping remaining blocks")
                        break

                    block_content = "\n".join(log_lines[block_start-1:block_end])

                    current_block_index = self._find_block_index_in_timestamp_blocks(
                        (block_start, block_end),
                        all_blocks
                    )

                    judgment_input = {
                        'log_content': block_content,
                        'block_start': block_start,
                        'block_end': block_end,
                        'current_block_index': current_block_index,
                        'all_timestamp_blocks': all_blocks,
                        'log_lines': log_lines,
                        'verified_blocks': verified_blocks,
                        'error_blocks': error_blocks,
                        'log_metadata': log_metadata,
                        'block_index': reverse_idx
                    }

                    result = self.judgment_agent.process(judgment_input)

                    contains_error = result.get('contains_error', False)
                    final_range = result.get('final_block_range', (block_start, block_end))

                    if contains_error:
                        print("Agent determined: contains error")
                        print(f"   Range: [{final_range[0]}-{final_range[1]}]")

                        blocks_to_extract.append({
                            'range': final_range,
                            'expanded_blocks_up': result.get('expanded_blocks_up', 0),
                            'expanded_blocks_down': result.get('expanded_blocks_down', 0)
                        })
                        judgment_stats["blocks_with_error"] += 1
                        error_blocks.add((block_start, block_end))
                        self.error_blocks_found_count += 1

                        self.consecutive_no_error_count = 0
                        self.recent_no_error_blocks = []
                    else:
                        print("Agent determined: no error, skipping")
                        judgment_stats["blocks_no_error"] += 1

                        self._check_early_termination(
                            block_start, block_end,
                            filtered_blocks, log_lines,
                            reverse_idx
                        )

                    verified_blocks.add((block_start, block_end))

                    if self.early_termination_triggered:
                        print("Early termination triggered, breaking loop")
                        break

                if log_metadata:
                    self.result_saver.save_judgment_result(
                        blocks_to_extract=[b['range'] for b in blocks_to_extract],
                        log_metadata=log_metadata,
                        status="success"
                    )
                    print(f"Saved judgment_agents.json")

            if phase == 'judgment_only':
                print("\nJudgment phase completed (judgment_only mode)")
                return {
                    "status": "success",
                    "blocks_to_extract": blocks_to_extract
                }

            merged_blocks = blocks_to_extract
            print(f"\nTotal {len(merged_blocks)} blocks to extract")

            extracted_errors = []
            skip_extraction = False

            if log_metadata and self.checkpoint_manager.check_agent_completed('extraction_agents', log_metadata):
                saved_extraction = self.checkpoint_manager.load_agent_result('extraction_agents', log_metadata)
                if saved_extraction:
                    print("Detected extraction_agents.json exists, skipping extraction phase")
                    final_ranges_raw = saved_extraction.get('final_ranges', [])

                    extracted_errors = [
                        {
                            'block': tuple(r),
                            'error_ranges': [tuple(r)]
                        }
                        for r in final_ranges_raw
                    ]
                    skip_extraction = True

            if not skip_extraction:
                print(f"\n=== Step 2: Extraction Phase ===")
                print(f"Total {len(merged_blocks)} blocks to extract")

                extraction_stats = {
                    "blocks_processed": 0,
                    "total_ranges_extracted": 0,
                    "large_blocks_sliding_window": 0
                }

                LARGE_BLOCK_THRESHOLD = 300

                for block_idx, block_info in enumerate(merged_blocks):
                    block_start, block_end = block_info['range']
                    block_lines_count = block_end - block_start + 1
                    expanded_up = block_info.get('expanded_blocks_up', 0)
                    expanded_down = block_info.get('expanded_blocks_down', 0)

                    print(f"\nExtracting block {block_idx + 1}/{len(merged_blocks)}: [{block_start}-{block_end}] ({block_lines_count} lines)")
                    if expanded_up > 0 or expanded_down > 0:
                        print(f"   (Expanded: up {expanded_up} blocks, down {expanded_down} blocks)")

                    if block_lines_count > LARGE_BLOCK_THRESHOLD:
                        print(f"   Block exceeds {LARGE_BLOCK_THRESHOLD} lines, using sliding window scan")
                        extraction_stats["large_blocks_sliding_window"] += 1

                        sliding_ranges = self._process_large_block_with_sliding_window(block_info, log_lines, log_metadata)

                        if sliding_ranges:
                            extracted_errors.append({
                                'block': (block_start, block_end),
                                'error_ranges': sliding_ranges,
                                'expanded_blocks_up': expanded_up,
                                'expanded_blocks_down': expanded_down,
                                'processed_with_sliding_window': True
                            })
                            print(f"Sliding window extracted {len(sliding_ranges)} error ranges")
                            extraction_stats["total_ranges_extracted"] += len(sliding_ranges)
                    else:
                        block_content = "\n".join(log_lines[block_start-1:block_end])

                        extraction_input = {
                            'log_content': block_content,
                            'block_start': block_start,
                            'block_end': block_end,
                            'log_metadata': log_metadata,
                            'block_index': block_idx
                        }

                        pruning_result = self.pruning_agent.process(extraction_input)
                        irrelevant_ranges = pruning_result.get('irrelevant_ranges', [])

                        final_ranges = self._subtract_ranges((block_start, block_end), irrelevant_ranges)

                        if final_ranges:
                            extracted_errors.append({
                                'block': (block_start, block_end),
                                'error_ranges': final_ranges,
                                'expanded_blocks_up': expanded_up,
                                'expanded_blocks_down': expanded_down,
                                'irrelevant_ranges': irrelevant_ranges
                            })
                            print(f"Identified {len(irrelevant_ranges)} irrelevant ranges, retained {len(final_ranges)} error-related ranges")
                            extraction_stats["total_ranges_extracted"] += len(final_ranges)
                        else:
                            print(f"Entire block is irrelevant content, skipped")

                    extraction_stats["blocks_processed"] += 1

                if log_metadata:
                    all_final_ranges = []
                    for err in extracted_errors:
                        all_final_ranges.extend(err['error_ranges'])

                    self.result_saver.save_extraction_result(
                        final_ranges=all_final_ranges,
                        log_metadata=log_metadata
                    )

            print(f"\nTotal {len(extracted_errors)} error blocks extracted")

            if self.early_termination_triggered:
                print(f"Early termination triggered, remaining blocks not processed")

            final_statistics = {
                "total_timestamp_blocks_unfiltered": total_timestamp_blocks,
                "planned_timestamp_blocks": len(filtered_blocks),
                "visited_timestamp_blocks": self.block_stats.visited_count,
                "blocks_with_errors": len(merged_blocks),
                "early_termination": {
                    "triggered": self.early_termination_triggered,
                    "consecutive_no_error_count": self.consecutive_no_error_count,
                    "result": self.early_termination_result
                }
            }

            result = {
                "status": "success",
                "extracted_errors": extracted_errors,
                "total_blocks_processed": self.block_stats.visited_count,
                "blocks_with_errors": len(merged_blocks),
                "statistics": final_statistics,
                "early_termination": {
                    "triggered": self.early_termination_triggered,
                    "consecutive_no_error_count": self.consecutive_no_error_count,
                    "result": self.early_termination_result
                },
            }

            return result

        except Exception as e:
            print(f"Conventional parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e),
                "extracted_errors": []
            }

    def _check_early_termination(
        self,
        block_start: int,
        block_end: int,
        filtered_blocks: List[Tuple[int, int]],
        log_lines: List[str],
        reverse_idx: int
    ):
        self.consecutive_no_error_count += 1
        self.recent_no_error_blocks.append((block_start, block_end))

        if len(self.recent_no_error_blocks) > self.CONSECUTIVE_NO_ERROR_THRESHOLD:
            self.recent_no_error_blocks.pop(0)

        print(f"   Consecutive no-error blocks: {self.consecutive_no_error_count}")

        if (
            self.HARD_CONSECUTIVE_NO_ERROR_CUTOFF
            and self.error_blocks_found_count > 0
            and self.consecutive_no_error_count >= self.HARD_CONSECUTIVE_NO_ERROR_CUTOFF
        ):
            print(
                "Hard early termination cutoff reached "
                f"({self.consecutive_no_error_count} consecutive no-error blocks after first error)"
            )
            self.early_termination_triggered = True
            self.early_termination_result = {
                "status": "success",
                "should_terminate": True,
                "reason": "hard_consecutive_no_error_cutoff",
                "consecutive_no_error_count": self.consecutive_no_error_count,
                "error_blocks_found_count": self.error_blocks_found_count
            }
            return

        if self.consecutive_no_error_count > 5 and self.consecutive_no_error_count % 2 == 0:
            print(f"\nConsecutive {self.consecutive_no_error_count} blocks with no errors, triggering early termination check...")

            recent_log_content = self._get_recent_blocks_content(
                self.recent_no_error_blocks,
                log_lines
            )

            termination_result = self.early_termination_agent.process({
                'log_content': recent_log_content,
                'consecutive_no_error_count': self.consecutive_no_error_count,
                'current_position': (block_start, block_end),
                'total_blocks': len(filtered_blocks),
                'processed_blocks': reverse_idx + 1
            })

            self.early_termination_result = termination_result

            if termination_result.get('should_terminate', False):
                print(f"Early termination judgment result: should terminate")
                print(f"   Reason: {termination_result.get('reason', 'N/A')}")
                self.early_termination_triggered = True
            else:
                print(f"Early termination judgment result: continue processing")
                print(f"   Reason: {termination_result.get('reason', 'N/A')}")

    def _find_block_index_in_timestamp_blocks(self, target_block: Tuple[int, int],
                                              timestamp_blocks: List[Tuple[int, int]]) -> Optional[int]:
        for idx, block in enumerate(timestamp_blocks):
            if block == target_block:
                return idx
        return None

    def _get_context_blocks(self, filtered_blocks: List[Tuple[int, int]],
                           current_idx: int, log_lines: List[str]) -> str:
        parts = []

        if current_idx > 0:
            prev_start, prev_end = filtered_blocks[current_idx - 1]
            prev_content = "\n".join(log_lines[prev_start-1:prev_end])
            parts.append(f"=== Previous Block [{prev_start}-{prev_end}] ===\n{prev_content}")

        curr_start, curr_end = filtered_blocks[current_idx]
        curr_content = "\n".join(log_lines[curr_start-1:curr_end])
        parts.append(f"=== Current Block [{curr_start}-{curr_end}] ===\n{curr_content}")

        if current_idx < len(filtered_blocks) - 1:
            next_start, next_end = filtered_blocks[current_idx + 1]
            next_content = "\n".join(log_lines[next_start-1:next_end])
            parts.append(f"=== Next Block [{next_start}-{next_end}] ===\n{next_content}")

        return "\n\n".join(parts)

    def _get_recent_blocks_content(self, recent_blocks: List[Tuple[int, int]],
                                   log_lines: List[str]) -> str:
        parts = []
        max_total_chars = 120000
        max_block_chars = max(8000, max_total_chars // max(1, len(recent_blocks)))
        for idx, (block_start, block_end) in enumerate(recent_blocks):
            block_content = "\n".join(log_lines[block_start-1:block_end])
            block_content = self._truncate_log_excerpt(block_content, max_block_chars)
            parts.append(f"=== Block {idx + 1} [{block_start}-{block_end}] ===\n{block_content}")

        return self._truncate_log_excerpt("\n\n".join(parts), max_total_chars)

    @staticmethod
    def _truncate_log_excerpt(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text

        marker = f"\n\n[... omitted {len(text) - max_chars} characters from oversized log excerpt ...]\n\n"
        if max_chars <= len(marker) + 20:
            return text[:max_chars]

        remaining = max_chars - len(marker)
        head = max(1, remaining // 3)
        tail = max(1, remaining - head)
        return text[:head] + marker + text[-tail:]

    def _process_large_block_with_sliding_window(
        self,
        block_info: Dict[str, Any],
        log_lines: List[str],
        log_metadata: Dict[str, str] = None
    ) -> List[Tuple[int, int]]:
        block_start, block_end = block_info['range']
        block_lines = log_lines[block_start-1:block_end]
        total_lines = len(block_lines)

        print(f"   Large block ({total_lines} lines) using sliding window scan...")

        sliding_agent = SlidingWindowParsingAgent(instance_id=1)

        result = sliding_agent.process({
            'raw_logs': block_lines,
            'existing_blocks': [],
            'line_offset': block_start - 1,
            'window_size': 300,
            'overlap_ratio': 0.3,
            'log_metadata': log_metadata
        })

        if result['status'] == 'success':
            extracted_blocks = result.get('extracted_blocks', [])
            print(f"   Sliding window scan found {len(extracted_blocks)} error ranges")
            return extracted_blocks
        else:
            print(f"   Sliding window scan failed: {result.get('error', 'Unknown')}")

            return [(block_start, block_end)]
