import sys
import os
import json
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from agents.classification_agent import ClassificationAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.sliding_window_parsing_agent import SlidingWindowParsingAgent
from core.log_range_processor import LogRangeProcessor
from core.run_stats import run_stats
from pipeline.conventional_parser import ConventionalParser
from agents.pruning import PruningPipelineFactory


def sort_and_merge_blocks(blocks: List) -> List[List[int]]:
    if not blocks:
        return []

    normalized = []
    for block in blocks:
        if isinstance(block, dict):
            start, end = block['range'] if 'range' in block else (block['start'], block['end'])
        elif isinstance(block, (list, tuple)):
            start, end = block[0], block[1]
        else:
            continue
        normalized.append([start, end])

    normalized.sort(key=lambda x: x[0])

    merged = [normalized[0]]
    for current in normalized[1:]:
        last = merged[-1]
        if last[1] + 1 >= current[0]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)

    return merged


def split_block_with_sliding_window(block_start: int, block_end: int,
                                     window_size: int = 300,
                                     overlap_ratio: float = 0.3) -> List[Tuple[int, int]]:
    block_size = block_end - block_start + 1
    if block_size <= window_size:
        return [(block_start, block_end)]

    overlap = int(window_size * overlap_ratio)
    step = window_size - overlap

    windows = []
    current_start = block_start
    while current_start <= block_end:
        current_end = min(current_start + window_size - 1, block_end)
        windows.append((current_start, current_end))
        if current_end >= block_end:
            break
        current_start += step

    return windows


def compute_kept_ranges_from_delete_lines(block_start: int, block_end: int,
                                           delete_lines: Set[int]) -> List[Tuple[int, int]]:
    if not delete_lines:
        return [(block_start, block_end)]

    kept_ranges = []
    current_start = None
    for line_no in range(block_start, block_end + 1):
        if line_no not in delete_lines:
            if current_start is None:
                current_start = line_no
        else:
            if current_start is not None:
                kept_ranges.append((current_start, line_no - 1))
                current_start = None

    if current_start is not None:
        kept_ranges.append((current_start, block_end))

    return kept_ranges


def merge_window_results_with_overlap_confirmation(
    window_results: List[Dict],
    windows: List[Tuple[int, int]],
    block_start: int,
    block_end: int
) -> Tuple[Set[int], List[Tuple[int, int]]]:
    if len(window_results) == 0:
        return set(), [(block_start, block_end)]

    if len(window_results) == 1:
        delete_lines = set(window_results[0].get('delete_lines', []))
        kept_ranges = window_results[0].get('kept_ranges', [(block_start, block_end)])
        return delete_lines, kept_ranges

    window_delete_sets = [set(r.get('delete_lines', [])) for r in window_results]

    line_to_windows: Dict[int, List[int]] = {}
    for i, (w_start, w_end) in enumerate(windows):
        for line_no in range(w_start, w_end + 1):
            if line_no not in line_to_windows:
                line_to_windows[line_no] = []
            line_to_windows[line_no].append(i)

    final_delete_lines: Set[int] = set()
    for line_no in range(block_start, block_end + 1):
        window_indices = line_to_windows.get(line_no, [])
        if len(window_indices) == 0:
            continue
        elif len(window_indices) == 1:
            w_idx = window_indices[0]
            if line_no in window_delete_sets[w_idx]:
                final_delete_lines.add(line_no)
        else:
            all_agree = all(line_no in window_delete_sets[w_idx] for w_idx in window_indices)
            if all_agree:
                final_delete_lines.add(line_no)

    kept_ranges = compute_kept_ranges_from_delete_lines(block_start, block_end, final_delete_lines)
    return final_delete_lines, kept_ranges


def _process_single_block(pipeline, log_lines: List[str],
                          block_start: int, block_end: int,
                          block_idx: int,
                          log_metadata: Dict[str, str]) -> Optional[Dict]:
    block_content = "\n".join(log_lines[block_start - 1:block_end])

    pruning_result = pipeline.process(
        log_content=block_content,
        block_start=block_start,
        block_end=block_end,
        block_id=str(block_idx),
        log_metadata=log_metadata
    )

    if pruning_result.get('status') != 'success':
        print(f"    Pruning failed: {pruning_result.get('error', 'Unknown')}")
        return {
            'block': [block_start, block_end],
            'error_ranges': [[block_start, block_end]],
            'irrelevant_ranges': [],
            'statistics': {}
        }

    kept_ranges = pruning_result.get('kept_ranges', [(block_start, block_end)])
    irrelevant_ranges = pruning_result.get('irrelevant_ranges', [])
    statistics = pruning_result.get('statistics', {})

    if kept_ranges:
        delete_count = len(pruning_result.get('delete_lines', []))
        print(f"    Deleted {delete_count} lines, kept {len(kept_ranges)} error-related ranges")
        return {
            'block': [block_start, block_end],
            'error_ranges': kept_ranges,
            'irrelevant_ranges': irrelevant_ranges,
            'statistics': statistics
        }
    else:
        print(f"    Entire block is irrelevant, skipped")
        return None


def _process_large_block_with_sliding_window(
    pipeline, log_lines: List[str],
    block_start: int, block_end: int,
    block_idx: int, log_metadata: Dict[str, str],
    window_size: int = 300, overlap_ratio: float = 0.3,
    max_llm_windows: int = 20,
    max_llm_block_lines: int = 5000
) -> Optional[Dict]:
    windows = split_block_with_sliding_window(block_start, block_end, window_size, overlap_ratio)

    block_size = block_end - block_start + 1
    print(f"    Block too large ({block_size} lines), splitting into {len(windows)} windows")
    print(f"    Window size: {window_size}, overlap: {overlap_ratio * 100:.0f}%")

    if len(windows) > max_llm_windows or block_size > max_llm_block_lines:
        print(
            "    Block exceeds pruning LLM budget; "
            "keeping entire error block without per-window LLM pruning"
        )
        return {
            'block': [block_start, block_end],
            'error_ranges': [[block_start, block_end]],
            'irrelevant_ranges': [],
            'statistics': {
                'sliding_window': True,
                'window_count': len(windows),
                'fast_path': 'kept_entire_large_block',
                'max_llm_windows': max_llm_windows,
                'max_llm_block_lines': max_llm_block_lines
            }
        }

    window_results = []
    all_failed = True

    for w_idx, (w_start, w_end) in enumerate(windows):
        w_size = w_end - w_start + 1
        print(f"\n    Window {w_idx + 1}/{len(windows)}: [{w_start}-{w_end}] ({w_size} lines)")

        window_content = "\n".join(log_lines[w_start - 1:w_end])

        result = pipeline.process(
            log_content=window_content,
            block_start=w_start,
            block_end=w_end,
            block_id=f"{block_idx}_{w_idx}",
            log_metadata=log_metadata
        )

        if result.get('status') == 'success':
            all_failed = False
            print(f"      Window pruned: deleted {len(result.get('delete_lines', []))} lines")
        else:
            print(f"      Window pruning failed: {result.get('error', 'Unknown')}")
            result = {
                'status': 'failed',
                'delete_lines': [],
                'kept_ranges': [(w_start, w_end)]
            }

        window_results.append(result)

    if all_failed:
        print(f"    All windows failed, keeping entire block")
        return {
            'block': [block_start, block_end],
            'error_ranges': [[block_start, block_end]],
            'irrelevant_ranges': [],
            'statistics': {'sliding_window': True, 'window_count': len(windows), 'all_failed': True}
        }

    print(f"\n    Merging window results (overlap confirmation)...")
    final_delete_lines, kept_ranges = merge_window_results_with_overlap_confirmation(
        window_results, windows, block_start, block_end
    )

    irrelevant_ranges = []
    if final_delete_lines:
        sorted_delete = sorted(final_delete_lines)
        range_start = sorted_delete[0]
        range_end = sorted_delete[0]
        for line_no in sorted_delete[1:]:
            if line_no == range_end + 1:
                range_end = line_no
            else:
                irrelevant_ranges.append([range_start, range_end])
                range_start = line_no
                range_end = line_no
        irrelevant_ranges.append([range_start, range_end])

    if kept_ranges:
        print(f"    Merged: deleted {len(final_delete_lines)} lines, kept {len(kept_ranges)} ranges")
        return {
            'block': [block_start, block_end],
            'error_ranges': [list(r) for r in kept_ranges],
            'irrelevant_ranges': irrelevant_ranges,
            'statistics': {
                'sliding_window': True,
                'window_count': len(windows),
                'overlap_ratio': overlap_ratio,
                'final_delete_count': len(final_delete_lines)
            }
        }
    else:
        print(f"    Entire block is irrelevant, skipped")
        return None


def perform_pruning(blocks_to_extract: List, log_lines: List[str],
                    log_metadata: Dict[str, str],
                    window_size: int = 300,
                    overlap_ratio: float = 0.3) -> List[Dict]:
    merged_blocks = sort_and_merge_blocks(blocks_to_extract)

    original_count = len(blocks_to_extract)
    merged_count = len(merged_blocks)
    if original_count != merged_count:
        print(f"  Blocks sorted and merged: {original_count} -> {merged_count}")

    pipeline = PruningPipelineFactory.create(
        enable_llm=True,
        verbose=True,
        strict_mode=False
    )
    large_block_config = pipeline.config.get('large_block_fast_path', {})
    max_llm_windows = int(large_block_config.get('max_llm_windows', 20))
    max_llm_block_lines = int(large_block_config.get('max_llm_block_lines', 5000))

    extracted_errors = []

    for block_idx, block in enumerate(merged_blocks):
        block_start, block_end = block[0], block[1]
        block_lines_count = block_end - block_start + 1

        print(f"\n  Pruning block {block_idx + 1}/{len(merged_blocks)}: "
              f"[{block_start}-{block_end}] ({block_lines_count} lines)")

        if block_lines_count > window_size:
            result = _process_large_block_with_sliding_window(
                pipeline=pipeline,
                log_lines=log_lines,
                block_start=block_start,
                block_end=block_end,
                block_idx=block_idx,
                log_metadata=log_metadata,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                max_llm_windows=max_llm_windows,
                max_llm_block_lines=max_llm_block_lines
            )
        else:
            result = _process_single_block(
                pipeline=pipeline,
                log_lines=log_lines,
                block_start=block_start,
                block_end=block_end,
                block_idx=block_idx,
                log_metadata=log_metadata
            )

        if result:
            extracted_errors.append(result)

    return extracted_errors


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_run_stat_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    run_statistics = result.get("run_statistics") or {}
    return {
        "parser_elapsed_seconds": run_statistics.get("parser_elapsed_seconds"),
        "llm_elapsed_seconds": run_statistics.get("llm_elapsed_seconds"),
        "total_tokens": run_statistics.get("total_tokens"),
        "llm_call_count": run_statistics.get("llm_call_count"),
        "token_count_available": run_statistics.get("token_count_available"),
    }


def _load_existing_run_stat_fields(result_file: Path) -> Dict[str, Any]:
    if not result_file.exists():
        return _extract_run_stat_fields({})
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            return _extract_run_stat_fields(json.load(f))
    except (OSError, json.JSONDecodeError):
        return _extract_run_stat_fields({})


def _sum_item_tokens(items: List[Dict[str, Any]]) -> int:
    total = 0
    for item in items:
        value = item.get("total_tokens")
        if isinstance(value, int):
            total += value
    return total


def _sum_item_metric(items: List[Dict[str, Any]], key: str) -> float:
    total = 0.0
    for item in items:
        value = item.get(key)
        if isinstance(value, (int, float)):
            total += float(value)
    return round(total, 3)


def _attach_run_statistics(
    result: Dict[str, Any],
    *,
    parse_start_time: float,
    completed_at: str
) -> Dict[str, Any]:
    parser_elapsed = time.time() - parse_start_time
    run_statistics = run_stats.snapshot(
        parser_elapsed_seconds=parser_elapsed,
        completed_at=completed_at,
    )
    result["run_statistics"] = run_statistics

    statistics = result.setdefault("statistics", {})
    statistics["parser_elapsed_seconds"] = run_statistics["parser_elapsed_seconds"]
    statistics["llm_elapsed_seconds"] = run_statistics["llm_elapsed_seconds"]
    statistics["total_tokens"] = run_statistics["total_tokens"]
    statistics["llm_call_count"] = run_statistics["llm_call_count"]
    statistics["token_count_available"] = run_statistics["token_count_available"]
    return result


def _save_parse_outputs(result: Dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "final_summary.json")
    stats_file = os.path.join(output_dir, "run_statistics.json")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(result.get("run_statistics", {}), f, indent=2, ensure_ascii=False)


def parse_log_file(log_file_path: str, output_dir: str, log_metadata: Optional[Dict[str, Any]] = None) -> Dict:
    parse_started_at = _utc_now_iso()
    parse_start_time = time.time()
    run_stats.reset(started_at=parse_started_at)

    def finish(result: Dict[str, Any], save: bool = True) -> Dict[str, Any]:
        completed_at = _utc_now_iso()
        result = _attach_run_statistics(
            result,
            parse_start_time=parse_start_time,
            completed_at=completed_at,
        )
        if save:
            _save_parse_outputs(result, output_dir)
        return result

    print(f"\n{'=' * 80}")
    print(f"Processing: {log_file_path}")
    print(f"{'=' * 80}")

    print(f"Reading log file...")
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_lines = f.read().splitlines()
    print(f"Total lines: {len(log_lines)}")

    print(f"\n{'─' * 80}")
    print("Step 1: Preprocessing")
    print(f"{'─' * 80}")

    preprocessing_agent = PreprocessingAgent()
    try:
        preprocess_input = {
            'logs': log_lines,
            'line_offset': 0
        }
        preprocessing_result = preprocessing_agent.process(preprocess_input)

        if preprocessing_result.get('status') != 'success':
            print(f"Preprocessing failed: {preprocessing_result.get('error', 'Unknown error')}")
            return finish({'status': 'failed', 'error': 'Preprocessing failed'})

        print(f"Preprocessing completed")

    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return finish({'status': 'failed', 'error': f'Preprocessing error: {e}'})

    print(f"\n{'─' * 80}")
    print("Step 2: Classification")
    print(f"{'─' * 80}")

    classification_agent = ClassificationAgent()
    log_metadata = dict(log_metadata or {})
    log_metadata.setdefault('log_file', log_file_path)

    classification_input = {
        'workflow_content': '',
        'log_lines': log_lines,
        'log_range': (1, len(log_lines)),
        'log_metadata': log_metadata
    }

    try:
        classification_result = classification_agent.process(classification_input)
        if classification_result.get('status') != 'success':
            error = classification_result.get('error', 'Classification failed')
            print(f"Classification failed: {error}")
            return finish({'status': 'failed', 'error': f'Classification failed: {error}'})
        method = classification_result.get('method')
        print(f"Classification result: Method {method}")
        print(f"Reason: {classification_result.get('reason', 'N/A')[:200]}...")
    except Exception as e:
        print(f"Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return finish({'status': 'failed', 'error': f'Classification failed: {e}'})

    result = None

    if method == 1:
        print(f"\n{'─' * 80}")
        print("Step 3: Method 1 - Judgment + Pruning Pipeline")
        print(f"{'─' * 80}")

        try:
            print("\n--- Phase 1: Judgment (identify error blocks) ---")
            parser = ConventionalParser(result_base_path=None)

            input_data = {
                'log_lines': log_lines,
                'preprocessing_result': preprocessing_result,
                'log_metadata': log_metadata,
                'phase': 'judgment_only'
            }

            judgment_result = parser.parse(input_data)

            if judgment_result.get('status') != 'success':
                print(f"Judgment phase failed: {judgment_result.get('error', 'Unknown')}")
                return finish({'status': 'failed', 'error': 'Judgment phase failed'})

            blocks_to_extract = judgment_result.get('blocks_to_extract', [])
            print(f"Judgment completed: {len(blocks_to_extract)} error blocks identified")

            if not blocks_to_extract:
                print("No error blocks found")
                result = {
                    'status': 'success',
                    'method_used': 1,
                    'extracted_errors': [],
                    'statistics': {'total_log_lines': len(log_lines)}
                }
            else:
                print(f"\n--- Phase 2: Pruning Pipeline (T0-T6) ---")
                print(f"Pruning {len(blocks_to_extract)} blocks...")

                extracted_errors = perform_pruning(
                    blocks_to_extract, log_lines, log_metadata
                )

                result = {
                    'status': 'success',
                    'method_used': 1,
                    'extracted_errors': [
                        {'block': err['block'], 'error_ranges': err['error_ranges']}
                        for err in extracted_errors
                    ],
                    'statistics': {
                        'total_log_lines': len(log_lines),
                        'total_error_blocks': len(blocks_to_extract),
                        'pruned_blocks': len(extracted_errors)
                    }
                }

            print(f"Method 1 completed successfully")

        except Exception as e:
            print(f"Method 1 error: {e}")
            import traceback
            traceback.print_exc()
            return finish({'status': 'failed', 'error': f'Method 1 error: {e}'})

    elif method == 2:
        print(f"\n{'─' * 80}")
        print("Step 3: Method 2 - Sliding Window Parsing")
        print(f"{'─' * 80}")

        try:
            log_range_processor = LogRangeProcessor()

            log_ranges = log_range_processor.process_log_range(log_lines)
            if not log_ranges:
                log_ranges = [{'log_range': (1, len(log_lines)), 'workflow_range': None}]

            print(f"Identified {len(log_ranges)} log ranges")

            sliding_agent = SlidingWindowParsingAgent(instance_id=1)
            all_extracted_ranges = []

            for range_idx, range_info in enumerate(log_ranges):
                log_range = range_info['log_range']
                start_line, end_line = log_range

                print(f"  Processing range {range_idx + 1}/{len(log_ranges)}: lines {start_line}-{end_line}")

                range_log_lines = log_lines[start_line - 1:end_line]
                preprocess_input = {
                    'logs': range_log_lines,
                    'line_offset': start_line - 1
                }
                preprocess_result = preprocessing_agent.process(preprocess_input)

                sliding_window_input = {
                    'raw_logs': range_log_lines,
                    'existing_blocks': [],
                    'line_offset': start_line - 1,
                    'log_metadata': log_metadata
                }

                sliding_result = sliding_agent.process(sliding_window_input)

                if sliding_result.get('status') == 'success':
                    extracted_blocks = sliding_result.get('extracted_blocks', [])
                    all_extracted_ranges.extend(extracted_blocks)

            print(f"  Total extracted ranges: {len(all_extracted_ranges)}")

            result = {
                'status': 'success',
                'method_used': 2,
                'extracted_errors': [
                    {'block': block, 'error_ranges': [block]}
                    for block in all_extracted_ranges
                ],
                'statistics': {
                    'total_log_lines': len(log_lines),
                    'total_ranges': len(log_ranges)
                }
            }

            print(f"Method 2 completed successfully")

        except Exception as e:
            print(f"Method 2 error: {e}")
            import traceback
            traceback.print_exc()
            return finish({'status': 'failed', 'error': f'Method 2 error: {e}'})

    else:
        print(f"Unknown method: {method}")
        return finish({'status': 'failed', 'error': f'Unknown method: {method}'})

    print(f"\n{'─' * 80}")
    print("Step 4: Save Results")
    print(f"{'─' * 80}")

    try:
        result_file = os.path.join(output_dir, "final_summary.json")
        result = finish(result)
        print(f"Results saved to: {result_file}")
        print(f"Run statistics saved to: {os.path.join(output_dir, 'run_statistics.json')}")

    except Exception as e:
        print(f"Failed to save results: {e}")
        return finish({'status': 'failed', 'error': f'Save failed: {e}'}, save=False)

    print(f"\n✓ Processing completed successfully!")
    return result


def discover_raw_data_logs(
    raw_data_dir: str,
    difficulty: Optional[str] = None,
    language: Optional[str] = None,
    repository: Optional[str] = None,
    post_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    base = Path(raw_data_dir)
    if not base.exists():
        raise FileNotFoundError(f"raw_data directory not found: {base}")
    if limit is not None and limit <= 0:
        return []

    records = []
    for log_file in sorted(base.glob("*/*/*/*/*/*whole_log.txt")):
        rel_parts = log_file.relative_to(base).parts
        if len(rel_parts) != 6:
            continue

        meta = {
            "difficulty": rel_parts[0],
            "language": rel_parts[1],
            "repository": rel_parts[2],
            "post_id": rel_parts[3],
            "run_id": rel_parts[4],
            "log_file_name": rel_parts[5],
        }

        if difficulty and meta["difficulty"] != difficulty:
            continue
        if language and meta["language"] != language:
            continue
        if repository and meta["repository"] != repository:
            continue
        if post_id and meta["post_id"] != post_id:
            continue
        if run_id and meta["run_id"] != run_id:
            continue

        records.append({
            "log_file": log_file,
            "metadata": meta,
        })

        if limit is not None and len(records) >= limit:
            break

    return records


def _result_output_dir(log_file: Path, raw_data_dir: str, output_root: str) -> Path:
    raw_base = Path(raw_data_dir)
    run_rel_dir = log_file.parent.relative_to(raw_base)
    leaf = log_file.stem
    if leaf.endswith("~whole_log"):
        leaf = leaf[:-len("~whole_log")]
    return Path(output_root) / run_rel_dir / leaf


def parse_raw_data_tree(
    raw_data_dir: str,
    output_root: str,
    difficulty: Optional[str] = None,
    language: Optional[str] = None,
    repository: Optional[str] = None,
    post_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    skip_existing: bool = True
) -> Dict[str, Any]:
    batch_started_at = datetime.now(timezone.utc).isoformat()
    batch_start_time = time.time()

    def write_batch_summary(summary: Dict[str, Any]) -> None:
        os.makedirs(output_root, exist_ok=True)
        summary_file = Path(output_root) / "batch_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def load_previous_items() -> Dict[str, Dict[str, Any]]:
        summary_file = Path(output_root) / "batch_summary.json"
        if not summary_file.exists():
            return {}

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                previous_summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

        previous_items = {}
        for item in previous_summary.get("items", []):
            log_file = item.get("log_file")
            if log_file:
                previous_items[log_file] = item
        return previous_items

    records = discover_raw_data_logs(
        raw_data_dir=raw_data_dir,
        difficulty=difficulty,
        language=language,
        repository=repository,
        post_id=post_id,
        run_id=run_id,
        limit=limit
    )

    print(f"\nDiscovered {len(records)} log file(s) under {raw_data_dir}")
    if not records:
        summary = {
            "status": "success",
            "raw_data_dir": raw_data_dir,
            "output_root": output_root,
            "started_at": batch_started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - batch_start_time, 3),
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_tokens": 0,
            "total_llm_elapsed_seconds": 0.0,
            "total_parser_elapsed_seconds": 0.0,
            "items": []
        }
        write_batch_summary(summary)
        summary_file = Path(output_root) / "batch_summary.json"
        print(f"\nBatch summary saved to: {summary_file}")
        return summary

    items = []
    processed = 0
    skipped = 0
    failed = 0
    previous_items_by_log_file = load_previous_items()

    for index, record in enumerate(records, 1):
        item_start_time = time.time()
        item_started_at = datetime.now(timezone.utc).isoformat()
        log_file = record["log_file"]
        metadata = record["metadata"]
        output_dir = _result_output_dir(log_file, raw_data_dir, output_root)
        result_file = output_dir / "final_summary.json"

        print(f"\n[{index}/{len(records)}] {log_file}")

        if skip_existing and result_file.exists():
            previous_item = previous_items_by_log_file.get(str(log_file))
            if previous_item and previous_item.get("status") == "success":
                print(f"  Existing successful result found, preserving previous timing: {result_file}")
                print(f"  Previous item elapsed: {previous_item.get('elapsed_seconds')}s")
                processed += 1
                item = dict(previous_item)
                for key, value in _load_existing_run_stat_fields(result_file).items():
                    if item.get(key) is None:
                        item[key] = value
                items.append(item)
                partial_summary = {
                    "status": "running",
                    "raw_data_dir": raw_data_dir,
                    "output_root": output_root,
                    "started_at": batch_started_at,
                    "completed_at": None,
                    "elapsed_seconds": round(time.time() - batch_start_time, 3),
                    "total": len(records),
                    "processed": processed,
                    "skipped": skipped,
                    "failed": failed,
                    "total_tokens": _sum_item_tokens(items),
                    "total_llm_elapsed_seconds": _sum_item_metric(items, "llm_elapsed_seconds"),
                    "total_parser_elapsed_seconds": _sum_item_metric(items, "parser_elapsed_seconds"),
                    "items": items
                }
                write_batch_summary(partial_summary)
                continue

            item_elapsed = time.time() - item_start_time
            print(f"  Existing result found, skipping: {result_file}")
            print(f"  Item elapsed: {item_elapsed:.2f}s")
            skipped += 1
            existing_stats = _load_existing_run_stat_fields(result_file)
            items.append({
                **metadata,
                "log_file": str(log_file),
                "output_dir": str(output_dir),
                "status": "skipped",
                "started_at": item_started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(item_elapsed, 3),
                **existing_stats,
            })
            partial_summary = {
                "status": "running",
                "raw_data_dir": raw_data_dir,
                "output_root": output_root,
                "started_at": batch_started_at,
                "completed_at": None,
                "elapsed_seconds": round(time.time() - batch_start_time, 3),
                "total": len(records),
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "total_tokens": _sum_item_tokens(items),
                "total_llm_elapsed_seconds": _sum_item_metric(items, "llm_elapsed_seconds"),
                "total_parser_elapsed_seconds": _sum_item_metric(items, "parser_elapsed_seconds"),
                "items": items
            }
            write_batch_summary(partial_summary)
            continue

        result = parse_log_file(
            log_file_path=str(log_file),
            output_dir=str(output_dir),
            log_metadata={
                **metadata,
                "log_file": str(log_file)
            }
        )

        status = result.get("status", "failed")
        if status == "success":
            processed += 1
        else:
            failed += 1

        item_elapsed = time.time() - item_start_time
        print(f"  Item elapsed: {item_elapsed:.2f}s")

        items.append({
            **metadata,
            "log_file": str(log_file),
            "output_dir": str(output_dir),
            "status": status,
            "error": result.get("error"),
            "method_used": result.get("method_used"),
            "started_at": item_started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(item_elapsed, 3),
            **_extract_run_stat_fields(result),
        })

        partial_summary = {
            "status": "running",
            "raw_data_dir": raw_data_dir,
            "output_root": output_root,
            "started_at": batch_started_at,
            "completed_at": None,
            "elapsed_seconds": round(time.time() - batch_start_time, 3),
            "total": len(records),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "total_tokens": _sum_item_tokens(items),
            "total_llm_elapsed_seconds": _sum_item_metric(items, "llm_elapsed_seconds"),
            "total_parser_elapsed_seconds": _sum_item_metric(items, "parser_elapsed_seconds"),
            "items": items
        }
        write_batch_summary(partial_summary)

    summary = {
        "status": "success" if failed == 0 else "partial_failure",
        "raw_data_dir": raw_data_dir,
        "output_root": output_root,
        "started_at": batch_started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - batch_start_time, 3),
        "total": len(records),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_tokens": _sum_item_tokens(items),
        "total_llm_elapsed_seconds": _sum_item_metric(items, "llm_elapsed_seconds"),
        "total_parser_elapsed_seconds": _sum_item_metric(items, "parser_elapsed_seconds"),
        "items": items
    }

    write_batch_summary(summary)
    summary_file = Path(output_root) / "batch_summary.json"

    print(f"\nBatch summary saved to: {summary_file}")
    print(f"Processed: {processed}, skipped: {skipped}, failed: {failed}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GHA-Agent multi-agent parser for GitHub Actions logs"
    )
    parser.add_argument("--input", "-i", help="Parse one specific whole_log.txt file")
    parser.add_argument("--raw-data-dir", default="raw_data", help="Root raw_data directory")
    parser.add_argument("--output-dir", "-o", default="results", help="Output directory")
    parser.add_argument("--difficulty", help="Filter raw_data difficulty, e.g. basic or complex")
    parser.add_argument("--language", help="Filter raw_data language, e.g. python, java, c++")
    parser.add_argument("--repository", help="Filter repository directory name")
    parser.add_argument("--post-id", help="Filter post id")
    parser.add_argument("--run-id", help="Filter run id")
    parser.add_argument("--limit", type=int, help="Maximum number of logs to process")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess logs even when final_summary.json already exists"
    )
    return parser


def main():
    print(f"\n{'█' * 80}")
    print("GHA-Agent - Multi-Agent Log Parsing System")
    print(f"{'█' * 80}")

    args = build_arg_parser().parse_args()

    try:
        if args.input:
            print(f"\nConfiguration:")
            print(f"  Input Log:  {args.input}")
            print(f"  Output Dir: {args.output_dir}")

            if not os.path.exists(args.input):
                print(f"\n✗ Error: Log file not found: {args.input}")
                return

            result = parse_log_file(args.input, args.output_dir)

            if result.get('status') == 'success':
                print(f"\n{'█' * 80}")
                print("SUCCESS - Log parsing completed")
                print(f"{'█' * 80}")
                print(f"\nResults saved to: {args.output_dir}/final_summary.json")
            else:
                print(f"\n{'█' * 80}")
                print("FAILED - Log parsing incomplete")
                print(f"{'█' * 80}")
                print(f"\nError: {result.get('error', 'Unknown error')}")
        else:
            parse_raw_data_tree(
                raw_data_dir=args.raw_data_dir,
                output_root=args.output_dir,
                difficulty=args.difficulty,
                language=args.language,
                repository=args.repository,
                post_id=args.post_id,
                run_id=args.run_id,
                limit=args.limit,
                skip_existing=not args.no_skip_existing
            )

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\n{'█' * 80}")
        print("ERROR - Unexpected error occurred")
        print(f"{'█' * 80}")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
