from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    _extract_run_stat_fields,
    _load_existing_run_stat_fields,
    _result_output_dir,
    _sum_item_metric,
    _sum_item_tokens,
    discover_raw_data_logs,
    parse_log_file,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_previous_items(output_root: Path) -> Dict[str, Dict[str, Any]]:
    summary_file = output_root / "batch_summary.json"
    if not summary_file.exists():
        return {}

    try:
        with summary_file.open("r", encoding="utf-8") as f:
            previous_summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    previous_items = {}
    for item in previous_summary.get("items", []):
        log_file = item.get("log_file")
        if log_file:
            previous_items[log_file] = item
    return previous_items


def _write_summary(output_root: Path, summary: Dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_file = output_root / "batch_summary.json.tmp"
    summary_file = output_root / "batch_summary.json"
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    os.replace(tmp_file, summary_file)


def _record_from_existing(
    record: Dict[str, Any],
    raw_data_dir: str,
    output_root: str,
    previous_item: Optional[Dict[str, Any]],
    index: int,
) -> Dict[str, Any]:
    log_file = Path(record["log_file"])
    output_dir = _result_output_dir(log_file, raw_data_dir, output_root)
    result_file = output_dir / "final_summary.json"
    existing_stats = _load_existing_run_stat_fields(result_file)
    if previous_item and previous_item.get("status") == "success":
        item = dict(previous_item)
        item.setdefault("index", index)
        for key, value in existing_stats.items():
            if item.get(key) is None:
                item[key] = value
        return item

    return {
        **record["metadata"],
        "index": index,
        "log_file": str(log_file),
        "output_dir": str(output_dir),
        "status": "success",
        "error": None,
        "method_used": None,
        "started_at": None,
        "completed_at": _utc_now(),
        "elapsed_seconds": None,
        "note": "Existing final_summary.json found; no previous elapsed time available",
        **existing_stats,
    }


def _run_one(record: Dict[str, Any], raw_data_dir: str, output_root: str, index: int) -> Dict[str, Any]:
    item_start_time = time.time()
    item_started_at = _utc_now()
    log_file = Path(record["log_file"])
    output_dir = _result_output_dir(log_file, raw_data_dir, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"

    with run_log.open("w", encoding="utf-8") as log_stream:
        log_stream.write(f"\n=== Parallel item {index} started at {item_started_at} ===\n")
        log_stream.flush()
        try:
            with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
                result = parse_log_file(
                    log_file_path=str(log_file),
                    output_dir=str(output_dir),
                    log_metadata={
                        **record["metadata"],
                        "log_file": str(log_file),
                    },
                )
        except Exception as exc:
            import traceback

            traceback.print_exc(file=log_stream)
            result = {"status": "failed", "error": str(exc)}

        completed_at = _utc_now()
        elapsed = time.time() - item_start_time
        log_stream.write(f"\n=== Parallel item {index} completed at {completed_at}; elapsed {elapsed:.2f}s ===\n")

    status = result.get("status", "failed")
    return {
        **record["metadata"],
        "index": index,
        "log_file": str(log_file),
        "output_dir": str(output_dir),
        "run_log": str(run_log),
        "status": status,
        "error": result.get("error"),
        "method_used": result.get("method_used"),
        "started_at": item_started_at,
        "completed_at": completed_at,
        "elapsed_seconds": round(elapsed, 3),
        **_extract_run_stat_fields(result),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel full runner for raw_data logs")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--output-dir", default="results/full_run")
    parser.add_argument("--difficulty")
    parser.add_argument("--language")
    parser.add_argument("--repository")
    parser.add_argument("--post-id")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_dir)
    batch_started_at = _utc_now()
    batch_start = time.time()

    records = discover_raw_data_logs(
        raw_data_dir=args.raw_data_dir,
        difficulty=args.difficulty,
        language=args.language,
        repository=args.repository,
        post_id=args.post_id,
        run_id=args.run_id,
        limit=args.limit,
    )
    previous_items = _load_previous_items(output_root)

    items_by_log_file: Dict[str, Dict[str, Any]] = {}
    pending = []
    skipped = 0

    for index, record in enumerate(records, 1):
        log_file = Path(record["log_file"])
        output_dir = _result_output_dir(log_file, args.raw_data_dir, args.output_dir)
        result_file = output_dir / "final_summary.json"
        previous_item = previous_items.get(str(log_file))

        if not args.no_skip_existing and result_file.exists():
            items_by_log_file[str(log_file)] = _record_from_existing(
                record, args.raw_data_dir, args.output_dir, previous_item, index
            )
            continue

        pending.append((index, record))

    def current_summary(status: str, completed_at: Optional[str] = None) -> Dict[str, Any]:
        ordered_items = [
            items_by_log_file[str(record["log_file"])]
            for record in records
            if str(record["log_file"]) in items_by_log_file
        ]
        processed = sum(1 for item in ordered_items if item.get("status") == "success")
        failed = sum(1 for item in ordered_items if item.get("status") == "failed")
        return {
            "status": status,
            "raw_data_dir": args.raw_data_dir,
            "output_root": args.output_dir,
            "workers": args.workers,
            "started_at": batch_started_at,
            "completed_at": completed_at,
            "elapsed_seconds": round(time.time() - batch_start, 3),
            "total": len(records),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "pending": len(records) - len(ordered_items),
            "total_tokens": _sum_item_tokens(ordered_items),
            "total_llm_elapsed_seconds": _sum_item_metric(ordered_items, "llm_elapsed_seconds"),
            "total_parser_elapsed_seconds": _sum_item_metric(ordered_items, "parser_elapsed_seconds"),
            "items": ordered_items,
        }

    _write_summary(output_root, current_summary("running"))
    print(f"Discovered {len(records)} log file(s)")
    print(f"Existing completed: {len(items_by_log_file)}")
    print(f"Pending: {len(pending)}")
    print(f"Workers: {args.workers}")

    if pending:
        max_workers = max(1, args.workers)
        pending_cursor = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            in_flight = {}

            def submit_next() -> None:
                nonlocal pending_cursor
                if pending_cursor >= len(pending):
                    return
                index, record = pending[pending_cursor]
                pending_cursor += 1
                future = executor.submit(_run_one, record, args.raw_data_dir, args.output_dir, index)
                in_flight[future] = (index, record)
                print(
                    f"Started [{index}/{len(records)}] "
                    f"in_flight={len(in_flight)}/{max_workers} {record['log_file']}",
                    flush=True,
                )

            for _ in range(min(max_workers, len(pending))):
                submit_next()

            while in_flight:
                done_futures, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done_futures:
                    index, record = in_flight.pop(future)
                    log_file = str(record["log_file"])
                    try:
                        item = future.result()
                    except Exception as exc:
                        output_dir = _result_output_dir(Path(log_file), args.raw_data_dir, args.output_dir)
                        item = {
                            **record["metadata"],
                            "index": index,
                            "log_file": log_file,
                            "output_dir": str(output_dir),
                            "status": "failed",
                            "error": str(exc),
                            "started_at": None,
                            "completed_at": _utc_now(),
                            "elapsed_seconds": None,
                        }

                    items_by_log_file[log_file] = item
                    _write_summary(output_root, current_summary("running"))
                    done = len(items_by_log_file)
                    print(
                        f"[{done}/{len(records)}] {item.get('status')} "
                        f"{item.get('elapsed_seconds')}s {log_file}",
                        flush=True,
                    )

                    submit_next()

    final = current_summary("success")
    if final["failed"]:
        final["status"] = "partial_failure"
    final["completed_at"] = _utc_now()
    final["elapsed_seconds"] = round(time.time() - batch_start, 3)
    _write_summary(output_root, final)
    print(f"Batch summary saved to: {output_root / 'batch_summary.json'}")
    print(f"Processed: {final['processed']}, failed: {final['failed']}, pending: {final['pending']}")
    return 0 if final["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
