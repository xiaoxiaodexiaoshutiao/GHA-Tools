import re
import os
import statistics
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from core.config_loader import config_loader


class TimestampExtractor:
    TIMESTAMP_PATTERN = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z'

    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path

    def extract_timestamps(self) -> List[Tuple[int, datetime, str]]:
        timestamps = []

        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    match = re.search(self.TIMESTAMP_PATTERN, line)
                    if match:
                        timestamp_str = match.group(0)
                        try:
                            parts = timestamp_str.rstrip('Z').split('.')
                            if len(parts) == 2:
                                main_part = parts[0]
                                microseconds = parts[1][:6].ljust(6, '0')
                                normalized_timestamp = f"{main_part}.{microseconds}Z"
                            else:
                                normalized_timestamp = timestamp_str

                            dt = datetime.strptime(normalized_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
                            timestamps.append((line_num, dt, timestamp_str))
                        except ValueError as e:
                            print(f"Warning: Invalid timestamp format at line {line_num}: {timestamp_str}, error: {e}")
                            continue

        except FileNotFoundError:
            raise FileNotFoundError(f"Log file does not exist: {self.log_file_path}")
        except Exception as e:
            raise Exception(f"Error reading log file: {e}")

        if not timestamps:
            raise ValueError("No valid timestamps found in log file")

        return timestamps

    def get_timestamp_intervals(self, timestamps: List[Tuple[int, datetime, str]]) -> List[Tuple[int, float]]:
        if len(timestamps) < 2:
            return []

        intervals = []
        for i in range(1, len(timestamps)):
            prev_line, prev_dt, _ = timestamps[i - 1]
            curr_line, curr_dt, _ = timestamps[i]

            interval = (curr_dt - prev_dt).total_seconds()
            intervals.append((curr_line, interval))

        return intervals


class AdaptiveSegmentation:
    MIN_SEGMENTATION_THRESHOLD = 0.01

    def __init__(self, k: float = 3.0, window_size: int = 10):
        self.k = k
        self.window_size = window_size

    def _check_indentation(self, line: str) -> bool:
        if not line:
            return False
        return line[0] in (' ', '\t')

    def _all_lines_indented(self, log_lines: List[str], line_num: int, range_size: int = 3) -> bool:
        indices_to_check = []
        for offset in range(-range_size, range_size + 1):
            idx = line_num - 1 + offset
            if 0 <= idx < len(log_lines):
                indices_to_check.append(idx)

        if not indices_to_check:
            return False

        return all(self._check_indentation(log_lines[idx]) for idx in indices_to_check)

    def find_segmentation_points(
        self,
        intervals: List[Tuple[int, float]],
        log_lines: Optional[List[str]] = None,
        line_offset: int = 0
    ) -> List[Tuple[int, float, float, float]]:
        if not intervals:
            return []

        segmentation_points = []

        for i, (line_num, interval) in enumerate(intervals):
            window_start = max(0, i - self.window_size + 1)
            window_end = i + 1

            window_intervals = [intervals[j][1] for j in range(window_start, window_end)]

            if len(window_intervals) < 2:
                continue

            mean = statistics.mean(window_intervals)
            std = statistics.stdev(window_intervals)

            threshold = mean + self.k * std

            if threshold < self.MIN_SEGMENTATION_THRESHOLD or interval < self.MIN_SEGMENTATION_THRESHOLD:
                continue

            if interval > threshold:
                if log_lines and self._all_lines_indented(log_lines, line_num, range_size=3):
                    absolute_line_num = line_num + line_offset
                    print(f"  Skipping line {absolute_line_num}: surrounding lines all have indentation, not splitting")
                    continue

                segmentation_points.append((line_num, interval, threshold, mean))

        return segmentation_points

    def segment_log(
        self,
        timestamps: List[Tuple[int, any, str]],
        segmentation_points: List[Tuple[int, float, float, float]]
    ) -> List[Tuple[int, int]]:
        if not timestamps:
            return []

        segments = []

        if not segmentation_points:
            first_line = timestamps[0][0]
            last_line = timestamps[-1][0]
            segments.append((first_line, last_line))
            return segments

        split_lines = [sp[0] for sp in segmentation_points]

        start_line = timestamps[0][0]

        for split_line in split_lines:
            end_line = split_line - 1

            if start_line <= end_line:
                segments.append((start_line, end_line))

            start_line = split_line

        last_line = timestamps[-1][0]
        if start_line <= last_line:
            segments.append((start_line, last_line))

        return segments

    def get_statistics(
        self,
        intervals: List[Tuple[int, float]],
        segmentation_points: List[Tuple[int, float, float, float]]
    ) -> dict:
        if not intervals:
            return {
                'total_intervals': 0,
                'total_segmentation_points': 0,
                'total_segments': 0,
                'avg_interval': 0,
                'max_interval': 0,
                'min_interval': 0
            }

        interval_values = [iv[1] for iv in intervals]

        return {
            'total_intervals': len(intervals),
            'total_segmentation_points': len(segmentation_points),
            'total_segments': len(segmentation_points) + 1,
            'avg_interval': statistics.mean(interval_values),
            'max_interval': max(interval_values),
            'min_interval': min(interval_values),
            'k_value': self.k,
            'window_size': self.window_size
        }


class TimestampSegmenter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = config_loader.get_agent_config('timestamp_segmenter')

        self.config = config
        self.k_value = config.get('k_value', 2.0)
        self.window_size = config.get('window_size', 50)

    def segment_logs(self, log_lines: List[str], k_value: Optional[float] = None,
                     window_size: Optional[int] = None, line_offset: int = 0):
        if k_value is None:
            k_value = self.k_value
        if window_size is None:
            window_size = self.window_size

        print("=" * 80)
        print("Log Timestamp Adaptive Segmentation")
        print("=" * 80)
        print()

        print(f"Log lines: {len(log_lines)}")
        if line_offset > 0:
            print(f"Line number offset: {line_offset} (output line numbers will be offset by this value)")
        print(f"Threshold coefficient k: {k_value}")
        print(f"Sliding window size: {window_size}")
        print()

        print("Step 1: Extracting timestamps...")
        timestamps = self._extract_timestamps_from_lines(log_lines)
        print(f"  ✓ Successfully extracted {len(timestamps)} timestamps")

        print("\nStep 2: Calculating time intervals...")
        intervals = self._get_timestamp_intervals(timestamps)
        print(f"  ✓ Successfully calculated {len(intervals)} time intervals")

        print("\nStep 3: Analyzing with adaptive threshold method...")
        segmenter = AdaptiveSegmentation(k=k_value, window_size=window_size)
        segmentation_points = segmenter.find_segmentation_points(intervals, log_lines, line_offset)
        print(f"  ✓ Detected {len(segmentation_points)} segmentation points")

        statistics_info = segmenter.get_statistics(intervals, segmentation_points)
        print(f"  ✓ Total segments: {statistics_info['total_segments']}")

        print("\nStep 4: Segmenting log...")
        segments = segmenter.segment_log(timestamps, segmentation_points)
        print(f"  ✓ Successfully divided into {len(segments)} segments")

        print()
        print("=" * 80)
        print("Processing complete!")
        print("=" * 80)

        if segmentation_points:
            print("\nSegmentation Points Summary:")
            print("-" * 80)
            for idx, (line_num, interval, threshold, mean) in enumerate(segmentation_points, 1):
                timestamp_str = None
                for ln, dt, ts_str in timestamps:
                    if ln == line_num:
                        timestamp_str = ts_str
                        break

                absolute_line_num = line_num + line_offset
                print(f"Segmentation point {idx}: Line {absolute_line_num}, Time {timestamp_str}, "
                      f"Interval {interval:.10f}s (Threshold: {threshold:.10f}s)")
            print("-" * 80)
        else:
            print("\nNo segmentation points detected, entire log is one continuous segment.")

        return timestamps, segmentation_points, segments, statistics_info

    def _extract_timestamps_from_lines(self, log_lines: List[str]) -> List[Tuple[int, datetime, str]]:
        timestamps = []
        pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z'

        for line_num, line in enumerate(log_lines, 1):
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(0)
                try:
                    parts = timestamp_str.rstrip('Z').split('.')
                    if len(parts) == 2:
                        main_part = parts[0]
                        microseconds = parts[1][:6].ljust(6, '0')
                        normalized_timestamp = f"{main_part}.{microseconds}Z"
                    else:
                        normalized_timestamp = timestamp_str

                    dt = datetime.strptime(normalized_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
                    timestamps.append((line_num, dt, timestamp_str))
                except ValueError as e:
                    print(f"Warning: Invalid timestamp format at line {line_num}: {timestamp_str}, error: {e}")
                    continue

        if not timestamps:
            raise ValueError("No valid timestamps found in log")

        return timestamps

    def _get_timestamp_intervals(self, timestamps: List[Tuple[int, datetime, str]]) -> List[Tuple[int, float]]:
        if len(timestamps) < 2:
            return []

        intervals = []
        for i in range(1, len(timestamps)):
            prev_line, prev_dt, _ = timestamps[i - 1]
            curr_line, curr_dt, _ = timestamps[i]
            interval = (curr_dt - prev_dt).total_seconds()
            intervals.append((curr_line, interval))

        return intervals

    def find_segmentation_points(
        self,
        intervals: List[Tuple[int, float]],
        k: float = 3.0,
        window_size: int = 10,
        log_lines: Optional[List[str]] = None,
        line_offset: int = 0
    ) -> List[Tuple[int, float, float, float]]:
        segmenter = AdaptiveSegmentation(k=k, window_size=window_size)
        return segmenter.find_segmentation_points(intervals, log_lines, line_offset)
