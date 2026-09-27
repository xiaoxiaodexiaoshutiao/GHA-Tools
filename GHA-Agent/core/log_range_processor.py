from typing import Dict, List, Tuple, Optional
from core.preprocessor import LogPreprocessor
from core.config_loader import config_loader
import os
import json


class LogRangeProcessor:
    def __init__(self):
        config = config_loader.get_agent_config('log_range_processor')
        self.max_line_length = config.get('max_line_length', 10000)
        self.keep_length = config.get('keep_length', 29)

    @staticmethod
    def extract_failed_step_names(json_data: Dict) -> List[str]:
        failed_steps = []

        if 'steps' in json_data and isinstance(json_data['steps'], list):
            for step in json_data['steps']:
                if isinstance(step, dict) and step.get('conclusion') == 'failure':
                    step_name = step.get('name')
                    if step_name:
                        failed_steps.append(step_name)
                        print(f"Found failed step: '{step_name}'")

        return failed_steps

    @staticmethod
    def get_json_path_from_log_path(log_path: str) -> Optional[str]:
        log_dir = os.path.dirname(log_path)

        dir_name = os.path.basename(log_dir)

        if dir_name.isdigit():
            json_path = os.path.join(log_dir, f"{dir_name}.json")
            if os.path.exists(json_path):
                print(f"Found corresponding JSON file: {json_path}")
                return json_path
            else:
                print(f"JSON file does not exist: {json_path}")
        else:
            print(f"Cannot extract run_id from directory name: {dir_name}")

        return None

    @staticmethod
    def load_failed_step_names_from_log_path(log_path: str) -> Optional[List[str]]:
        json_path = LogRangeProcessor.get_json_path_from_log_path(log_path)
        if json_path is None:
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            failed_steps = LogRangeProcessor.extract_failed_step_names(json_data)
            print(f"Loaded {len(failed_steps)} failed steps from JSON file")
            return failed_steps if failed_steps else None
        except Exception as e:
            print(f"Failed to read JSON file: {e}")
            return None

    def truncate_long_line(self, line: str) -> str:
        if len(line) > self.max_line_length:
            return line[:self.keep_length]
        return line

    def process_log_range(self, logs: List[str], failed_step_names: Optional[List[str]] = None) -> List[Dict]:
        print("Starting log processing range identification...")

        total_lines = len(logs)

        processed_lines = []
        for line in logs:
            processed_line = LogPreprocessor.preprocess_line(line)
            processed_lines.append(processed_line)

        error_lines = []
        for i, line in enumerate(processed_lines):
            if line.startswith("##[error]"):
                error_lines.append(i + 1)

        if not error_lines:
            print("No '##[error]' markers found")

            if failed_step_names:
                print("Trying to match log range using failed_step_names")
                start_line = self._find_step_start_line(failed_step_names, processed_lines, total_lines)

                if start_line is not None:
                    end_line = total_lines
                    for i in range(start_line - 1, total_lines):
                        if "Post job cleanup." in processed_lines[i]:
                            end_line = i + 1
                            print(f"Found 'Post job cleanup.' at line {end_line}, setting as range end")
                            break

                    workflow_end = None
                    for i in range(start_line - 1, end_line):
                        if processed_lines[i].startswith("##[endgroup]"):
                            workflow_end = i + 1
                            print(f"Found workflow end marker: line {workflow_end}")
                            break

                    if workflow_end is None:
                        print(f"No workflow end marker found, workflow_range set to None")

                    print(f"Range determined based on failed_step_names: line {start_line} to line {end_line}")
                    if workflow_end:
                        print(f"Workflow range: line {start_line} to line {workflow_end} ({workflow_end - start_line + 1} lines)")

                    return [{
                        'log_range': (start_line, end_line),
                        'workflow_range': (start_line, workflow_end) if workflow_end else None
                    }]

            print("Returning full text range")
            return [{
                'log_range': (1, total_lines),
                'workflow_range': None
            }]

        print(f"Found {len(error_lines)} error markers")

        group_run_lines = []
        for i, line in enumerate(processed_lines):
            if line.startswith("##[group]Run "):
                group_run_lines.append(i + 1)

        if not group_run_lines:
            print("No '##[group]Run ' markers found, using log beginning")
            group_run_lines = [1]

        print(f"Found {len(group_run_lines)} '##[group]Run ' markers")

        group_mapping = {}

        for error_line in error_lines:
            start_line = None
            if failed_step_names:
                start_line = self._find_step_start_line(failed_step_names, processed_lines, error_line)

            if start_line is None:
                print(f"Using backward search logic for error line {error_line}")
                start_line = 1
                for group_run_line in reversed(group_run_lines):
                    if group_run_line < error_line:
                        start_line = group_run_line
                        break

            if start_line not in group_mapping:
                group_mapping[start_line] = []
            group_mapping[start_line].append(error_line)
            print(f"Error line {error_line} mapped to group {start_line}")

        selected_pairs = []
        for group_start, error_list in group_mapping.items():
            selected_error = max(error_list)
            selected_pairs.append((group_start, selected_error))
            print(f"Group {group_start} has {len(error_list)} error lines, selecting largest: {selected_error}")

        ranges = []
        for start_line, error_line in selected_pairs:
            final_end_line = self._determine_final_end_line(error_line, processed_lines, total_lines)

            workflow_end = None
            for i in range(start_line - 1, final_end_line):
                if processed_lines[i].startswith("##[endgroup]"):
                    workflow_end = i + 1
                    print(f"Found workflow end marker: line {workflow_end}")
                    break

            if workflow_end is None:
                print(f"No workflow end marker found, workflow_range set to None")

            range_dict = {
                'log_range': (start_line, final_end_line),
                'workflow_range': (start_line, workflow_end) if workflow_end else None
            }
            ranges.append(range_dict)

            print(f"Log range: line {start_line} to line {final_end_line} ({final_end_line - start_line + 1} lines)")
            if workflow_end:
                print(f"Workflow range: line {start_line} to line {workflow_end} ({workflow_end - start_line + 1} lines)")

        print(f"Total {len(ranges)} log ranges identified")
        return ranges

    def _find_step_start_line(self, step_names: List[str], processed_lines: List[str],
                             search_end_line: int) -> Optional[int]:
        if not step_names:
            return None

        matched_lines = []

        for step_name in step_names:
            normalized_step_name = step_name
            if step_name.lower().startswith('run '):
                normalized_step_name = step_name[4:]

            search_pattern = f"##[group]Run {normalized_step_name}"

            for i in range(0, search_end_line):
                if processed_lines[i].startswith(search_pattern):
                    matched_line = i + 1
                    matched_lines.append(matched_line)
                    print(f"Found matching step start line: line {matched_line} (step: '{step_name}')")

        if not matched_lines:
            print(f"No matching step start line found, step names: {step_names}")
            return None

        closest_line = max(matched_lines)
        print(f"Selected closest step start line: line {closest_line}")
        return closest_line

    def _determine_final_end_line(self, error_line: int, processed_lines: List[str], total_lines: int) -> int:
        error_line_content = processed_lines[error_line - 1]
        if error_line_content.startswith("##[error]Process completed with exit code"):
            print(f"Error line {error_line} is '##[error]Process completed with exit code', using it as end line")
            return error_line

        if error_line < total_lines:
            next_line_idx = error_line
            next_line = processed_lines[next_line_idx]

            if next_line.startswith("##[group]Run ") or "Post job cleanup." in next_line:
                print(f"Next line of error line {error_line} is a marker line, using error line as end line")
                return error_line

        for i in range(error_line, total_lines):
            line = processed_lines[i]

            if line.startswith("##[group]Run ") or "Post job cleanup." in line:
                marker_line = i + 1
                final_end_line = marker_line - 1
                print(f"Found marker at line {marker_line}, using previous line {final_end_line} as end line")
                return final_end_line

        print(f"No subsequent marker line found, using log end {total_lines} as end line")
        return total_lines

    @staticmethod
    def extract_range(logs: List[str], start_line: int, end_line: int) -> List[str]:
        start_idx = start_line - 1
        end_idx = end_line

        start_idx = max(0, start_idx)
        end_idx = min(len(logs), end_idx)

        extracted = logs[start_idx:end_idx]
        print(f"Extracted log range: line {start_line} to line {end_line}, {len(extracted)} lines total")

        return extracted
