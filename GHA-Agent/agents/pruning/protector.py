import re
from typing import List, Set, Dict, Optional
from .types import LogLine, Anchor, AnchorType, AnchorPriority, AnchorResult, ProtectResult
from .anchor_finder import AnchorFinder


class Protector:
    STACK_TRACE_PATTERNS = [
        re.compile(r'^\s+at\s+'),
        re.compile(r'^File ".+", line \d+'),
        re.compile(r'^\s+at [a-zA-Z0-9$.]+\(.+:\d+\)'),
        re.compile(r'^\s+at .+ in .+:line \d+'),
        re.compile(r'^#\d+\s+.+\(\d+\):'),
        re.compile(r'^Caused by:'),
        re.compile(r'^\s+\.\.\. \d+ more'),
        re.compile(r'^During handling of'),
        re.compile(r'^The above exception'),
    ]

    COMMAND_PATTERNS = [
        re.compile(r'^##\[command\]'),
        re.compile(r'^Run '),
        re.compile(r'^##\[group\]'),
        re.compile(r'^shell:'),
        re.compile(r'^working-directory:'),
        re.compile(r'^\$\s+'),
        re.compile(r'^>\s+'),
    ]

    FILE_LOCATION_PATTERNS = [
        re.compile(r'[a-zA-Z0-9_/\\.-]+:\d+:\d+'),
        re.compile(r'File ".+", line \d+'),
        re.compile(r'in .+:line \d+'),
        re.compile(r'\([a-zA-Z0-9_/\\.-]+:\d+\)'),
    ]

    def __init__(self, k_before: int = 4, k_after: int = 6, soft_window: int = 10):
        self.k_before = k_before
        self.k_after = k_after
        self.soft_window = soft_window

    def build(self, log_lines: List[LogLine], anchors: AnchorResult) -> ProtectResult:
        result = ProtectResult()

        line_dict = {line.no: line for line in log_lines}
        all_line_nos = set(line_dict.keys())
        min_line = min(all_line_nos) if all_line_nos else 0
        max_line = max(all_line_nos) if all_line_nos else 0

        hard_protect = self._build_hard_protect(
            log_lines, anchors, line_dict, min_line, max_line
        )
        result.hard_protect_set = hard_protect

        soft_protect = self._build_soft_protect(
            log_lines, anchors, line_dict, min_line, max_line, hard_protect
        )
        result.soft_protect_set = soft_protect

        result.candidate_set = all_line_nos - hard_protect

        return result

    def _build_hard_protect(self, log_lines: List[LogLine], anchors: AnchorResult,
                           line_dict: Dict[int, LogLine],
                           min_line: int, max_line: int) -> Set[int]:
        hard_protect = set()

        for anchor in anchors.anchors_hard:
            hard_protect.add(anchor.line_no)

            stack_lines = self._find_stack_trace_continuation(
                anchor.line_no, line_dict, max_line
            )
            hard_protect.update(stack_lines)

            if anchor.anchor_type == AnchorType.TRACEBACK:
                for line_no in range(anchor.line_no + 1, min(anchor.line_no + 50, max_line + 1)):
                    if line_no not in line_dict:
                        break
                    line = line_dict[line_no]

                    if self._is_stack_trace_line(line.text) or self._contains_exception(line.text):
                        hard_protect.add(line_no)
                    elif line.text.strip() == '':
                        continue
                    else:
                        if line.text.startswith(' ') and not line.text.startswith('  '):
                            break

                        if re.match(r'^[A-Z][a-zA-Z]+Error:', line.text) or \
                           re.match(r'^[A-Z][a-zA-Z]+Exception:', line.text):
                            hard_protect.add(line_no)

        for line in log_lines:
            if self._contains_file_location(line.text):
                if self._is_error_related(line.text):
                    hard_protect.add(line.no)

        return hard_protect

    def _build_soft_protect(self, log_lines: List[LogLine], anchors: AnchorResult,
                           line_dict: Dict[int, LogLine],
                           min_line: int, max_line: int,
                           hard_protect: Set[int]) -> Set[int]:
        soft_protect = set()

        for anchor in anchors.anchors_hard:
            for i in range(1, self.k_before + 1):
                line_no = anchor.line_no - i
                if line_no >= min_line and line_no not in hard_protect:
                    soft_protect.add(line_no)

            for i in range(1, self.k_after + 1):
                line_no = anchor.line_no + i
                if line_no <= max_line and line_no not in hard_protect:
                    soft_protect.add(line_no)

        for line in log_lines:
            if self._is_command_line(line.text):
                if line.no not in hard_protect:
                    soft_protect.add(line.no)

        for anchor in anchors.anchors_soft:
            if anchor.line_no not in hard_protect:
                soft_protect.add(anchor.line_no)

            half_window = self.soft_window // 2
            for i in range(1, half_window + 1):
                before = anchor.line_no - i
                after = anchor.line_no + i

                if before >= min_line and before not in hard_protect:
                    soft_protect.add(before)
                if after <= max_line and after not in hard_protect:
                    soft_protect.add(after)

        return soft_protect

    def _find_stack_trace_continuation(self, start_line: int,
                                       line_dict: Dict[int, LogLine],
                                       max_line: int) -> Set[int]:
        stack_lines = set()

        max_scan = 100

        for line_no in range(start_line + 1, min(start_line + max_scan, max_line + 1)):
            if line_no not in line_dict:
                continue

            line = line_dict[line_no]

            if self._is_stack_trace_line(line.text):
                stack_lines.add(line_no)

            elif line.text.strip() == '':
                continue

            elif self._is_new_error_start(line.text):
                break

            else:
                if line.text.startswith('    ') or line.text.startswith('\t'):
                    stack_lines.add(line_no)
                else:
                    break

        return stack_lines

    def _is_stack_trace_line(self, text: str) -> bool:
        for pattern in self.STACK_TRACE_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _is_command_line(self, text: str) -> bool:
        for pattern in self.COMMAND_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _contains_file_location(self, text: str) -> bool:
        for pattern in self.FILE_LOCATION_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _contains_exception(self, text: str) -> bool:
        exception_patterns = [
            r'Exception',
            r'Error:',
            r'error:',
            r'FAILED',
            r'failed',
            r'Traceback',
        ]
        for pattern in exception_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _is_error_related(self, text: str) -> bool:
        error_keywords = [
            'error', 'Error', 'ERROR',
            'fail', 'Fail', 'FAIL',
            'exception', 'Exception',
            'fatal', 'Fatal', 'FATAL',
            'panic', 'Panic',
            'assert', 'Assert',
        ]
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in error_keywords)

    def _is_new_error_start(self, text: str) -> bool:
        new_error_patterns = [
            r'^##\[error\]',
            r'^::error::',
            r'^Traceback',
            r'^Exception in thread',
            r'^Error:',
            r'^FAILED',
        ]
        for pattern in new_error_patterns:
            if re.search(pattern, text):
                return True
        return False
