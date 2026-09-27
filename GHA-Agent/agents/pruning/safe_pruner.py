import re
from typing import List, Set, Dict, Tuple
from .types import LogLine, ProtectResult, DeleteProposal


class SafePruner:
    SEPARATOR_PATTERNS = [
        re.compile(r'^[-=_*#~]{10,}$'),
        re.compile(r'^[\s]*[-=_*#~]+[\s]*$'),
        re.compile(r'^\+[-+]+\+$'),
    ]

    PROGRESS_PATTERNS = [
        re.compile(r'\[\s*[=#>\s]+\]\s*\d+%'),
        re.compile(r'\d+%\s*\|[█▓▒░\s]+\|'),
        re.compile(r'^\s*\d+%\s*$'),
        re.compile(r'Downloading[.]+\s*\d+%'),
        re.compile(r'Uploading[.]+\s*\d+%'),
        re.compile(r'Progress:\s*\d+%'),
        re.compile(r'\(\d+/\d+\)'),
        re.compile(r'^\s*[.]+\s*$'),
    ]

    TELEMETRY_PATTERNS = [
        re.compile(r'^\s*\*+\s*$'),
        re.compile(r'Welcome to'),
        re.compile(r'Thank you for using'),
        re.compile(r'Please consider'),
        re.compile(r'Donate'),
        re.compile(r'Sponsor'),
        re.compile(r'npm notice'),
        re.compile(r'^\s*│\s*│\s*$'),
    ]

    TIMESTAMP_PATTERN = re.compile(
        r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}'
    )

    def __init__(self,
                 min_distance_from_anchor: int = 20,
                 max_consecutive_duplicates: int = 3):
        self.min_distance_from_anchor = min_distance_from_anchor
        self.max_consecutive_duplicates = max_consecutive_duplicates

    def identify(self, log_lines: List[LogLine],
                protect: ProtectResult) -> Set[int]:
        predelete_ids = set()

        line_dict = {line.no: line for line in log_lines}

        empty_separator_lines = self._find_empty_and_separators(log_lines, protect)
        predelete_ids.update(empty_separator_lines)

        progress_lines = self._find_progress_lines(log_lines, protect)
        predelete_ids.update(progress_lines)

        duplicate_lines = self._find_duplicate_lines(log_lines, protect)
        predelete_ids.update(duplicate_lines)

        telemetry_lines = self._find_telemetry_lines(log_lines, protect)
        predelete_ids.update(telemetry_lines)

        predelete_ids = predelete_ids - protect.hard_protect_set

        return predelete_ids

    def _find_empty_and_separators(self, log_lines: List[LogLine],
                                   protect: ProtectResult) -> Set[int]:
        result = set()

        for line in log_lines:
            if line.no in protect.hard_protect_set:
                continue

            is_empty_or_sep = False

            if line.text.strip() == '':
                is_empty_or_sep = True

            if not is_empty_or_sep:
                for pattern in self.SEPARATOR_PATTERNS:
                    if pattern.match(line.text.strip()):
                        is_empty_or_sep = True
                        break

            if is_empty_or_sep:
                if not self._is_near_hard_protect(line.no, protect, distance=2):
                    result.add(line.no)

        return result

    def _find_progress_lines(self, log_lines: List[LogLine],
                            protect: ProtectResult) -> Set[int]:
        result = set()

        for line in log_lines:
            if line.no in protect.hard_protect_set:
                continue

            is_progress = False
            for pattern in self.PROGRESS_PATTERNS:
                if pattern.search(line.text):
                    is_progress = True
                    break

            if is_progress:
                if not self._is_near_hard_protect(
                    line.no, protect, distance=self.min_distance_from_anchor
                ):
                    result.add(line.no)

        return result

    def _find_duplicate_lines(self, log_lines: List[LogLine],
                             protect: ProtectResult) -> Set[int]:
        result = set()

        if len(log_lines) < 3:
            return result

        def normalize(text: str) -> str:
            return text.strip()

        i = 0
        while i < len(log_lines):
            line = log_lines[i]

            if line.no in protect.hard_protect_set:
                i += 1
                continue

            normalized = normalize(line.text)

            if not normalized:
                i += 1
                continue

            duplicate_indices = [i]
            j = i + 1
            while j < len(log_lines):
                next_line = log_lines[j]

                if next_line.no in protect.hard_protect_set:
                    break
                if normalize(next_line.text) == normalized:
                    duplicate_indices.append(j)
                    j += 1
                else:
                    break

            if len(duplicate_indices) >= self.max_consecutive_duplicates:
                for idx in duplicate_indices[1:-1]:
                    result.add(log_lines[idx].no)
                i = j
            else:
                i += 1

        return result

    def _find_telemetry_lines(self, log_lines: List[LogLine],
                             protect: ProtectResult) -> Set[int]:
        result = set()

        for line in log_lines:
            if line.no in protect.hard_protect_set:
                continue

            is_telemetry = False
            for pattern in self.TELEMETRY_PATTERNS:
                if pattern.search(line.text):
                    is_telemetry = True
                    break

            if is_telemetry:
                if not self._is_near_hard_protect(
                    line.no, protect, distance=self.min_distance_from_anchor
                ):
                    result.add(line.no)

        return result

    def _is_near_hard_protect(self, line_no: int,
                              protect: ProtectResult,
                              distance: int) -> bool:
        for protected_line in protect.hard_protect_set:
            if abs(line_no - protected_line) <= distance:
                return True
        return False

    def create_proposals(self, predelete_ids: Set[int],
                        line_dict: Dict[int, LogLine]) -> List[DeleteProposal]:
        proposals = []
        for line_no in sorted(predelete_ids):
            line = line_dict.get(line_no)
            if line:
                proposals.append(DeleteProposal(
                    line_no=line_no,
                    text=line.text,
                    reason="safe_prune",
                    source="predelete"
                ))
        return proposals
