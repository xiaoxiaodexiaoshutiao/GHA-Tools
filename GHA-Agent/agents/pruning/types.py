from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional
from enum import Enum


class AnchorPriority(Enum):
    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3


class AnchorType(Enum):
    GHA_ERROR = "gha_error"
    GHA_WARNING = "gha_warning"
    GHA_EXIT_CODE = "gha_exit_code"

    COMPILER_ERROR = "compiler_error"
    LINKER_ERROR = "linker_error"
    BUILD_ERROR = "build_error"

    TRACEBACK = "traceback"
    EXCEPTION = "exception"
    PANIC = "panic"

    TEST_FAILURE = "test_failure"
    ASSERTION_ERROR = "assertion_error"

    GENERIC_ERROR = "generic_error"
    GENERIC_FAILED = "generic_failed"

    LLM_IDENTIFIED = "llm_identified"


@dataclass
class LogLine:
    id: str
    no: int
    text: str
    raw_text: str = ""

    def __hash__(self):
        return hash(self.no)

    def __eq__(self, other):
        if isinstance(other, LogLine):
            return self.no == other.no
        return False


@dataclass
class Anchor:
    line_no: int
    anchor_type: AnchorType
    priority: AnchorPriority
    pattern_matched: str = ""
    text: str = ""

    def __hash__(self):
        return hash(self.line_no)

    def __eq__(self, other):
        if isinstance(other, Anchor):
            return self.line_no == other.line_no
        return False


@dataclass
class AnchorResult:
    anchors_hard: List[Anchor] = field(default_factory=list)
    anchors_soft: List[Anchor] = field(default_factory=list)

    @property
    def hard_line_nos(self) -> Set[int]:
        return {a.line_no for a in self.anchors_hard}

    @property
    def soft_line_nos(self) -> Set[int]:
        return {a.line_no for a in self.anchors_soft}

    @property
    def all_line_nos(self) -> Set[int]:
        return self.hard_line_nos | self.soft_line_nos

    def is_empty(self) -> bool:
        return len(self.anchors_hard) == 0 and len(self.anchors_soft) == 0


@dataclass
class ProtectResult:
    hard_protect_set: Set[int] = field(default_factory=set)
    soft_protect_set: Set[int] = field(default_factory=set)
    candidate_set: Set[int] = field(default_factory=set)

    def is_hard_protected(self, line_no: int) -> bool:
        return line_no in self.hard_protect_set

    def is_soft_protected(self, line_no: int) -> bool:
        return line_no in self.soft_protect_set

    def is_candidate(self, line_no: int) -> bool:
        return line_no in self.candidate_set


@dataclass
class DeleteProposal:
    line_no: int
    text: str = ""
    reason: str = ""
    source: str = "unknown"

    def __hash__(self):
        return hash(self.line_no)

    def __eq__(self, other):
        if isinstance(other, DeleteProposal):
            return self.line_no == other.line_no
        return False


@dataclass
class RCAResult:
    failed_step: str = ""
    failed_command: str = ""
    primary_error_message: str = ""
    exit_code: Optional[int] = None
    exception_type: str = ""
    file_path: str = ""
    line_number: Optional[int] = None

    def core_fields_match(self, other: 'RCAResult') -> bool:
        if self.primary_error_message and other.primary_error_message:
            if self.primary_error_message != other.primary_error_message:
                return False
        elif self.primary_error_message and not other.primary_error_message:
            return False

        if self.exit_code is not None and other.exit_code is not None:
            if self.exit_code != other.exit_code:
                return False
        elif self.exit_code is not None and other.exit_code is None:
            return False

        return True


@dataclass
class PruningResult:
    delete_lines: List[int] = field(default_factory=list)
    kept_lines: List[int] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def delete_count(self) -> int:
        return len(self.delete_lines)

    @property
    def kept_count(self) -> int:
        return len(self.kept_lines)

    def to_ranges(self) -> List[tuple]:
        if not self.delete_lines:
            return []

        sorted_lines = sorted(self.delete_lines)
        ranges = []
        start = sorted_lines[0]
        end = sorted_lines[0]

        for line_no in sorted_lines[1:]:
            if line_no == end + 1:
                end = line_no
            else:
                ranges.append((start, end))
                start = line_no
                end = line_no

        ranges.append((start, end))
        return ranges
