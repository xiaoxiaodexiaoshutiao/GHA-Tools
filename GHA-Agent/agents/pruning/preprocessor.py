import re
from typing import List, Optional
from .types import LogLine


class Preprocessor:
    ANSI_ESCAPE_PATTERN = re.compile(
        r'''
        \x1b  # ESC character
        (?:   # Non-capturing group
            [@-Z\\-_]  # Single character command
            |
            \[  # CSI sequence start
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
        ''',
        re.VERBOSE
    )

    CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    CARRIAGE_RETURN_PATTERN = re.compile(r'\r(?!\n)')

    def __init__(self, block_id: str = "0"):
        self.block_id = block_id

    def process(self, log_content: str, start_line_no: int = 1) -> List[LogLine]:
        content = self._normalize_newlines(log_content)

        raw_lines = content.split('\n')

        result = []
        for i, raw_text in enumerate(raw_lines):
            line_no = start_line_no + i

            clean_text = self._strip_ansi(raw_text)

            clean_text = self._strip_control_chars(clean_text)

            clean_text = self._handle_carriage_return(clean_text)

            log_line = LogLine(
                id=f"{self.block_id}:{line_no}",
                no=line_no,
                text=clean_text,
                raw_text=raw_text
            )
            result.append(log_line)

        return result

    def _normalize_newlines(self, content: str) -> str:
        content = content.replace('\r\n', '\n')
        content = content.replace('\r', '\n')
        return content

    def _strip_ansi(self, text: str) -> str:
        return self.ANSI_ESCAPE_PATTERN.sub('', text)

    def _strip_control_chars(self, text: str) -> str:
        return self.CONTROL_CHAR_PATTERN.sub('', text)

    def _handle_carriage_return(self, text: str) -> str:
        if '\r' not in text:
            return text

        parts = text.split('\r')

        non_empty_parts = [p for p in parts if p.strip()]
        if non_empty_parts:
            return non_empty_parts[-1]
        return text

    @staticmethod
    def from_lines(lines: List[str], start_line_no: int = 1, block_id: str = "0") -> List[LogLine]:
        preprocessor = Preprocessor(block_id)
        content = '\n'.join(lines)
        return preprocessor.process(content, start_line_no)

    @staticmethod
    def get_text_dict(log_lines: List[LogLine]) -> dict:
        return {line.no: line.text for line in log_lines}

    @staticmethod
    def get_line_nos(log_lines: List[LogLine]) -> List[int]:
        return [line.no for line in log_lines]
