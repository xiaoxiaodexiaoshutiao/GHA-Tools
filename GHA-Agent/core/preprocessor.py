import re


class LogPreprocessor:
    ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')

    TIMESTAMP_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?')

    @classmethod
    def remove_ansi_codes(cls, text: str) -> str:
        return cls.ANSI_PATTERN.sub('', text)

    @classmethod
    def remove_timestamp(cls, text: str) -> str:
        return cls.TIMESTAMP_PATTERN.sub('', text)

    @classmethod
    def preprocess_line(cls, line: str) -> str:
        line = cls.remove_ansi_codes(line)
        line = cls.remove_timestamp(line)
        return line.rstrip()
