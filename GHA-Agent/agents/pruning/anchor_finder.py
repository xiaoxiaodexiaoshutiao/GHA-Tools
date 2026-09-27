import re
from typing import List, Dict, Tuple, Optional, Set
from .types import LogLine, Anchor, AnchorType, AnchorPriority, AnchorResult


class AnchorPatterns:
    GHA_ERROR_PATTERNS = [
        (r'^##\[error\]', AnchorType.GHA_ERROR, "GHA ##[error]"),
        (r'^::error::', AnchorType.GHA_ERROR, "GHA ::error::"),
        (r'::error\s+file=.+,line=\d+', AnchorType.GHA_ERROR, "GHA error with location"),
        (r'Process completed with exit code [1-9]', AnchorType.GHA_EXIT_CODE, "Process exit code"),
        (r'The process .+ failed with exit code', AnchorType.GHA_EXIT_CODE, "Process failed"),
    ]

    C_CPP_ERROR_PATTERNS = [
        (r':\d+:\d+: error:', AnchorType.COMPILER_ERROR, "GCC/Clang error"),
        (r'fatal error:', AnchorType.COMPILER_ERROR, "Fatal error"),
        (r'undefined reference to', AnchorType.LINKER_ERROR, "Undefined reference"),
        (r'multiple definition of', AnchorType.LINKER_ERROR, "Multiple definition"),
        (r'^ld: ', AnchorType.LINKER_ERROR, "Linker error"),
        (r'collect2: error:', AnchorType.LINKER_ERROR, "Collect2 error"),
        (r'make(\[\d+\])?: \*\*\*', AnchorType.BUILD_ERROR, "Make error"),
        (r'^CMake Error', AnchorType.BUILD_ERROR, "CMake error"),
        (r'Segmentation fault', AnchorType.EXCEPTION, "Segfault"),
        (r'==\d+==ERROR: AddressSanitizer', AnchorType.EXCEPTION, "ASAN error"),
    ]

    JAVA_ERROR_PATTERNS = [
        (r'\.java:\d+: error:', AnchorType.COMPILER_ERROR, "Java compile error"),
        (r'Exception in thread ".+"', AnchorType.EXCEPTION, "Java exception"),
        (r'java\.lang\.\w+Exception', AnchorType.EXCEPTION, "Java runtime exception"),
        (r'^\[ERROR\]', AnchorType.BUILD_ERROR, "Maven error"),
        (r'BUILD FAILURE', AnchorType.BUILD_ERROR, "Maven build failure"),
        (r'^FAILURE:', AnchorType.BUILD_ERROR, "Gradle failure"),
        (r'BUILD FAILED', AnchorType.BUILD_ERROR, "Gradle build failed"),
        (r'> Task :.+ FAILED', AnchorType.BUILD_ERROR, "Gradle task failed"),
    ]

    JS_ERROR_PATTERNS = [
        (r'^npm ERR!', AnchorType.BUILD_ERROR, "npm error"),
        (r'SyntaxError:', AnchorType.EXCEPTION, "JS SyntaxError"),
        (r'TypeError:', AnchorType.EXCEPTION, "JS TypeError"),
        (r'ReferenceError:', AnchorType.EXCEPTION, "JS ReferenceError"),
        (r'Cannot find module', AnchorType.EXCEPTION, "Module not found"),
        (r'Error: E[A-Z]+', AnchorType.EXCEPTION, "Node error code"),
        (r'error TS\d+:', AnchorType.COMPILER_ERROR, "TypeScript error"),
        (r'^FAIL\s', AnchorType.TEST_FAILURE, "Jest FAIL"),
        (r'^ERROR in', AnchorType.BUILD_ERROR, "Webpack error"),
        (r'Module not found:', AnchorType.BUILD_ERROR, "Module not found"),
    ]

    PYTHON_ERROR_PATTERNS = [
        (r'^Traceback \(most recent call last\):', AnchorType.TRACEBACK, "Python traceback"),
        (r'SyntaxError:', AnchorType.EXCEPTION, "Python SyntaxError"),
        (r'IndentationError:', AnchorType.EXCEPTION, "Python IndentationError"),
        (r'ImportError:', AnchorType.EXCEPTION, "Python ImportError"),
        (r'ModuleNotFoundError:', AnchorType.EXCEPTION, "Python ModuleNotFoundError"),
        (r'AttributeError:', AnchorType.EXCEPTION, "Python AttributeError"),
        (r'TypeError:', AnchorType.EXCEPTION, "Python TypeError"),
        (r'ValueError:', AnchorType.EXCEPTION, "Python ValueError"),
        (r'KeyError:', AnchorType.EXCEPTION, "Python KeyError"),
        (r'NameError:', AnchorType.EXCEPTION, "Python NameError"),
        (r'IndexError:', AnchorType.EXCEPTION, "Python IndexError"),
        (r'FileNotFoundError:', AnchorType.EXCEPTION, "Python FileNotFoundError"),
        (r'AssertionError', AnchorType.ASSERTION_ERROR, "Python AssertionError"),
        (r'^ERROR: ', AnchorType.BUILD_ERROR, "pip error"),
        (r'^FAILED\s', AnchorType.TEST_FAILURE, "pytest FAILED"),
        (r'=+ FAILURES =+', AnchorType.TEST_FAILURE, "pytest failures"),
    ]

    CSHARP_ERROR_PATTERNS = [
        (r'error CS\d{4}:', AnchorType.COMPILER_ERROR, "C# compile error"),
        (r'MSBUILD : error', AnchorType.BUILD_ERROR, "MSBuild error"),
        (r'Build FAILED', AnchorType.BUILD_ERROR, "Build failed"),
        (r'Unhandled exception\.', AnchorType.EXCEPTION, ".NET unhandled exception"),
        (r'System\.[A-Za-z]+Exception', AnchorType.EXCEPTION, ".NET exception"),
        (r'error NU\d{4}', AnchorType.BUILD_ERROR, "NuGet error"),
    ]

    GO_ERROR_PATTERNS = [
        (r'panic:', AnchorType.PANIC, "Go panic"),
        (r'fatal error:', AnchorType.EXCEPTION, "Go fatal error"),
    ]

    RUST_ERROR_PATTERNS = [
        (r'error\[E\d+\]:', AnchorType.COMPILER_ERROR, "Rust compile error"),
        (r'panicked at', AnchorType.PANIC, "Rust panic"),
    ]

    PHP_ERROR_PATTERNS = [
        (r'Fatal error:', AnchorType.EXCEPTION, "PHP Fatal error"),
        (r'PHP Fatal error:', AnchorType.EXCEPTION, "PHP Fatal error"),
        (r'Parse error:', AnchorType.EXCEPTION, "PHP Parse error"),
        (r'PHP Parse error:', AnchorType.EXCEPTION, "PHP Parse error"),
    ]

    R_ERROR_PATTERNS = [
        (r'^Error in .+:', AnchorType.EXCEPTION, "R Error"),
        (r'^Error:', AnchorType.EXCEPTION, "R Error"),
        (r'Execution halted', AnchorType.EXCEPTION, "R Execution halted"),
        (r'ERROR: installation of package', AnchorType.BUILD_ERROR, "R package install error"),
    ]

    TEST_FAILURE_PATTERNS = [
        (r'\bFAILED\b', AnchorType.TEST_FAILURE, "Test FAILED"),
        (r'FAILURES!!!', AnchorType.TEST_FAILURE, "Test failures"),
        (r'tests? failed', AnchorType.TEST_FAILURE, "Tests failed"),
        (r'Failing tests:', AnchorType.TEST_FAILURE, "Failing tests"),
        (r'AssertionError', AnchorType.ASSERTION_ERROR, "AssertionError"),
    ]

    GENERIC_ERROR_PATTERNS = [
        (r'exit code [1-9]\d*', AnchorType.GHA_EXIT_CODE, "Exit code"),
        (r'returned [1-9]\d*', AnchorType.GHA_EXIT_CODE, "Returned non-zero"),
        (r'[Pp]ermission denied', AnchorType.GENERIC_ERROR, "Permission denied"),
        (r'[Cc]onnection refused', AnchorType.GENERIC_ERROR, "Connection refused"),
        (r'No such file or directory', AnchorType.GENERIC_ERROR, "File not found"),
    ]

    WARNING_PATTERNS = [
        (r'^##\[warning\]', AnchorType.GHA_WARNING, "GHA ##[warning]"),
        (r'^::warning::', AnchorType.GHA_WARNING, "GHA ::warning::"),
        (r':\d+:\d+: warning:', AnchorType.GHA_WARNING, "Compiler warning"),
        (r'^npm WARN', AnchorType.GHA_WARNING, "npm warning"),
        (r'^Warning:', AnchorType.GHA_WARNING, "Warning"),
        (r'warning CS\d{4}:', AnchorType.GHA_WARNING, "C# warning"),
        (r'^CMake Warning', AnchorType.GHA_WARNING, "CMake warning"),
    ]

    @classmethod
    def get_p0_patterns(cls) -> List[Tuple[str, AnchorType, str]]:
        return (
            cls.GHA_ERROR_PATTERNS +
            cls.C_CPP_ERROR_PATTERNS +
            cls.JAVA_ERROR_PATTERNS +
            cls.JS_ERROR_PATTERNS +
            cls.PYTHON_ERROR_PATTERNS +
            cls.CSHARP_ERROR_PATTERNS +
            cls.GO_ERROR_PATTERNS +
            cls.RUST_ERROR_PATTERNS +
            cls.PHP_ERROR_PATTERNS +
            cls.R_ERROR_PATTERNS +
            cls.TEST_FAILURE_PATTERNS +
            cls.GENERIC_ERROR_PATTERNS
        )

    @classmethod
    def get_p1_patterns(cls) -> List[Tuple[str, AnchorType, str]]:
        return cls.WARNING_PATTERNS


class AnchorFinder:
    def __init__(self, min_anchors_threshold: int = 3, llm_client=None):
        self.min_anchors_threshold = min_anchors_threshold
        self.llm_client = llm_client

        self._compile_patterns()

    def _compile_patterns(self):
        self.p0_patterns = []
        for pattern, anchor_type, desc in AnchorPatterns.get_p0_patterns():
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self.p0_patterns.append((compiled, anchor_type, desc))
            except re.error as e:
                print(f"Warning: Failed to compile pattern '{pattern}': {e}")

        self.p1_patterns = []
        for pattern, anchor_type, desc in AnchorPatterns.get_p1_patterns():
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self.p1_patterns.append((compiled, anchor_type, desc))
            except re.error as e:
                print(f"Warning: Failed to compile pattern '{pattern}': {e}")

    def find(self, log_lines: List[LogLine]) -> AnchorResult:
        result = AnchorResult()

        for line in log_lines:
            anchor = self._match_p0(line)
            if anchor:
                result.anchors_hard.append(anchor)
                continue

            anchor = self._match_p1(line)
            if anchor:
                result.anchors_soft.append(anchor)

        if len(result.anchors_hard) < self.min_anchors_threshold and self.llm_client:
            llm_anchors = self._llm_fallback(log_lines, result)
            result.anchors_hard.extend(llm_anchors)

        return result

    def _match_p0(self, line: LogLine) -> Optional[Anchor]:
        for pattern, anchor_type, desc in self.p0_patterns:
            if pattern.search(line.text):
                return Anchor(
                    line_no=line.no,
                    anchor_type=anchor_type,
                    priority=AnchorPriority.P0,
                    pattern_matched=desc,
                    text=line.text
                )
        return None

    def _match_p1(self, line: LogLine) -> Optional[Anchor]:
        for pattern, anchor_type, desc in self.p1_patterns:
            if pattern.search(line.text):
                return Anchor(
                    line_no=line.no,
                    anchor_type=anchor_type,
                    priority=AnchorPriority.P1,
                    pattern_matched=desc,
                    text=line.text
                )
        return None

    def _llm_fallback(self, log_lines: List[LogLine],
                      existing_result: AnchorResult) -> List[Anchor]:
        if not self.llm_client:
            return []

        existing_line_nos = existing_result.all_line_nos

        lines_text = []
        for line in log_lines:
            if line.no not in existing_line_nos:
                lines_text.append(f"[no={line.no}] {line.text}")

        if not lines_text:
            return []

        if len(lines_text) > 200:
            lines_text = lines_text[:200]

        prompt = f"""Please analyze the following log lines and identify line numbers most likely to be failure root cause evidence.
Only output a list of line numbers, no more than 20, in format: [line_no1, line_no2, ...]

Log lines:
{chr(10).join(lines_text)}

Please only output a JSON format line number array, e.g.: [101, 105, 120]
"""

        try:
            messages = [
                {"role": "system", "content": "You are a professional log analysis expert, skilled at identifying error root causes."},
                {"role": "user", "content": prompt}
            ]

            response = self.llm_client.call(messages)

            import json

            match = re.search(r'\[[\d,\s]+\]', response)
            if match:
                line_nos = json.loads(match.group())

                llm_anchors = []
                line_dict = {line.no: line for line in log_lines}

                for line_no in line_nos[:20]:
                    if line_no in line_dict and line_no not in existing_line_nos:
                        llm_anchors.append(Anchor(
                            line_no=line_no,
                            anchor_type=AnchorType.LLM_IDENTIFIED,
                            priority=AnchorPriority.P0,
                            pattern_matched="LLM identified",
                            text=line_dict[line_no].text
                        ))

                return llm_anchors
        except Exception as e:
            print(f"LLM fallback failed: {e}")

        return []

    @staticmethod
    def is_stack_trace_line(text: str) -> bool:
        stack_patterns = [
            r'^\s+at\s+',
            r'^File ".+", line \d+',
            r'^\s+at [a-zA-Z0-9$.]+\(.+:\d+\)',
            r'^\s+at .+ in .+:line \d+',
            r'^#\d+\s+.+\(\d+\):',
            r'^Caused by:',
        ]

        for pattern in stack_patterns:
            if re.search(pattern, text):
                return True
        return False

    @staticmethod
    def is_error_location_line(text: str) -> bool:
        location_patterns = [
            r'[a-zA-Z0-9_/\\.-]+:\d+:\d+',
            r'File ".+", line \d+',
            r'in .+:line \d+',
        ]

        for pattern in location_patterns:
            if re.search(pattern, text):
                return True
        return False
