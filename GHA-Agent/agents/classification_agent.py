from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent
from core.function_executor import create_classification_executor
from core.function_schemas import get_classification_functions, get_classification_tool_collection
import re


class ClassificationAgent(BaseAgent):
    def __init__(self):
        super().__init__('classification_agent')
        self.initial_read_lines = self.config.get('initial_read_lines', 10)
        self.max_additional_reads = self.config.get('max_additional_reads', 3)
        self.enable_function_calling = self.config.get('enable_function_calling', True)
        self.max_tool_iterations = self.config.get('max_tool_iterations', 10)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            workflow_content = input_data.get('workflow_content', '')
            log_lines = input_data['log_lines']
            log_range = input_data['log_range']
            log_metadata = input_data.get('log_metadata', None)

            total_lines = len(log_lines)
            start_line, end_line = log_range

            self.log(f"Starting classification, log range: {start_line}-{end_line}, total lines: {total_lines}")

            if self.enable_function_calling:
                try:
                    return self._process_with_function_calling(
                        workflow_content, log_lines, total_lines, log_metadata
                    )
                except Exception as e:
                    self.log(f"Function calling failed, falling back to traditional mode: {e}", "warning")
            return self._process_traditional(
                workflow_content, log_lines, total_lines, start_line, end_line, log_metadata
            )

        except Exception as e:
            self.log(f"Classification failed: {e}", "error")
            return {
                "status": "failed",
                "error": str(e),
                "method": 1
            }

    def _process_with_function_calling(self, workflow_content: str,
                                      log_lines: List[str], total_lines: int,
                                      log_metadata: Dict[str, str] = None) -> Dict[str, Any]:
        self.log("Using Function Calling mode")

        executor = create_classification_executor()

        context = {
            "log_lines": log_lines,
            "workflow_content": workflow_content,
            "current_head_lines": 0,
            "current_tail_lines": self.initial_read_lines
        }
        executor.set_context(context)

        initial_log_content = self._extract_log_snippet(
            log_lines,
            head_lines=0,
            tail_lines=self.initial_read_lines
        )

        user_prompt = self.format_prompt(
            'user_prompt_template',
            workflow_content=workflow_content[:500] + "..." if len(workflow_content) > 500 else workflow_content,
            total_lines=total_lines,
            tail_lines=self.initial_read_lines,
            log_content=initial_log_content
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]

        tool_collection = get_classification_tool_collection()

        def llm_call():
            response = self.call_llm_with_tools(
                messages=messages,
                tools=tool_collection,
                function_executor=executor,
                log_metadata=log_metadata,
                call_index=0,
                extra_info={"mode": "function_calling"},
                max_iterations=self.max_tool_iterations
            )
            self.log(f"LLM response length: {len(response)}", "debug")
            self.log(f"LLM response first 500 chars: {response[:500]}", "debug")
            return response

        def parse_response(response):
            method = self._extract_method_from_response(response)
            if method is None:
                self.log(f"Cannot extract method from response", "warning")
                self.log(f"Full response: {response}", "debug")
                raise ValueError(f"Cannot extract method (response length: {len(response)} chars)")
            return method

        success, method, response = self.retry_on_parse_failure(llm_call, parse_response)

        if success:
            method_name = 'Timestamp-based' if method == 1 else 'Sliding Window'
            self.log(f"Selected parsing method: {method} ({method_name})", "success")
            return {
                "status": "success",
                "method": method,
                "reason": response
            }
        else:
            self.log(f"Parse failed, using default method 1 (Timestamp-based)", "warning")
            self.log(f"Final response content: {response[:1000] if response else 'None'}", "warning")
            return {
                "status": "success",
                "method": 1,
                "reason": "Default selection (parse failed)"
            }

    def _process_traditional(self, workflow_content: str, log_lines: List[str],
                           total_lines: int, start_line: int, end_line: int,
                           log_metadata: Dict[str, str] = None) -> Dict[str, Any]:
        self.log("Using traditional text parsing mode")

        read_count = 0
        current_tail_lines = self.initial_read_lines
        current_head_lines = 0

        while read_count < self.max_additional_reads:
            log_content = self._extract_log_snippet(
                log_lines,
                head_lines=current_head_lines,
                tail_lines=current_tail_lines
            )

            current_read_info = f"head {current_head_lines} lines + tail {current_tail_lines} lines"

            user_prompt = self.format_prompt(
                'user_prompt_template',
                workflow_content=workflow_content,
                total_lines=total_lines,
                tail_lines=current_tail_lines,
                current_read_info=current_read_info,
                log_content=log_content
            )

            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]

            def llm_call():
                return self.call_llm(
                    messages,
                    log_metadata=log_metadata,
                    call_index=read_count,
                    extra_info={"mode": "traditional", "read_count": read_count}
                )

            def parse_response(response):
                result = self._parse_response(response)
                if result['type'] == 'unknown':
                    raise ValueError(f"Cannot parse response type: {response[:200]}")
                return result

            success, parse_result, response = self.retry_on_parse_failure(llm_call, parse_response)

            if not success:
                self.log(f"Parse failed, skipping current read", "warning")
                break

            if parse_result['type'] == 'need_more':
                position = parse_result['position']
                lines_needed = parse_result['lines']

                if position == 'HEAD':
                    current_head_lines += lines_needed
                else:
                    current_tail_lines += lines_needed

                read_count += 1
                self.log(f"Need more lines, current read: head {current_head_lines} lines + tail {current_tail_lines} lines")

            elif parse_result['type'] == 'method':
                method = parse_result['method']
                self.log(f"Selected parsing method: {method} ({'Timestamp-based' if method == 1 else 'Sliding Window'})", "success")

                return {
                    "status": "success",
                    "method": method,
                    "reason": response
                }

        self.log("Exceeded max read count, defaulting to timestamp-based parsing", "warning")
        return {
            "status": "success",
            "method": 1,
            "reason": "Default selection (exceeded max read count)"
        }

    def _extract_log_snippet(self, log_lines: List[str], head_lines: int = 0,
                            tail_lines: int = 10) -> str:
        parts = []

        if head_lines > 0:
            head_content = "\n".join(log_lines[:head_lines])
            parts.append(f"=== Head {head_lines} lines ===\n{head_content}")

        if tail_lines > 0:
            tail_content = "\n".join(log_lines[-tail_lines:])
            parts.append(f"=== Tail {tail_lines} lines ===\n{tail_content}")

        return "\n\n".join(parts)

    def _extract_method_from_response(self, response: str) -> Optional[int]:
        if not response:
            return None

        patterns = [
            r'METHOD\s*:\s*([12])',
            r'METHOD\s+([12])',
            r'method\s*:\s*([12])',
            r'Method\s*:\s*([12])',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                method = int(match.group(1))
                self.log(f"Successfully extracted method: {method} (pattern: {pattern})", "debug")
                return method

        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            line = line.strip()
            if line in ['1', '2']:
                method = int(line)
                self.log(f"Extracted method from last lines: {method}", "debug")
                return method

        return None

    def _parse_response(self, response: str) -> Dict[str, Any]:
        response = response.strip()

        need_more_pattern = r'NEED_MORE_LINES\|(HEAD|TAIL)\|(\d+)'
        match = re.search(need_more_pattern, response)
        if match:
            position = match.group(1)
            lines = int(match.group(2))
            return {
                'type': 'need_more',
                'position': position,
                'lines': lines
            }

        method = self._extract_method_from_response(response)
        if method:
            return {
                'type': 'method',
                'method': method
            }

        return {'type': 'unknown'}
