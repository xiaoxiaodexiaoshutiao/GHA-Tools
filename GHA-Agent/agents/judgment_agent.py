from typing import Dict, Any, List
from core.base_agent import BaseAgent
from core.function_executor import create_judgment_executor
from core.function_schemas import get_judgment_tool_collection
import re


class JudgmentAgent(BaseAgent):
    def _calculate_expansion_limits(
        self,
        current_block_index: int,
        all_timestamp_blocks: List,
        error_blocks: set
    ) -> tuple:
        max_down = 0

        for i in range(1, 4):
            next_idx = current_block_index + i
            if next_idx >= len(all_timestamp_blocks):
                max_down = i
                break
            next_block = tuple(all_timestamp_blocks[next_idx])
            if next_block in error_blocks:
                break
            max_down = i

        max_up = 10 - max_down

        self.log(f"Expansion limits: {max_down} unverified blocks before error in next 3, can expand up {max_up} blocks, down {max_down} blocks")

        return max_up, max_down

    def __init__(self, instance_id: int = 1):
        agent_config_key = f'agent_{instance_id}'

        from core.config_loader import config_loader
        judgment_agents_config = config_loader.get_agent_config('judgment_agents')

        if agent_config_key not in judgment_agents_config:
            raise ValueError(f"Judgment agent config not found: {agent_config_key}")

        agent_config = judgment_agents_config[agent_config_key]

        super().__init__('judgment_agent', agent_config)
        self.instance_id = instance_id
        self.agent_name = f"judgment_agent_{instance_id}"

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            log_content = input_data['log_content']
            block_start = input_data['block_start']
            block_end = input_data['block_end']
            current_block_index = input_data.get('current_block_index')
            all_timestamp_blocks = input_data.get('all_timestamp_blocks', [])
            log_lines = input_data.get('log_lines', [])
            verified_blocks = input_data.get('verified_blocks', set())
            error_blocks = input_data.get('error_blocks', set())
            log_metadata = input_data.get('log_metadata', None)
            block_index = input_data.get('block_index', 0)

            self.log(f"Judging timestamp block [{block_start}-{block_end}]")

            max_expand_up = 10
            max_expand_down = 0
            if current_block_index is not None and all_timestamp_blocks:
                max_expand_up, max_expand_down = self._calculate_expansion_limits(
                    current_block_index, all_timestamp_blocks, error_blocks
                )

            user_prompt = self.format_prompt(
                'user_prompt_template',
                log_content=log_content,
                block_start=block_start,
                block_end=block_end,
                max_expand_up=max_expand_up,
                max_expand_down=max_expand_down
            )

            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]

            if current_block_index is not None and all_timestamp_blocks and log_lines:
                executor = create_judgment_executor()

                executor.set_context({
                    'current_block_index': current_block_index,
                    'all_timestamp_blocks': all_timestamp_blocks,
                    'log_lines': log_lines,
                    'expanded_blocks_up': 0,
                    'expanded_blocks_down': 0,
                    'verified_blocks': verified_blocks,
                    'max_expand_up': max_expand_up,
                    'max_expand_down': max_expand_down
                })

                tool_collection = get_judgment_tool_collection()

                def llm_call():
                    return self.call_llm_with_tools(
                        messages=messages,
                        tools=tool_collection,
                        function_executor=executor,
                        log_metadata=log_metadata,
                        call_index=block_index,
                        extra_info={"block_range": [block_start, block_end]},
                        max_iterations=12
                    )

                def parse_response(resp):
                    return self._parse_response(resp)

                success, contains_error, response = self.retry_on_parse_failure(llm_call, parse_response)

                if not success:
                    self.log(f"After {3} retries, unable to parse response, defaulting to no error", "warning")
                    contains_error = False
                    response = response or ""

                expanded_blocks_up = executor.context.get('expanded_blocks_up', 0)
                expanded_blocks_down = executor.context.get('expanded_blocks_down', 0)

                final_start = block_start
                final_end = block_end

                if expanded_blocks_up > 0:
                    up_block_index = current_block_index - expanded_blocks_up
                    if up_block_index >= 0:
                        final_start = all_timestamp_blocks[up_block_index][0]

                if expanded_blocks_down > 0:
                    down_block_index = current_block_index + expanded_blocks_down
                    if down_block_index < len(all_timestamp_blocks):
                        final_end = all_timestamp_blocks[down_block_index][1]

                final_block_range = (final_start, final_end)

            else:
                def llm_call():
                    return self.call_llm(
                        messages,
                        log_metadata=log_metadata,
                        call_index=block_index,
                        extra_info={"block_range": [block_start, block_end]}
                    )

                def parse_response(resp):
                    return self._parse_response(resp)

                success, contains_error, response = self.retry_on_parse_failure(llm_call, parse_response)

                if not success:
                    self.log(f"After {3} retries, unable to parse response, defaulting to no error", "warning")
                    contains_error = False
                    response = response or ""

                expanded_blocks_up = 0
                expanded_blocks_down = 0
                final_block_range = (block_start, block_end)

            result_text = "contains error" if contains_error else "no error"
            self.log(f"Judgment result: {result_text}", "success")

            if expanded_blocks_up > 0 or expanded_blocks_down > 0:
                self.log(f"Expansion info: up {expanded_blocks_up} blocks, down {expanded_blocks_down} blocks")
                self.log(f"Final range: [{final_block_range[0]}-{final_block_range[1]}]")

            return {
                "status": "success",
                "contains_error": contains_error,
                "raw_response": response,
                "expanded_blocks_up": expanded_blocks_up,
                "expanded_blocks_down": expanded_blocks_down,
                "final_block_range": final_block_range
            }

        except Exception as e:
            self.log(f"Judgment failed: {e}", "error")
            import traceback
            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e),
                "contains_error": False,
                "expanded_blocks_up": 0,
                "expanded_blocks_down": 0,
                "final_block_range": (input_data.get('block_start', 0), input_data.get('block_end', 0))
            }

    def _parse_response(self, response: str) -> bool:
        response_upper = response.strip().upper()

        if "CONTAINS_ERROR" in response_upper:
            return True
        elif "NO_ERROR" in response_upper:
            return False
        else:
            raise ValueError(f"Response does not contain CONTAINS_ERROR or NO_ERROR keyword")
