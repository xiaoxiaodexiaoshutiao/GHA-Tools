from typing import Dict, Any, List, Tuple
from core.base_agent import BaseAgent
import re


class EarlyTerminationAgent(BaseAgent):
    def __init__(self, agent_type: str, instance_id: int = 1):
        from core.config_loader import config_loader
        early_termination_config = config_loader.get_agent_config('early_termination_agents')

        if agent_type == 'judgment':
            agent_config_key = f'judgment_agent_{instance_id}'
            prompt_key = 'early_termination_agent_judgment'
        elif agent_type == 'final':
            agent_config_key = 'final_agent'
            prompt_key = 'early_termination_agent_final'
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        if agent_config_key not in early_termination_config:
            raise ValueError(f"Early termination agent config not found: {agent_config_key}")

        agent_config = early_termination_config[agent_config_key]

        super().__init__(prompt_key, agent_config)
        self.agent_type = agent_type
        self.instance_id = instance_id
        self.agent_name = f"early_termination_{agent_type}_{instance_id}"


class EarlyTerminationJudgmentAgent(EarlyTerminationAgent):
    def __init__(self, instance_id: int = 1):
        super().__init__('judgment', instance_id)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            log_content = input_data['log_content']
            consecutive_no_error_count = input_data['consecutive_no_error_count']
            current_position = input_data.get('current_position', (0, 0))
            total_blocks = input_data.get('total_blocks', 0)
            processed_blocks = input_data.get('processed_blocks', 0)

            self.log(f"Judging whether to terminate ({consecutive_no_error_count} consecutive no-error blocks)")

            user_prompt = self.format_prompt(
                'user_prompt_template',
                log_content=log_content,
                consecutive_no_error_count=consecutive_no_error_count,
                current_position_start=current_position[0],
                current_position_end=current_position[1],
                total_blocks=total_blocks,
                processed_blocks=processed_blocks
            )

            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]

            def llm_call():
                return self.call_llm(messages)

            def parse_response(resp):
                return self._parse_response(resp)

            success, should_terminate, response = self.retry_on_parse_failure(llm_call, parse_response)

            if not success:
                should_terminate = False
                response = ""

            result_text = "should terminate" if should_terminate else "continue processing"
            self.log(f"Judgment result: {result_text}", "success")

            return {
                "status": "success",
                "should_terminate": should_terminate,
                "raw_response": response
            }

        except Exception as e:
            self.log(f"Judgment failed: {e}", "error")
            return {
                "status": "failed",
                "error": str(e),
                "should_terminate": False
            }

    def _parse_response(self, response: str) -> bool:
        response_upper = response.strip().upper()

        if "SHOULD_TERMINATE" in response_upper:
            return True
        elif "CONTINUE_PROCESSING" in response_upper:
            return False
        else:
            self.log(f"Cannot parse response, defaulting to continue: {response[:100]}", "warning")
            return False
