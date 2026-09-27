from typing import Dict, Any, Callable, Optional
import json

from core.tools import (
    ToolCollection,
    create_classification_tools,
    create_judgment_tools,
    create_sliding_window_extraction_tools
)


class FunctionExecutor:
    def __init__(self, tool_collection: Optional[ToolCollection] = None):
        self._tools = tool_collection or ToolCollection()
        self.context: Dict[str, Any] = {}

    def register_function(self, name: str, func: Callable):
        from core.tools import FunctionTool
        from pydantic import BaseModel, create_model

        params_model = create_model(f'{name}Params')

        tool = FunctionTool(
            func=func,
            params_model=params_model,
            name=name
        )
        self._tools.add(tool)

    def set_context(self, context: Dict[str, Any]):
        self.context = context

    def execute(self, function_name: str, arguments: Dict[str, Any]) -> str:
        return self._tools.execute(function_name, arguments, self.context)

    def get_openai_tools(self):
        return self._tools.to_openai_tools()

    @property
    def tool_names(self):
        return self._tools.names


def create_classification_executor() -> FunctionExecutor:
    tools = create_classification_tools()
    return FunctionExecutor(tools)


def create_judgment_executor() -> FunctionExecutor:
    tools = create_judgment_tools()
    return FunctionExecutor(tools)


def create_sliding_window_extraction_executor() -> FunctionExecutor:
    tools = create_sliding_window_extraction_tools()
    return FunctionExecutor(tools)


def read_log_lines_impl(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    from core.tools import read_log_lines, ReadLogLinesParams
    params = ReadLogLinesParams(**args)
    return json.loads(read_log_lines.execute(args, context))


def read_workflow_content_impl(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    from core.tools import read_workflow_content
    return json.loads(read_workflow_content.execute(args, context))


def search_log_content_impl(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    from core.tools import search_log_content
    return json.loads(search_log_content.execute(args, context))


def expand_timestamp_block_impl(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    from core.tools import expand_timestamp_block
    return json.loads(expand_timestamp_block.execute(args, context))
