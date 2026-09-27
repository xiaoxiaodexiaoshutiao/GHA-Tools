from typing import Dict, Any, List


from core.tools import (
    create_classification_tools,
    create_judgment_tools,
    create_sliding_window_extraction_tools,
    ToolCollection,
)


def get_classification_functions() -> List[Dict[str, Any]]:
    tools = create_classification_tools()
    return tools.to_openai_tools()


def get_judgment_functions() -> List[Dict[str, Any]]:
    tools = create_judgment_tools()
    return tools.to_openai_tools()


def get_classification_tool_collection() -> ToolCollection:
    return create_classification_tools()


def get_judgment_tool_collection() -> ToolCollection:
    return create_judgment_tools()


def get_sliding_window_extraction_tool_collection() -> ToolCollection:
    return create_sliding_window_extraction_tools()


def get_sliding_window_extraction_functions() -> List[Dict[str, Any]]:
    tools = create_sliding_window_extraction_tools()
    return tools.to_openai_tools()
