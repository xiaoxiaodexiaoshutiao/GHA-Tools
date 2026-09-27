from typing import Dict, Any, List, Callable, Optional, Type, get_type_hints, Union
from pydantic import BaseModel, Field
from functools import wraps
import inspect
import json


class ToolRegistry:
    _tools: Dict[str, 'FunctionTool'] = {}

    @classmethod
    def register(cls, tool: 'FunctionTool'):
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> Optional['FunctionTool']:
        return cls._tools.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, 'FunctionTool']:
        return cls._tools.copy()

    @classmethod
    def clear(cls):
        cls._tools.clear()


class FunctionTool:
    def __init__(
        self,
        func: Callable,
        params_model: Type[BaseModel],
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.func = func
        self.params_model = params_model
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()

    def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        try:
            validated_params = self.params_model(**arguments)

            sig = inspect.signature(self.func)
            param_names = list(sig.parameters.keys())

            if len(param_names) == 2 and 'context' in param_names:
                result = self.func(validated_params, context or {})
            elif len(param_names) == 1:
                result = self.func(validated_params)
            else:
                result = self.func(**validated_params.model_dump(), context=context or {})

            if isinstance(result, str):
                return result
            else:
                return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "error": f"Tool execution error: {str(e)}",
                "success": False
            }, ensure_ascii=False)

    def to_openai_schema(self) -> Dict[str, Any]:
        json_schema = self.params_model.model_json_schema()

        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        cleaned_properties = {}
        for key, value in properties.items():
            cleaned_prop = {
                "type": value.get("type", "string"),
                "description": value.get("description", "")
            }

            if "enum" in value:
                cleaned_prop["enum"] = value["enum"]

            if "minimum" in value:
                cleaned_prop["minimum"] = value["minimum"]
            if "maximum" in value:
                cleaned_prop["maximum"] = value["maximum"]
            if "default" in value:
                cleaned_prop["default"] = value["default"]
            cleaned_properties[key] = cleaned_prop

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": cleaned_properties,
                    "required": required
                }
            }
        }

def function_tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    params_model: Optional[Type[BaseModel]] = None
) -> Union[Callable, FunctionTool]:
    def decorator(f: Callable) -> FunctionTool:
        model = params_model
        if model is None:
            hints = get_type_hints(f)
            sig = inspect.signature(f)
            first_param = list(sig.parameters.keys())[0] if sig.parameters else None

            if first_param and first_param in hints:
                hint_type = hints[first_param]
                if isinstance(hint_type, type) and issubclass(hint_type, BaseModel):
                    model = hint_type

        if model is None:
            raise ValueError(
                f"Cannot infer parameter model for function {f.__name__}. "
                "Please ensure the first parameter has a Pydantic BaseModel type annotation, "
                "or explicitly specify using the params_model parameter."
            )

        tool = FunctionTool(
            func=f,
            params_model=model,
            name=name,
            description=description
        )

        ToolRegistry.register(tool)

        return tool

    if func is not None:
        return decorator(func)
    return decorator


class ToolCollection:
    def __init__(self, tools: List[FunctionTool] = None):
        self._tools: Dict[str, FunctionTool] = {}
        if tools:
            for tool in tools:
                self.add(tool)

    def add(self, tool: FunctionTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[FunctionTool]:
        return self._tools.get(name)

    def execute(self, name: str, arguments: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> str:
        tool = self.get(name)
        if tool is None:
            return json.dumps({
                "error": f"Unknown tool: {name}",
                "success": False
            }, ensure_ascii=False)
        return tool.execute(arguments, context)

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    @property
    def names(self) -> List[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())


class ReadLogLinesParams(BaseModel):
    position: str = Field(
        ...,
        description="Read position: HEAD to read from log beginning, TAIL to read from log end",
        json_schema_extra={"enum": ["HEAD", "TAIL"]}
    )
    count: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Number of lines to read"
    )


class ReadWorkflowContentParams(BaseModel):
    pass


class SearchLogContentParams(BaseModel):
    keyword: str = Field(
        ...,
        description="Keyword or phrase to search for"
    )
    context_lines: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of context lines around matching lines (default: 3)"
    )


class ExpandTimestampBlockParams(BaseModel):
    direction: str = Field(
        ...,
        description="Expansion direction: UP for previous timestamp blocks (recommended), DOWN for subsequent timestamp blocks (may be blocked)",
        json_schema_extra={"enum": ["UP", "DOWN"]}
    )
    reason: str = Field(
        ...,
        description="[Required] Specific reason for calling this function: 1. What information is currently missing? 2. What do you expect to obtain through expansion?"
    )


@function_tool(description="Read more log lines. Call this function when the current log content is insufficient for making a judgment.")
def read_log_lines(params: ReadLogLinesParams, context: Dict[str, Any]) -> Dict[str, Any]:
    position = params.position
    count = params.count

    log_lines = context.get("log_lines", [])
    current_head_lines = context.get("current_head_lines", 0)
    current_tail_lines = context.get("current_tail_lines", 0)

    if not log_lines:
        return {
            "success": False,
            "error": "No log lines available",
            "content": ""
        }

    if position == "HEAD":
        new_head_lines = current_head_lines + count
        new_head_lines = min(new_head_lines, len(log_lines))
        context["current_head_lines"] = new_head_lines

        new_lines = log_lines[current_head_lines:new_head_lines]
        content = "\n".join(new_lines)

        return {
            "success": True,
            "position": "HEAD",
            "lines_read": len(new_lines),
            "total_head_lines": new_head_lines,
            "content": content
        }

    else:
        new_tail_lines = current_tail_lines + count
        new_tail_lines = min(new_tail_lines, len(log_lines))
        context["current_tail_lines"] = new_tail_lines

        if current_tail_lines == 0:
            new_lines = log_lines[-new_tail_lines:] if new_tail_lines > 0 else []
        else:
            start_idx = -new_tail_lines
            end_idx = -current_tail_lines if current_tail_lines > 0 else len(log_lines)
            new_lines = log_lines[start_idx:end_idx]

        content = "\n".join(new_lines)

        return {
            "success": True,
            "position": "TAIL",
            "lines_read": len(new_lines),
            "total_tail_lines": new_tail_lines,
            "content": content
        }


@function_tool(description="Read complete workflow configuration content. Call this function if you need to view the complete workflow configuration to understand the build process.")
def read_workflow_content(params: ReadWorkflowContentParams, context: Dict[str, Any]) -> Dict[str, Any]:
    workflow_content = context.get("workflow_content", "")

    if not workflow_content:
        return {
            "success": False,
            "error": "No workflow content available",
            "content": ""
        }

    return {
        "success": True,
        "content": workflow_content,
        "length": len(workflow_content)
    }


@function_tool(description="Search for keywords in the log. Call this function to search when you need to find specific error messages or keywords.")
def search_log_content(params: SearchLogContentParams, context: Dict[str, Any]) -> Dict[str, Any]:
    keyword = params.keyword
    context_lines = params.context_lines

    log_lines = context.get("log_lines", [])

    if not keyword:
        return {
            "success": False,
            "error": "No keyword provided",
            "matches": []
        }

    if not log_lines:
        return {
            "success": False,
            "error": "No log lines available",
            "matches": []
        }

    matches = []
    for i, line in enumerate(log_lines):
        if keyword.lower() in line.lower():
            start_idx = max(0, i - context_lines)
            end_idx = min(len(log_lines), i + context_lines + 1)

            context_snippet = []
            for j in range(start_idx, end_idx):
                marker = ">>> " if j == i else "    "
                context_snippet.append(f"{marker}{log_lines[j]}")

            matches.append({
                "line_number": i + 1,
                "line_content": log_lines[i],
                "context": "\n".join(context_snippet)
            })

    return {
        "success": True,
        "keyword": keyword,
        "total_matches": len(matches),
        "matches": matches[:10]
    }


@function_tool(
    description="""[COST WARNING] Calling this function will deduct 10-15 points. Please confirm you really need additional context before calling.

Expand timestamp block to get more context. Use only in the following situations:
1. The current log block information is truly insufficient for judgment
2. There is cross-block error context that needs to be traced
3. Need to understand the complete call chain of the error

Please perform Thought reasoning before use to confirm:
- What information is currently missing?
- What information do you expect to obtain after expansion?
- Is this expansion really necessary?

Expands one timestamp block per call. Expansion count is dynamically limited:
- Maximum expansion counts for forward (UP) and backward (DOWN) are dynamically calculated by the system based on error status of subsequent blocks
- Forward: maximum max_expand_up blocks, backward: maximum max_expand_down blocks
- Total expansions cannot exceed 10 times

Since the system uses reverse traversal from back to front, timestamp blocks in the DOWN direction may have already been processed, making downward expansion impossible.
Recommend prioritizing UP direction to get more preceding context."""
)
def expand_timestamp_block(params: ExpandTimestampBlockParams, context: Dict[str, Any]) -> Dict[str, Any]:
    direction = params.direction
    reason = params.reason

    current_block_index = context.get("current_block_index")
    all_timestamp_blocks = context.get("all_timestamp_blocks", [])
    log_lines = context.get("log_lines", [])
    expanded_blocks_up = context.get("expanded_blocks_up", 0)
    expanded_blocks_down = context.get("expanded_blocks_down", 0)
    verified_blocks = context.get("verified_blocks", set())

    max_expand_up = context.get("max_expand_up", 10)
    max_expand_down = context.get("max_expand_down", 0)

    total_expanded = expanded_blocks_up + expanded_blocks_down
    current_score = 100 - (total_expanded * 10)

    if current_block_index is None:
        return {
            "success": False,
            "error": "Current block index not set",
            "content": "",
            "cost_info": {
                "total_expansions": total_expanded,
                "current_score": current_score
            }
        }

    if not all_timestamp_blocks:
        return {
            "success": False,
            "error": "No timestamp blocks available",
            "content": "",
            "cost_info": {
                "total_expansions": total_expanded,
                "current_score": current_score
            }
        }

    if total_expanded >= 10:
        return {
            "success": False,
            "error": "Maximum expansion count (10 times) reached. Current score has dropped to 0. Please make judgment based on available information.",
            "content": "",
            "cost_info": {
                "total_expansions": total_expanded,
                "current_score": 0,
                "warning": "Expansion limit reached"
            }
        }

    if direction == "UP":
        if expanded_blocks_up >= max_expand_up:
            return {
                "success": False,
                "error": f"Forward (UP direction) expansion limit reached (maximum {max_expand_up} blocks). Currently expanded forward {expanded_blocks_up} blocks. Please make judgment based on available information, or try backward (DOWN direction) expansion.",
                "content": "",
                "cost_info": {
                    "total_expansions": total_expanded,
                    "current_score": current_score,
                    "max_expand_up": max_expand_up,
                    "max_expand_down": max_expand_down
                }
            }

        target_index = current_block_index - expanded_blocks_up - 1

        if target_index < 0:
            return {
                "success": False,
                "error": "Reached log beginning, cannot continue expanding upward. Please make judgment based on available information.",
                "content": "",
                "cost_info": {
                    "total_expansions": total_expanded,
                    "current_score": current_score
                }
            }

        context["expanded_blocks_up"] = expanded_blocks_up + 1

    else:
        if expanded_blocks_down >= max_expand_down:
            return {
                "success": False,
                "error": f"Backward (DOWN direction) expansion limit reached (maximum {max_expand_down} blocks). Currently expanded backward {expanded_blocks_down} blocks. Since subsequent blocks may contain errors, the system has limited backward expansion. Please make judgment based on available information, or try forward (UP direction) expansion.",
                "content": "",
                "cost_info": {
                    "total_expansions": total_expanded,
                    "current_score": current_score,
                    "max_expand_up": max_expand_up,
                    "max_expand_down": max_expand_down
                }
            }

        target_index = current_block_index + expanded_blocks_down + 1

        if target_index >= len(all_timestamp_blocks):
            return {
                "success": False,
                "error": "Reached log end, cannot continue expanding downward. Please make judgment based on available information.",
                "content": "",
                "cost_info": {
                    "total_expansions": total_expanded,
                    "current_score": current_score
                }
            }

        target_block = all_timestamp_blocks[target_index]
        if target_block in verified_blocks:
            return {
                "success": False,
                "error": "The timestamp block for downward expansion has been verified and processed, cannot continue traversing downward. Due to reverse traversal, recommend prioritizing upward (UP direction) expansion to get more context.",
                "blocked_by_verified": True,
                "verified_block_range": list(target_block),
                "content": "",
                "cost_info": {
                    "total_expansions": total_expanded,
                    "current_score": current_score
                }
            }

        context["expanded_blocks_down"] = expanded_blocks_down + 1

    block_start, block_end = all_timestamp_blocks[target_index]
    block_content = "\n".join(log_lines[block_start-1:block_end])

    new_total_expanded = context["expanded_blocks_up"] + context.get("expanded_blocks_down", 0)
    new_score = 100 - (new_total_expanded * 10)

    remaining_up = max_expand_up - context["expanded_blocks_up"]
    remaining_down = max_expand_down - context.get("expanded_blocks_down", 0)

    return {
        "success": True,
        "direction": direction,
        "reason_provided": reason,
        "block_index": target_index,
        "block_range": [block_start, block_end],
        "total_expanded_up": context["expanded_blocks_up"],
        "total_expanded_down": context.get("expanded_blocks_down", 0),
        "content": block_content,
        "cost_info": {
            "this_expansion_cost": 10,
            "total_expansions": new_total_expanded,
            "current_score": new_score,
            "remaining_expansions": 10 - new_total_expanded,
            "remaining_up": remaining_up,
            "remaining_down": remaining_down,
            "reminder": f"Current score: {new_score} points. Remaining forward expansion: {remaining_up} blocks, backward expansion: {remaining_down} blocks. Please evaluate if you have sufficient information to make judgment."
        }
    }


def create_classification_tools() -> ToolCollection:
    return ToolCollection([
        read_log_lines,
        read_workflow_content,
        search_log_content
    ])


def create_judgment_tools() -> ToolCollection:
    return ToolCollection([
        expand_timestamp_block
    ])


class ExpandSlidingWindowParams(BaseModel):
    direction: str = Field(
        ...,
        description="Expansion direction: FORWARD to expand forward (get content before current window), BACKWARD to expand backward (get content after current window)",
        json_schema_extra={"enum": ["FORWARD", "BACKWARD"]}
    )
    reason: str = Field(
        ...,
        description="[Required] Specific reason for calling this function: 1. Why is the current window content insufficient? 2. What do you expect to obtain through expansion? 3. Is it because an error block was truncated by the window boundary?"
    )


class ViewPreviousExtractionsParams(BaseModel):
    count: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of previous extraction results to view (1-10, default 5)"
    )


class ModifyPreviousExtractionParams(BaseModel):
    extraction_index: int = Field(
        ...,
        ge=0,
        description="Index of extraction result to modify (0 for most recent, 1 for second most recent, etc.)"
    )
    new_ranges: str = Field(
        ...,
        description="New line number range list, in JSON array format, e.g.: [[100, 110], [120, 130]]"
    )
    reason: str = Field(
        ...,
        description="[Required] Modification reason: must detail why the original extraction result was incorrect"
    )
    verification: str = Field(
        ...,
        description="[Required] Verification explanation: explain why the new extraction result is correct and how to verify"
    )


@function_tool(
    description="""[COST WARNING] Expansion is an expensive operation! Forward expansion maximum 1 time, backward expansion maximum 3 times.

Expand sliding window to get more context. Use only in the following situations:
1. The first or last error block in the current window is truncated by window boundary, semantically incomplete
2. Need to understand the full context of a truncated error

[Forward Expansion Conditions] (FORWARD direction):
- Maximum 1 expansion allowed
- If the previous window's extracted error code block already includes the current window's start line, forward expansion is not allowed

[Backward Expansion] (BACKWARD direction):
- Maximum 3 expansions allowed
- Need to carefully consider whether to continue after each expansion

Please perform [Thought] reasoning before use to confirm:
- Is the first or last error block you're preparing to extract truncated by window boundary?
- Does the truncation cause semantic incompleteness?
- How to handle overlapping content after expansion?"""
)
def expand_sliding_window(params: ExpandSlidingWindowParams, context: Dict[str, Any]) -> Dict[str, Any]:
    direction = params.direction
    reason = params.reason

    window_start = context.get("window_start", 0)
    window_end = context.get("window_end", 0)
    total_lines = context.get("total_lines", 0)
    log_lines = context.get("log_lines", [])
    window_size = context.get("window_size", 200)

    forward_expansions = context.get("forward_expansions", 0)
    backward_expansions = context.get("backward_expansions", 0)

    max_forward_expansion = context.get("max_forward_expansion", 1)
    max_backward_expansion = context.get("max_backward_expansion", 3)

    can_expand_forward = context.get("can_expand_forward", True)
    previous_extractions = context.get("previous_extractions", [])

    if direction == "FORWARD":
        if forward_expansions >= max_forward_expansion:
            return {
                "success": False,
                "error": f"Forward expansion limit reached (maximum {max_forward_expansion} times). Currently forward expanded {forward_expansions} times. Please extract based on available information.",
                "content": "",
                "expansion_info": {
                    "forward_expansions": forward_expansions,
                    "backward_expansions": backward_expansions,
                    "max_forward": max_forward_expansion,
                    "max_backward": max_backward_expansion
                }
            }

        if not can_expand_forward:
            return {
                "success": False,
                "error": "Forward expansion blocked: The previous window's extracted error code block already includes the current window's start line, cannot expand forward. This is to avoid duplicate extraction.",
                "blocked_reason": "previous_extraction_overlap",
                "content": "",
                "expansion_info": {
                    "forward_expansions": forward_expansions,
                    "backward_expansions": backward_expansions
                }
            }

        if window_start <= 1:
            return {
                "success": False,
                "error": "Reached log beginning, cannot continue forward expansion.",
                "content": "",
                "expansion_info": {
                    "forward_expansions": forward_expansions,
                    "backward_expansions": backward_expansions
                }
            }

        expand_start = max(1, window_start - window_size)
        expand_end = window_start - 1

        expand_content = "\n".join(log_lines[expand_start-1:expand_end])

        context["forward_expansions"] = forward_expansions + 1
        context["expanded_window_start"] = expand_start

        return {
            "success": True,
            "direction": "FORWARD",
            "reason_provided": reason,
            "expanded_range": [expand_start, expand_end],
            "content": expand_content,
            "lines_added": expand_end - expand_start + 1,
            "expansion_info": {
                "forward_expansions": context["forward_expansions"],
                "backward_expansions": backward_expansions,
                "remaining_forward": max_forward_expansion - context["forward_expansions"],
                "remaining_backward": max_backward_expansion - backward_expansions
            },
            "note": "Forward expansion successful. Note: The end of expansion content may overlap with the beginning of original window, deduplication needed during extraction."
        }

    else:
        if backward_expansions >= max_backward_expansion:
            return {
                "success": False,
                "error": f"Backward expansion limit reached (maximum {max_backward_expansion} times). Currently backward expanded {backward_expansions} times. Please extract based on available information.",
                "content": "",
                "expansion_info": {
                    "forward_expansions": forward_expansions,
                    "backward_expansions": backward_expansions,
                    "max_forward": max_forward_expansion,
                    "max_backward": max_backward_expansion
                }
            }

        if window_end >= total_lines:
            return {
                "success": False,
                "error": "Reached log end, cannot continue backward expansion.",
                "content": "",
                "expansion_info": {
                    "forward_expansions": forward_expansions,
                    "backward_expansions": backward_expansions
                }
            }

        expand_start = window_end + 1
        expand_end = min(total_lines, window_end + window_size)

        expand_content = "\n".join(log_lines[expand_start-1:expand_end])

        context["backward_expansions"] = backward_expansions + 1
        context["expanded_window_end"] = expand_end

        return {
            "success": True,
            "direction": "BACKWARD",
            "reason_provided": reason,
            "expanded_range": [expand_start, expand_end],
            "content": expand_content,
            "lines_added": expand_end - expand_start + 1,
            "expansion_info": {
                "forward_expansions": forward_expansions,
                "backward_expansions": context["backward_expansions"],
                "remaining_forward": max_forward_expansion - forward_expansions,
                "remaining_backward": max_backward_expansion - context["backward_expansions"]
            },
            "note": "Backward expansion successful. Note: The beginning of expansion content may overlap with the end of original window, deduplication needed during extraction."
        }


@function_tool(
    description="""View error code block results extracted from previous windows.

Usage:
1. Understand the error context found in previous windows
2. Avoid extracting the same error blocks repeatedly
3. Determine if the current window's errors are related to previous errors

Can view up to 10 most recent extraction results."""
)
def view_previous_extractions(params: ViewPreviousExtractionsParams, context: Dict[str, Any]) -> Dict[str, Any]:
    count = params.count

    previous_extractions = context.get("previous_extractions", [])

    if not previous_extractions:
        return {
            "success": True,
            "message": "No previous extraction results. This is the first window or previous windows found no errors.",
            "extractions": [],
            "total_count": 0
        }

    recent_extractions = list(reversed(previous_extractions[-count:]))

    formatted_results = []
    for i, extraction in enumerate(recent_extractions):
        formatted_results.append({
            "index": i,
            "window_index": extraction.get("window_index", "unknown"),
            "window_range": extraction.get("window_range", [0, 0]),
            "extracted_ranges": extraction.get("ranges", []),
            "extraction_time": extraction.get("timestamp", "unknown")
        })

    return {
        "success": True,
        "extractions": formatted_results,
        "total_count": len(previous_extractions),
        "showing_count": len(formatted_results),
        "note": "index=0 means most recent extraction result, higher numbers indicate earlier results"
    }


@function_tool(
    description="""[SERIOUS WARNING] Modifying previous extraction results is a very serious operation!

Modification is only allowed in the following situations:
1. Found obvious errors in previous extraction results (missed or over-extracted)
2. Current window context proves previous extraction range is incorrect
3. Confirmed modification is needed after careful thought and verification

Modification must provide:
- Detailed reason for modification
- Why the original result was incorrect
- How to verify the new result is correct

Abusing this function will cause result confusion, please use with caution!"""
)
def modify_previous_extraction(params: ModifyPreviousExtractionParams, context: Dict[str, Any]) -> Dict[str, Any]:
    extraction_index = params.extraction_index
    new_ranges_str = params.new_ranges
    reason = params.reason
    verification = params.verification

    previous_extractions = context.get("previous_extractions", [])

    if not previous_extractions:
        return {
            "success": False,
            "error": "No previous extraction results to modify",
            "extractions_count": 0
        }

    if extraction_index < 0 or extraction_index >= len(previous_extractions):
        return {
            "success": False,
            "error": f"Invalid index: {extraction_index}. Valid range: 0-{len(previous_extractions)-1} (0 means most recent result)",
            "extractions_count": len(previous_extractions)
        }

    try:
        new_ranges = json.loads(new_ranges_str)
        if not isinstance(new_ranges, list):
            raise ValueError("new_ranges must be in array format")
        for r in new_ranges:
            if not isinstance(r, list) or len(r) != 2:
                raise ValueError("Each range must be in [start, end] format")
            if r[0] > r[1]:
                raise ValueError(f"Invalid range: {r}, start line cannot be greater than end line")
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Cannot parse new_ranges parameter: {str(e)}. Please use JSON array format, e.g.: [[100, 110], [120, 130]]"
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }

    actual_index = len(previous_extractions) - 1 - extraction_index
    old_extraction = previous_extractions[actual_index]
    old_ranges = old_extraction.get("ranges", [])

    modification_record = {
        "modified_at": context.get("current_window_index", "unknown"),
        "original_ranges": old_ranges,
        "new_ranges": new_ranges,
        "reason": reason,
        "verification": verification
    }

    previous_extractions[actual_index]["ranges"] = new_ranges
    previous_extractions[actual_index]["modified"] = True
    previous_extractions[actual_index]["modification_history"] = (
        previous_extractions[actual_index].get("modification_history", []) + [modification_record]
    )

    context["has_modifications"] = True
    if "modification_log" not in context:
        context["modification_log"] = []
    context["modification_log"].append(modification_record)

    return {
        "success": True,
        "message": "Modification successful",
        "extraction_index": extraction_index,
        "window_index": old_extraction.get("window_index", "unknown"),
        "original_ranges": old_ranges,
        "new_ranges": new_ranges,
        "modification_record": modification_record,
        "warning": "Please ensure this modification was carefully considered. If the modification is incorrect, you may need to call this function again to correct it."
    }


def create_sliding_window_extraction_tools() -> ToolCollection:
    return ToolCollection([
        expand_sliding_window,
        view_previous_extractions,
        modify_previous_extraction
    ])
