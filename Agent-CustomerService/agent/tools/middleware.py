from typing import Callable

from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger
from langgraph.runtime import Runtime
from utils.prompt_loader import load_system_prompts, load_report_prompts


# 工具执行的监控
@wrap_tool_call
def monitor_tool(request: ToolCallRequest, handler:Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
    """
    requset: 请求的数据封装
    handler: 执行的函数本身
    """
    logger.info(f"[tool monitor] tool using :{request.tool_call['name']}")
    logger.info(f"[tool monitor] parameter: {request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor] tool {request.tool_call['name']} calling successful ")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"tool {request.tool_call['name']} calling error, due to {str(e)}")
        raise e
    


# 在模型执行之前输出日志
@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime
):
    logger.info(f"[log_before_model] Going to use the model, with {len(state['messages'])} messages")
    logger.debug(f"[log_before_model] {type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None

# 动态切换提示词
@dynamic_prompt #每一次生成一次提示词的时候会调用这个函数
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()
