import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain.agents import create_agent

from agent.tools.agent_tools import (
    CURRENT_PREFERRED_SOLVER,
    detect_case_path,
    detect_solver_name,
    get_last_solution_tool,
    inspect_case_tool,
    list_solver_tool,
    set_case_context,
    solve_dispatch_case_tool,
)
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt


def _message_to_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


class MeituanDispatchAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[
                solve_dispatch_case_tool,
                inspect_case_tool,
                list_solver_tool,
                get_last_solution_tool,
            ],
        )

    def execute(
        self,
        query: str,
        case_path: str | None = None,
        input_text: str | None = None,
        solver_name: str | None = None,
        limit: int = 20,
    ) -> str:
        resolved_solver_name = solver_name or detect_solver_name(query, default=CURRENT_PREFERRED_SOLVER)
        resolved_case_path = case_path or detect_case_path(query)
        set_case_context(
            input_text=input_text,
            case_source=resolved_case_path,
            preferred_solver=resolved_solver_name,
        )

        user_message = (
            f"{query}\n\n"
            f"补充上下文：\n"
            f"- 如果用户没有明确指定 solver，默认优先使用 {resolved_solver_name}。\n"
            f"- 如果用户要输出结果，优先调用工具，并将 limit 设为 {limit}。\n"
        )
        if resolved_case_path:
            user_message += f"- 当前 case 路径候选：{resolved_case_path}\n"
        if input_text:
            user_message += "- 当前存在上传的 case 内容，优先基于上传内容调用工具。\n"

        result = self.agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": user_message},
                ]
            }
        )
        return _message_to_text(result["messages"][-1])

    def execute_stream(self, query: str, **kwargs):
        yield self.execute(query, **kwargs)


if __name__ == "__main__":
    agent = MeituanDispatchAgent()
    print(agent.execute("请使用 sa 求解 example/large_seed301.txt，并输出前 10 条选择", limit=10))
