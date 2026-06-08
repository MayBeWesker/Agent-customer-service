import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_ROOT = os.path.dirname(CURRENT_DIR)
UTILS_DIR = os.path.join(AGENT_ROOT, "utils")

for path in (AGENT_ROOT, UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_loacation, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)

try:
    from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
    AGENT_MIDDLEWARE = [monitor_tool, log_before_model, report_prompt_switch]
except Exception:
    AGENT_MIDDLEWARE = []

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_loacation, get_user_id,
                    get_current_month, fetch_external_data, fill_context_for_report],
            middleware=AGENT_MIDDLEWARE
        )

    def execute_stream(self, query:str):
        input_dict = {
            "messages" :[
                {"role": "user", "content": query},
            ]
        }
        # 第三个参数是上下文 runtime 中的信息，是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("帮我生成 user_id 为1001 的历史使用报告"):
        print(chunk, end="", flush=True)
