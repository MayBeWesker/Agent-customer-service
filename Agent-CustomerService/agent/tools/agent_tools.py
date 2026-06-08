import csv
import os
import random
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
UTILS_DIR = os.path.join(AGENT_ROOT, "utils")

for path in (AGENT_ROOT, UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.logger_handler import logger
from rag.rag_service import RagSummarizeService
from langchain_core.tools import tool
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path


rag = None
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07"]
external_data = {}


def get_rag_service() -> RagSummarizeService:
    global rag
    if rag is None:
        rag = RagSummarizeService()
    return rag

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str :
    return get_rag_service().rag_summarize(query)

@tool(description="获取指定城市的天气情况")
def get_weather(city: str) -> str:
    return f"city: {city}, the weather is sunny, and the tempreture is 26"

@tool(description="获取用户所在的城市")
def get_user_loacation() -> str:
    return random.choice(["Shenzhen", "Hefei", "Hangzhou"])

@tool(description="获取用户的 ID")
def get_user_id() -> str:
    return random.choice(user_ids)

@tool(description="获取当前月份")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id":{
            "month":{"特征", "效率"}
            "month":{"特征", "效率"}
            "month":{"特征", "效率"}
        }
    }
    """
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileExistsError(f"External data file does not exist")
        with open(external_data_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for arr in reader:
                if len(arr) < 6:
                    logger.warning(f"[generate_external_data] invalid csv row: {arr}")
                    continue

                user_id, feature, efficiency, consumables, comparison, time = arr[:6]

                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison
                }



@tool(description="从外部系统中，获取用户在指定月份的使用记录，没有检索到则返回空字符串")
def fetch_external_data(user_id:str, month:str) -> str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data] can not fetch the using history for user {user_id} in {month}")
        return ""
    

@tool(description="无入参数，仅仅出发中间键")
def fill_context_for_report():
    return "[fill_context_for_report] is used"
