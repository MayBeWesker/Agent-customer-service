from __future__ import annotations

from pathlib import Path
import os

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
PROMPTS_DIR = APP_ROOT / "prompts"
CONFIG_DIR = APP_ROOT / "config"
ASSETS_DIR = APP_ROOT / "assets"

KNOWLEDGE_PATH = DATA_DIR / "medication_knowledge.json"
MEDICATION_BOX_PATH = DATA_DIR / "medication_box.json"
REMINDER_LOG_PATH = DATA_DIR / "reminder_logs.json"
SESSION_HISTORY_PATH = DATA_DIR / "qa_history.json"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"
WORKFLOW_PATH = CONFIG_DIR / "workflow.json"

DEFAULT_USER_ID = "demo-user-001"
DEFAULT_USER_NAME = "陈阿姨"

DEFAULT_REMINDER_CHOICES = ["已服用", "稍后", "漏服"]
DEFAULT_FREQUENCY_MAP = {
    "每日1次": ["08:00"],
    "每日2次": ["08:00", "20:00"],
    "每日3次": ["08:00", "14:00", "20:00"],
}

SAMPLE_OCR_TEXT = {
    "阿司匹林": "阿司匹林肠溶片 100mg 口服 一次1片 一日1次 早餐后服用。",
    "氨氯地平": "苯磺酸氨氯地平片 5mg 用法用量：一次1片，每日1次，早晨服用。",
    "二甲双胍": "盐酸二甲双胍片 500mg 一次1片 一日2次 建议随餐服用。",
    "999感冒灵": "999感冒灵颗粒 开水冲服 一次1袋 一日3次。用于感冒引起的头痛 发热 鼻塞 流涕 咽痛。",
}

DISCLAIMER = (
    "本 Demo 仅用于产品演示，不替代医生或药师建议；遇到严重不适、特殊人群用药或联合用药冲突，请及时咨询专业人员。"
)

DEFAULT_MEDICATION_NAME = "999感冒灵颗粒"

LLM_PROVIDER = os.getenv("SANJIU_LLM_PROVIDER", "mock")
LLM_API_KEY = os.getenv("SANJIU_LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("SANJIU_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL_NAME = os.getenv("SANJIU_LLM_MODEL_NAME", "qwen-plus")
