from __future__ import annotations

from datetime import datetime

import streamlit as st

import config
from app_file_uploader import render_medication_uploader
from app_qa import render_qa_panel
from knowledge_base import KnowledgeBaseService
from storage import ensure_json_file, load_json, save_json, now_iso


st.set_page_config(
    page_title="AI用药伴侣 Demo",
    page_icon="💊",
    layout="wide",
)


def initialize_storage() -> None:
    ensure_json_file(config.MEDICATION_BOX_PATH, [])
    ensure_json_file(config.REMINDER_LOG_PATH, [])
    ensure_json_file(config.SESSION_HISTORY_PATH, [])
    medication_box = load_json(config.MEDICATION_BOX_PATH, default=[])
    changed = False
    for item in medication_box:
        if item.get("name") == "999":
            item["name"] = config.DEFAULT_MEDICATION_NAME
            changed = True
    if changed:
        save_json(config.MEDICATION_BOX_PATH, medication_box)


def render_home_header(medication_box: list[dict], reminder_logs: list[dict]) -> None:
    st.title("AI用药伴侣")
    st.caption("面向 55-75 岁中老年人的用药记录、提醒与问答 Demo")
    col1, col2, col3 = st.columns(3)
    col1.metric("电子药箱药品数", len(medication_box))
    col2.metric("今日提醒记录", len([item for item in reminder_logs if item["scheduled_at"].startswith(datetime.now().strftime("%Y-%m-%d"))]))
    col3.metric("知识库药品条目", len(KnowledgeBaseService().list_documents()))


def build_today_schedule(medication_box: list[dict], reminder_logs: list[dict]) -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    schedule = []
    for item in medication_box:
        for reminder_time in item["times"]:
            scheduled_at = f"{today} {reminder_time}"
            status = "待服用"
            for log in reminder_logs:
                if log["medication_id"] == item["id"] and log["scheduled_at"] == scheduled_at:
                    status = log["status"]
            schedule.append(
                {
                    "medication_id": item["id"],
                    "name": item["name"],
                    "usage": item["usage"],
                    "scheduled_at": scheduled_at,
                    "status": status,
                    "dedupe_key": f"{item['id']}-{scheduled_at}",
                    "logged": any(log["medication_id"] == item["id"] and log["scheduled_at"] == scheduled_at for log in reminder_logs),
                }
            )
    schedule.sort(key=lambda item: item["scheduled_at"])
    return schedule


def render_medication_box(medication_box: list[dict]) -> None:
    st.subheader("3. 电子药箱")
    if not medication_box:
        st.info("还没有药品，请先在上方完成拍照录药。")
        return
    for item in medication_box:
        with st.container(border=True):
            left, right = st.columns([2, 1])
            left.markdown(f"**{item['name']}**  {item['spec']}")
            left.write(f"频次：{item['frequency']} | 时间：{'、'.join(item['times'])}")
            left.write(f"用法用量：{item['usage']}")
            right.caption(f"来源：{item['source_file']}")
            right.caption(f"建档时间：{item['created_at']}")


def render_reminder_panel(medication_box: list[dict], reminder_logs: list[dict]) -> None:
    st.subheader("4. 定时提醒与服药记录")
    schedule = build_today_schedule(medication_box, reminder_logs)
    if not schedule:
        st.info("当前还没有生成今日计划。")
        return

    for item in schedule:
        with st.container(border=True):
            top_left, top_right = st.columns([3, 2])
            top_left.markdown(f"**{item['name']}**")
            top_left.write(f"计划时间：{item['scheduled_at']} | 当前状态：{item['status']}")
            top_left.caption(f"提示文案：该服药啦，请按计划服用。")
            top_right.write(f"用法：{item['usage']}")

            cols = st.columns(3)
            for idx, status in enumerate(config.DEFAULT_REMINDER_CHOICES):
                disabled = item["logged"]
                if cols[idx].button(status, key=f"{item['dedupe_key']}-{status}", use_container_width=True, disabled=disabled):
                    reminder_logs.append(
                        {
                            "medication_id": item["medication_id"],
                            "medication_name": item["name"],
                            "scheduled_at": item["scheduled_at"],
                            "status": status,
                            "recorded_at": now_iso(),
                        }
                    )
                    save_json(config.REMINDER_LOG_PATH, reminder_logs)
                    st.rerun()


def render_family_summary(reminder_logs: list[dict]) -> None:
    st.subheader("6. 家属协同 / 持续管理")
    if not reminder_logs:
        st.info("暂无服药日志。")
        return
    status_count: dict[str, int] = {}
    for log in reminder_logs:
        status_count[log["status"]] = status_count.get(log["status"], 0) + 1
    cols = st.columns(len(status_count))
    for idx, (status, count) in enumerate(status_count.items()):
        cols[idx].metric(status, count)
    st.dataframe(reminder_logs, use_container_width=True)


def render_sidebar() -> None:
    st.sidebar.header("Demo 说明")
    st.sidebar.write("1. 先在“拍照录药”里上传图片或选择样例药品。")
    st.sidebar.write("2. 确认识别结果后，系统会生成电子药箱与今日计划。")
    st.sidebar.write("3. 到“用药问答”页签可体验 RAG 问答。")
    st.sidebar.info(config.DISCLAIMER)


def main() -> None:
    initialize_storage()
    render_sidebar()
    medication_box = load_json(config.MEDICATION_BOX_PATH, default=[])
    reminder_logs = load_json(config.REMINDER_LOG_PATH, default=[])

    render_home_header(medication_box, reminder_logs)
    st.divider()

    tabs = st.tabs(["拍照录药", "电子药箱", "提醒记录", "用药问答", "家属视图"])
    with tabs[0]:
        render_medication_uploader()
    with tabs[1]:
        render_medication_box(medication_box)
    with tabs[2]:
        render_reminder_panel(medication_box, reminder_logs)
    with tabs[3]:
        render_qa_panel(medication_box)
    with tabs[4]:
        render_family_summary(reminder_logs)


if __name__ == "__main__":
    main()
