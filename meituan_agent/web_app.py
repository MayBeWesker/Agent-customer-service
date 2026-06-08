import os
import sys

import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from agent.react_agent import MeituanDispatchAgent


st.set_page_config(page_title="Meituan Dispatch Agent", layout="wide")

st.title("美团调度 Agent")
st.caption("使用自然语言提问，调用求解器输出任务分配选择结果。")
st.divider()

with st.sidebar:
    st.subheader("求解设置")
    solver_mode = st.selectbox("Solver", ["auto", "sa", "baseline"], index=0)
    result_limit = st.slider("展示结果条数", min_value=5, max_value=50, value=20, step=5)
    uploaded_file = st.file_uploader("上传 case 文件", type=["txt"])
    st.markdown(
        "示例提问：`请使用 sa 求解 example/large_seed301.txt，并输出前 10 条选择`"
    )

if "agent" not in st.session_state:
    st.session_state["agent"] = MeituanDispatchAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("请输入你的问题，比如：请使用 sa 求解当前 case，并输出对应的选择")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    input_text = None
    if uploaded_file is not None:
        input_text = uploaded_file.getvalue().decode("utf-8-sig")

    solver_name = None if solver_mode == "auto" else solver_mode

    with st.spinner("Agent 正在求解中"):
        response_chunks = []
        response_stream = st.session_state["agent"].execute_stream(
            prompt,
            input_text=input_text,
            solver_name=solver_name,
            limit=result_limit,
        )

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    yield char

        st.chat_message("assistant").write_stream(capture(response_stream, response_chunks))
        st.session_state["messages"].append(
            {"role": "assistant", "content": "".join(response_chunks)}
        )
        st.rerun()
