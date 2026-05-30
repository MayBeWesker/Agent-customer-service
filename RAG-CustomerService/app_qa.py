
import time
from rag import RagService
import streamlit as st
import config_data as config

st.title("智能客服")
st.divider() # 分隔符

if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"Hello,how can I help you"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()


# 在页面上输出历史记录
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面的最下端提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user", "content":prompt})

    # 用于处理 stream 输出的情况
    ai_res_list = []
    def capture(generator, cache_list):
        for chunk in generator:
            cache_list.append(chunk)
            yield chunk

    with st.spinner("AI 思考中"):
        res_stream = st.session_state["rag"].chain.invoke({"input":prompt}, config.session_config)
        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
        st.session_state["message"].append({"role":"assistant", "content":"".join(ai_res_list)})





