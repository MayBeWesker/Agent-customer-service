"""
基于 Streamlit 完成 WEB 网页上传
"""

import streamlit as st
from knowledge_base import KnowledgeBaseService
import time

st.title("知识库更新服务")

uploader_file = st.file_uploader(
    "请上传TXT文件",
    type=['txt'],
    accept_multiple_files=False
)


#防止状态丢失导致无法维护
if "counter" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


# 基础信息
if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size/1024 # KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    text = uploader_file.getvalue().decode("utf-8")

    with st.spinner("Loading knowledge database ..."):
        time.sleep(1)
        result = st.session_state['service'].upload_by_str(text, file_name)
        st.write(result)