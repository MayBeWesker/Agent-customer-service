from __future__ import annotations

import streamlit as st

import config
from rag import MedicationRagService


def render_qa_panel(medication_box: list[dict]) -> None:
    st.subheader("5. 用药问答")
    st.caption("问题会结合药品知识库和当前电子药箱一起回答，尽量降低纯大模型幻觉。")
    st.caption(f"当前默认药品语境：{config.DEFAULT_MEDICATION_NAME}")

    quick_questions = [
        "这个药有什么副作用？",
        "漏服一次怎么办？",
        "能不能和我现在药箱里的药一起吃？",
    ]
    preset = st.radio("快捷问题", ["自定义提问"] + quick_questions, horizontal=True)
    question = st.text_input(
        "请输入问题",
        value="" if preset == "自定义提问" else preset,
        placeholder="例如：我早上忘记吃降压药了怎么办？",
    )

    if st.button("开始问答", use_container_width=True):
        rag_service = MedicationRagService()
        result = rag_service.answer(question, medication_box)
        st.markdown("#### AI 回答")
        st.write(result.answer)

        with st.expander("查看 system prompt 与检索上下文"):
            st.code(result.prompt, language="text")

        with st.expander("查看命中的知识库条目"):
            for doc in result.references:
                st.json(doc)

        if rag_service.llm_client.is_configured():
            st.success(f"当前使用真实大模型：{config.LLM_MODEL_NAME}")
        else:
            st.warning("当前未配置 API Key，仍在使用本地规则型回答。配置后会自动切换到真实大模型。")
        st.info(config.DISCLAIMER)
