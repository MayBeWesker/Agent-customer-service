from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config
from knowledge_base import KnowledgeBaseService
from llm_client import LLMClient
from storage import load_json, save_json, now_iso


@dataclass
class RagAnswer:
    answer: str
    references: list[dict[str, Any]]
    prompt: str


class MedicationRagService:
    def __init__(self) -> None:
        self.knowledge_base = KnowledgeBaseService()
        self.system_prompt = config.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        self.llm_client = LLMClient()

    def answer(self, question: str, medication_box: list[dict[str, Any]]) -> RagAnswer:
        references = self._retrieve_references(question, medication_box)
        prompt = self._build_prompt(question, medication_box, references)
        answer = self._generate_answer(question, medication_box, references, prompt)
        self._save_history(question, answer)
        return RagAnswer(answer=answer, references=references, prompt=prompt)

    def _retrieve_references(self, question: str, medication_box: list[dict[str, Any]]) -> list[dict[str, Any]]:
        explicit_mentions = self.knowledge_base.find_explicit_mentions(question)
        if explicit_mentions:
            return explicit_mentions[:3]
        default_doc = self.knowledge_base.get_document_by_name(config.DEFAULT_MEDICATION_NAME)
        if default_doc:
            return [default_doc]
        retrieval_query = self._build_retrieval_query(question, medication_box)
        return [item.payload for item in self.knowledge_base.vector_store.query(retrieval_query, top_k=3)]

    @staticmethod
    def _build_retrieval_query(question: str, medication_box: list[dict[str, Any]]) -> str:
        medication_names = " ".join(item.get("name", "") for item in medication_box)
        return f"{question} {medication_names}".strip()

    def _build_prompt(
        self,
        question: str,
        medication_box: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> str:
        med_context = "\n".join(
            f"- {item['name']} {item['spec']} | 频次：{item['frequency']} | 时间：{', '.join(item['times'])}"
            for item in medication_box
        ) or "- 当前电子药箱为空"
        reference_context = "\n".join(
            f"- {doc['name']}: {doc['summary']}"
            for doc in references
        ) or "- 无命中药品资料"
        return (
            f"【用户电子药箱】\n{med_context}\n\n"
            f"【检索到的参考资料】\n{reference_context}\n\n"
            f"【用户问题】\n{question}"
        )

    def _compose_answer(
        self,
        question: str,
        medication_box: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> str:
        lowered = question.lower()
        primary = references[0] if references else None
        med_names = [item["name"] for item in medication_box]

        if not primary:
            return (
                "我暂时没有从药品知识库里检索到足够的信息。"
                "建议先确认药品名称和规格，或把说明书补充进知识库后再提问。\n\n"
                f"提示：{config.DISCLAIMER}"
            )

        if "副作用" in question or "不良反应" in question:
            return (
                f"{primary['name']} 常见需要关注的副作用：{self._normalize_text(primary['side_effects'])}。\n"
                f"用法提示：{self._normalize_text(primary['usage'])}。\n"
                f"如果出现持续不适、胸闷、呼吸困难或明显低血糖等情况，请尽快联系医生。\n\n"
                f"提示：{config.DISCLAIMER}"
            )

        if "一天" in question and ("几次" in question or "怎么吃" in question or "用法" in question or "用量" in question):
            return (
                f"{primary['name']} 的建议用法用量是：{self._normalize_text(primary['usage'])}。\n"
                f"也就是默认按 {primary.get('default_frequency', '说明书建议频次')} 服用，推荐时间为：{'、'.join(primary.get('default_times', [])) or '请按说明书或医嘱'}。\n\n"
                f"提示：{config.DISCLAIMER}"
            )

        if "漏服" in question or "忘记" in question:
            return (
                f"关于 {primary['name']} 的漏服处理：{self._normalize_text(primary['missed_dose'])}。\n"
                "一般不要自行加倍补服；如果已经接近下一次服药时间，优先按原计划继续，并观察身体反应。\n\n"
                f"提示：{config.DISCLAIMER}"
            )

        if "一起吃" in question or "同服" in question or "冲突" in question or "相互作用" in lowered:
            med_list = "、".join(med_names) if med_names else "当前药箱内药品"
            return (
                f"我已结合你当前电子药箱中的药品：{med_list} 进行判断。\n"
                f"知识库提示：{self._normalize_text(primary['interactions'])}。\n"
                "如果你正在同时服用多种慢病药、抗凝药或出现头晕心慌等异常，请优先咨询医生或药师确认联合用药方案。\n\n"
                f"提示：{config.DISCLAIMER}"
            )

        return (
            f"根据检索到的药品资料，{primary['name']} 的核心信息如下：{self._normalize_text(primary['summary'])}。\n"
            f"推荐你重点关注：用法用量为“{self._normalize_text(primary['usage'])}”；禁忌提醒为“{self._normalize_text(primary['contraindications'])}”。\n"
            "如果你愿意，我还可以继续帮你看副作用、漏服处理或是否适合与当前药箱中的其他药同服。\n\n"
            f"提示：{config.DISCLAIMER}"
        )

    def _generate_answer(
        self,
        question: str,
        medication_box: list[dict[str, Any]],
        references: list[dict[str, Any]],
        prompt: str,
    ) -> str:
        if self.llm_client.is_configured():
            try:
                return self.llm_client.answer(self.system_prompt, prompt)
            except Exception:
                return self._compose_answer(question, medication_box, references)
        return self._compose_answer(question, medication_box, references)

    @staticmethod
    def _normalize_text(value: str) -> str:
        return value.rstrip("。.!！?")

    def _save_history(self, question: str, answer: str) -> None:
        history = load_json(config.SESSION_HISTORY_PATH, default=[])
        history.append(
            {
                "question": question,
                "answer": answer,
                "created_at": now_iso(),
            }
        )
        save_json(config.SESSION_HISTORY_PATH, history[-20:])
