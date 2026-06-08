"""
总结服务类：用户提问，搜索参考的资料，将提问和参考资料提交给模型，让模型总结回复
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_ROOT = os.path.dirname(CURRENT_DIR)
UTILS_DIR = os.path.join(AGENT_ROOT, "utils")

for path in (AGENT_ROOT, UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt |self.model | StrOutputParser()
        return chain
    
    def retriever_docs(self, query:str) -> list[Document] :
        return self.retriever.invoke(query)

    def rag_summarize(self, query:str) -> str :
        context_docs = self.retriever_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"reference doc {counter}: reference doc: {doc.page_content}, reference doc metadata: {doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )
    
if __name__ == '__main__':
    rag = RagSummarizeService()
    res = rag.rag_summarize("扫地机器人一般有什么清扫的模式？")
    print(res)
