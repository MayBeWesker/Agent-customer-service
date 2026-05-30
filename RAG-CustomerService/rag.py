from vector_store import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from file_history_store import get_history
import config_data as config

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagService(object):
    def __init__(self):
        self.verctor_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system","以我提供的一只参考资料为主，"
                "简洁和专业地回答用户问题。参考资料：{context}"),
                ("system","同时，用户的历史对话记录如下："),
                MessagesPlaceholder("history"),
                ("user","请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.chain = self.__get_chain()

    def __get_chain(self):
        """获得最终的执行链"""
        retriever = self.verctor_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                return "No related documents."
            formatted_str = ""
            for doc in docs:
                formatted_str += f"doc peices: {doc.page_content}\n metadata of docs:{doc.metadata}\n\n"
            return formatted_str
        
        def format_for_retriever(value: dict) ->str:
            return value["input"]
        
        def format_for_prompt_template(value):
            new_value={}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context":RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        convertion_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        return convertion_chain
    
if __name__ =="__main__":
    session_config = {
        "configurable":{
            "session_id":"锟哥"
        }
    }
    res = RagService().chain.invoke({"input":"我体重 160斤，推荐尺码"},session_config)
    print(res)