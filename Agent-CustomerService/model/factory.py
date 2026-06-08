from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_config

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        raise NotImplementedError


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> [Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_config["chat_model_name"])
    
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> [Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])
    

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
