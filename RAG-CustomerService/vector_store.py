from langchain_chroma import Chroma
import config_data as config

class VectorStoreService():
    def __init__(self, embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )
    
    def get_retriever(self):
        """返回向量检测器，方便加入 chain """
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})
    

if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()
    r = retriever.invoke("我的体重 160 斤，尺码推荐什么")
    print(r)

