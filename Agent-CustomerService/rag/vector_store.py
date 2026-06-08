import os
import sys

from langchain_chroma import Chroma

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_ROOT = os.path.dirname(CURRENT_DIR)
UTILS_DIR = os.path.join(AGENT_ROOT, "utils")

for path in (AGENT_ROOT, UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.path_tool import get_abs_path
from utils.config_handler import chroma_config
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from langchain_core.documents import Document

class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_config["persist_directory"]
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap = chroma_config["chunk_overlap"],
            separators = chroma_config["seperators"],
            length_function=len
        )
    
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k" : chroma_config['k']})

    def load_document(self):
        """
        从数据文件夹没读取数据文件，转化为向量存入向量库
        要计算的文件进行 md5 去重复
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                open(get_abs_path(chroma_config["md5_hex_store"]), "w", encoding='utf-8').close()
                return False
            with open(get_abs_path(chroma_config["md5_hex_store"]), "r", encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False
        
        def save_md5_hex(md5_for_check):
            with open(get_abs_path(chroma_config["md5_hex_store"]), "a", encoding='utf-8') as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []
        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"])
            )
        
        for path in allowed_files_path:
            # 获取文件的 md5
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[Loading knowledge base] {path} is now in the knowledge base")
                continue
            try :
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[Loading knowledge base] no validate contents in {path}")
                    continue
                split_document = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[Loading knowledge base] after spliter, no validate contents in {path}")
                    continue
                #将合法的内容存斤向量库中
                self.vector_store.add_documents(split_document)

                # 记录，避免之后重复加载
                save_md5_hex(md5_hex)
                logger.info(f"[Loading knowledge base] {path} loading sucessfully")
            except Exception as e:
                # exc_info=True 会记录详细的堆栈
                logger.error(f"[Loading knowledge base] {path} loading error: {str(e)}", exc_info=True)
                continue

if __name__ =='__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("="*20)
