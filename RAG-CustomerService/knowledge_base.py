"""
知识库
"""

import os
import config_data as config
import hashlib
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def check_md5(md5_str: str):
    """检查传入的 md5 是否已经被处理过了"""
    if not os.path.exists(config.md5_path):
        # 文件还没有处理过，我们创建这个文件
        open(config.md5_path, "w", encoding='utf-8').close()
        return False
    else:
        # 文件已经存在
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip() # 处理字符串前后的空格和回车
            if line == md5_str: # 已经处理过这个文件
                return True
        return False
    


def save_md5(md5_str):
    """将传入的 md5 字符串，记录到文件中进行保存"""
    with open(config.md5_path, 'a',encoding='utf-8') as f:
        f.write(md5_str + '\n')



def get_string_md5(input_str:str, encoding='utf-8'):
    """将传入的字符串转化为 md5 字符串"""
    # 将字符串转化为 bytes 数组
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest() # md5 的十六进制字符串

    return md5_hex


class KnowledgeBaseService(object):

    def __init__(self):
        # 文件夹不存在就创建出来
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name, # 数据库的表名
            embedding_function=DashScopeEmbeddings(model='text-embedding-v4'),
            persist_directory=config.persist_directory #数据库本地存储文件夹
        ) # 向量存储的实例 Chroma 向量库对象

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, # 分割后文本段最大长度
            chunk_overlap=config.chunk_overlap, # 连续文本段之间字符重叠数量
            separators=config.separators, # 自然段落划分的符号
            length_function=len # 使用python自带的len 函数做长度的统计依据
        ) # 文本分割器对象

    def upload_by_str(self, data: str, file_nmae):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        # 先拿到 md5 值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "内容中已经存在知识库中"
        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source":file_nmae,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "锟哥"
        }
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )

        save_md5(md5_hex)

        return "upload successfully"





if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰伦","test_file")
    print(r)
