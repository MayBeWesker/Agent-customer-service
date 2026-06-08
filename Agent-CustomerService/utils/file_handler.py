import os
import hashlib
from xml.dom.minidom import Document
from logger_handler import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 获取文件的 md5 的十六进制字符串
def get_file_md5_hex(filepath:str):
    if not os.path.exists(filepath):
        logger.error(f"[md5 计算] {filepath} does not exist")
        return 
    if not os.path.isfile(filepath):
        logger.error(f"[md5 计算] {filepath} is not a file")
        return 
    
    md5_obj = hashlib.md5()

    chunk_size = 4096
    try:
        with open(filepath,'rb') as f:
            while chunk :=f.read(chunk_size):
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"Computing document {filepath} error {str(e)}")
        return None


# 返回文件夹内的文件列表
def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] {path} is not a dir")
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)

def pdf_loader(filepath:str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath,passwd).load()

def txt_loader(filepath:str) -> list[Document]:
    return TextLoader(filepath,encoding='utf-8').load()