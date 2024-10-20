import time

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import re


def save_faiss_db(txt_path, faiss_dir):
    text_splitter = CharacterTextSplitter(
        separator = r'\d+\.',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )
    with open(txt_path) as f:
        real_estate_sales = f.read()
    docs = text_splitter.create_documents([real_estate_sales])
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local(faiss_dir)

if __name__ == '__main__':
    # 家装向量数据库
    save_faiss_db("home_decoration_sales_data.txt", "home_decoration_sales")
    save_faiss_db("real_estate_sales_data.txt", "real_estate_sales")


    db = FAISS.load_local("home_decoration_sales", OpenAIEmbeddings(), allow_dangerous_deserialization=True)



    docs = db.similarity_search("贵不贵")
    # 输出 Faiss 中最相似结果
    print(docs[0].page_content)

