# 假设你使用的是适用于文本的类 Document
from langchain.schema import Document
from FlagEmbedding import BGEM3FlagModel
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch

chunk_size=50
chunk_overlap=15

# 初始化模型（确保使用GPU）
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 读取文件并处理每一行
with open('all.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 将每一行转换为 Document 对象
documents = [Document(page_content=line.strip()) for line in lines]


EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(model_name='bge_large_zh_v1.5',model_kwargs={'device': EMBEDDING_DEVICE}, encode_kwargs={'normalize_embeddings': True})

vectordb = Chroma(embedding_function=embeddings, collection_name="lance", collection_metadata={"hnsw:space":"cosine"})

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# 创建内存存储对象,存储每个被切割的小文档块所属的原始文档及其索引Id，
store = InMemoryStore()

# #创建父文档检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    child_splitter=text_splitter,
    search_kwargs={"k":1},
    search_type="similarity",
)

retriever.add_documents(documents, ids=None)

while True:
    query = input("请输入查询语句：")
    if query == "e":
        break
    ans = retriever.get_relevant_documents(query)
    print(ans)
