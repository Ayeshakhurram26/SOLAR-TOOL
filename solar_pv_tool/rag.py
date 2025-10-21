from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def load_documents(file_paths):
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif path.endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())
    return docs

def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

def create_rag_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

