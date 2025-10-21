import pdfplumber
import docx2txt
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document

def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif uploaded_file.name.endswith(".docx"):
            text = docx2txt.process(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
        else:
            text = ""
        documents.append(Document(page_content=text))
    return documents

def create_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(split_docs, embeddings)

def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
