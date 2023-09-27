import streamlit as st
from langchain.llms import HuggingFaceHub
from utils import *
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import time
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxx"
PINCONE_API_KEY = "xxxxxxx"
PINCONE_API_ENV = "xxxxxx"

st.set_page_config(page_title="Chat with your PDFs")
st.title("Chat with your Own PDF Files...")

pdf = st.file_uploader("Upload PDF File", type=["pdf"],accept_multiple_files=True)

our_query = st.text_input("Query: ",key=1)
submit=st.button("Get Answer",key=2)

if submit :
    with st.spinner('Wait for it...'):
        data=create_docs(pdf)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
        docs = text_splitter.split_documents(data)   
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # pinecone.init(
        #     api_key=PINCONE_API_KEY, 
        #     environment=PINCONE_API_ENV
        # )
        # index_name = "langchainpinecone"
        # if index_name not in pinecone.list_indexes():
        #     pinecone.create_index(index_name,dimension=384,metric='cosine',pods=1,pod_type='p1.x2')
        #     docserch = Pinecone.from_texts([t.page_content for t in docs],embeddings,index_name=index_name)    
        
     
        # relavant_docs=similar_docs(our_query,"PinconeAPI","PinconeEnvName","pineconeIndexname",embeddings)        
        answer=get_answer(PINCONE_API_KEY,PINCONE_API_ENV,"pineconeIndexname",embeddings,our_query)
        st.write(answer)

