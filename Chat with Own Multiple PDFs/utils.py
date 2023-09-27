from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import time
import os
from langchain.llms import OpenAI

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxx"

#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_docs(user_pdf_list):
    docs=[]
    for filename in user_pdf_list:        
        chunks=get_pdf_text(filename)       
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"type=":filename.type,"size":filename.size},
        ))

    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


#Function to push data to Vector Store - Pinecone here
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )
    print("done......2")
    Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)
    
    


#Function to pull info from Vector Store 
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )
    index_name = pinecone_index_name  
    index = Pinecone.from_existing_index(index_name, embeddings)
    return index


#Function to help us get relavant documents from vector store
def similar_docs(query,pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    time.sleep(5)
    similar_docs = index.similarity_search(query)
    #print(similar_docs)
    return similar_docs


# get the summary of a document
def get_answer(pinecone_apikey,pinecone_environment,index_name,embeddings,query):
   
    docsearch = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    docs = docsearch.similarity_search(query)

    llm = HuggingFaceHub(repo_id = "google/flan-t5-large")
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})    

    chain=load_qa_chain(llm,chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query) 

    return answer


    