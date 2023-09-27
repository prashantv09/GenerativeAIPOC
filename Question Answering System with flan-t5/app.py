import os
import streamlit as st
from langchain.llms import HuggingFaceHub
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxx"

st.set_page_config(page_title="Question Answering System")
st.title="Question Answering System"


our_query = st.text_input("Ask Question: ",key=1)
submit=st.button("Get Answer",key=2)

if submit:
    llm = HuggingFaceHub(repo_id = "google/flan-t5-large")
    st.write(llm(our_query))

