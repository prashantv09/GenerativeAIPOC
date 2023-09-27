from langchain.llms import OpenAI
import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from apikey import apikey
os.environ["OPENAI_API_KEY"] = apikey

def app():
    st.title("CSV Query App")
    st.write("Upload a CSV file and ask your  query")
    file = st.file_uploader("Upload your file",type={"csv"})
    if not file:
        st.stop
    
    if file is not None:
        data = pd.read_csv(file)
        st.write("Data Preview")
        st.dataframe(data.head())
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data,verbose=True)
        query = st.text_input("Enter a query ")

        if st.button("Execute"):
            answer = agent.run(query)
            st.write(answer)
        

if __name__ == "__main__" :
    app()