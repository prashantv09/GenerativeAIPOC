import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid
 

def main():
    st.session_state['unique_id'] =''
    load_dotenv()

    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance...")
   
    job_description = st.text_area("JOB DESCRIPTION here...",key="1")
    document_count = st.text_input("No.of 'RESUMES' to return",key="2")

    # Upload the Resumes (pdf files)
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)

    submit=st.button("Get Matching CV")

    if submit:
        with st.spinner('Wait ...'):
            
            st.session_state['unique_id']=uuid.uuid4().hex
            
            final_docs_list=create_docs(pdf,st.session_state['unique_id'])
           
            st.write("*Resumes uploaded* :"+str(len(final_docs_list)))
            
            embeddings=create_embeddings_load_data()
          
            push_to_pinecone("PINCONEAPI","PINCONEENV","PINECONEINDEX",embeddings,final_docs_list)

            relavant_docs=similar_docs(job_description,document_count,"PINCONEAPI","PINCONEENV","PINECONEINDEX",embeddings,st.session_state['unique_id'])
           
            st.write(":heavy_minus_sign:" * 30)

            
            for item in range(len(relavant_docs)):                
                st.subheader(str(item+1))            
                st.write("**CV Name** : "+relavant_docs[item][0].metadata['name'])
            
                with st.expander('Expand...' ): 
                    st.info("*Match Score** : "+str(relavant_docs[item][1]))                    
                    summary = get_summary(relavant_docs[item][0])
                    st.write("**Summary** : "+summary)

        st.success("Task Done!!")


#Invoking main function
if __name__ == '__main__':
    main()
