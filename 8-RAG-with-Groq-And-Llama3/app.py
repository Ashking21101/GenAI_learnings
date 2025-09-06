# point of using Groq is to reduce the cost of using OpenAI models and also to have a faster response time.


import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


from dotenv import load_dotenv
load_dotenv()

# env variables
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")   OR
# groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


#llm = ChatGroq(model="Llama3-8b-8192", api_key=groq_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key)
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context provided only.
    please provide most accurate and concise answer based on the question.
    <context>
    {context}
    <context>
    Question:{input}
    """
)



# creating vector store
def create_vector_stores():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.loader = PyPDFDirectoryLoader(r"D:\GenAI\1-Langchain\8-RAG-with-Groq-And-Llama3") # data injestion
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)# we can limit to 50 documents for fast purpose by st.session_state.docs[:50]
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


user_input = st.text_input("Enter your question from the research paper")

if st.button("Document embeddings and store in vector DB"):
    create_vector_stores()
    st.write("Vector store created successfully!")


# calculating respone time
import time
if user_input:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input":user_input})
    print(f"Response Time: {time.process_time() - start}")# syntax

    st.write(response["answer"])
    
    # with streamlit expander
    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------------------")




# Flow of your code

# User input → goes into the retrieval chain as {"input": user_input}.
# Retriever (st.session_state.vectors.as_retriever())
# Based on user_input, it searches your FAISS vector store.
# Returns the top-k most relevant documents.
# create_retrieval_chain
# Automatically takes those retrieved docs.
# It fills the {context} placeholder in your ChatPromptTemplate with the concatenated content of those docs.
# LLM Call
# The filled-in prompt (question + context) is sent to the LLM.


# That’s why you don’t need to manually pass context in invoke.
# The retriever handles fetching docs → the chain injects them → prompt gets completed.
