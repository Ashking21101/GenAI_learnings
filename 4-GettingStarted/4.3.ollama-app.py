import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st 
#from langchain_community import Ollama
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ['LANCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT") 

llm = Ollama(model="gemma:2b")

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant. Please answer the question based on the context provided"),
        ("human", "{input}"),
    ]
)


# streamlit 
st.title("langchain app")
input_text = st.text_input("Enter your question")


# output parser
output_parser = StrOutputParser()
# create the chain
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"input": input_text}))
    

