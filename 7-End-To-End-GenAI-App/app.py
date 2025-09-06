import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()


# langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PORJECT"] = "Q&A Chatbot with App"


# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers users questions and search all internet to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer."),
        ("user", "Question:{question}"),# {question} is a variable
    ]
)



# temperature = close to 0(just on point)
#               close to 1(more creative)
def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question":question})# "question" variable from above used
    return answer



# title of app
st.title("Q&A Chatbot")

# sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# drop down for model selection
llm = st.sidebar.selectbox("select model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# user input
st.write("Ask a question and get an answer!")
user_input = st.text_input("Ask a question")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get an answer.")

