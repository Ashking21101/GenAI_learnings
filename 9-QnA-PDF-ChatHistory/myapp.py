# note we need to give different session_id to demonstrate chat history across sessions
# also note that the chat history is stored in the memory of the streamlit app. so if the app is restarted, the chat history is lost.
# so for production, you need to store the chat history in a persistent storage like a database


import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import MessagesPlaceholder


import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 1. streamlit app
st.title("Conversational RAG with PDF uploads(with Chat History)")
st.write("Upload a PDF and chat with it. The chat history is maintained across interactions.")

# 2. input API
api_key = st.text_input("Enter your Groq API Key", type="password")
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    
    # chat interface
    session_id = st.text_input("Session_ID", value="default_session")
    
    # manage Chat History
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")







    # Issues with this is You’re overwriting ./temp.pdf each time in the loop → only the last file actually survives in disk.
    # process PDF and saving in the temporary directory 
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:  # we are designing to upload multiple files at once
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:         
                file.write(uploaded_file.getvalue()) # getting all content
                file_name = uploaded_file.name       # reading name of file

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)


        # text splitter and creating embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        

        contextualize_q_system_prompt = (
            """give a chat history and the latest user question,
            which might reference content from the chat history,
            rephrase the user question to be a standalone question.
            without the chat history do not answer the question.
            just rephrase it if needed and otherwise return the question as is."""
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # create_history_aware_retriever function brings the history
        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt,
        )



        # Question Answer prompt
        system_prompt = (
            """You are a helpful AI assistant that helps people find information
            from a set of context documents. Use the following pieces of context to
            answer the question at the end. If you don't know the answer, just say
            that you don't know, don't try to make up an answer. Always be polite and
            professional.
            \n\n
            {context}"""
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )



        # not just a normal chain. it is STUFF cause it takes all the documents and also it can take other chains
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt       
        )



        # You would typically use create_stuff_documents_chain within a create_retrieval_chain setup.
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )



        # setting up chat history store
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Rag Chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, 
            get_session_history,
            input_messages_key="input",
            history_messages_key = "chat_history",
            output_messages_key="answer"
        )
        # get_session_history – Function that returns a new BaseChatMessageHistory. This function should either take a single positional argument session_id of type string and return a corresponding chat message history instance.
        # input_messages_key – Must be specified if the base runnable accepts a dict as input. The key in the input dict that contains the messages.
        # output_messages_key – Must be specified if the base Runnable returns a dict as output. The key in the output dict that contains the messages.
        # history_messages_key – Must be specified if the base runnable accepts a dict as input and expects a separate key for historical messages.
        # history_factory_config – Configure fields that should be passed to the chat history factory. See ConfigurableFieldSpec for more details
 









        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )

            # display chat history
            # st.write(st.session_state.store) important to understand just
            st.success(f"Assistant: {response['answer']}")
            # st.write("Chat History:", session_history.messages)  important to understand just




    else:
        st.warning("Please enter your Groq API key ")