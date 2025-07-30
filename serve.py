from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="gemma-2b", groq_api_key=groq_api_key)

# 1. create a prompt template
system_template = "Translate the following into {Language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}"),
])

# 2. create an output parser
parser = StrOutputParser()

# 3. create a chain
chain = prompt_template | model | parser

# 4. create a FastAPI app
app = FastAPI(title = "Langchain server",
              version = "1.0",
              description = "A simple Langchain server using Groq",)

# 5. add chain routes to the app
add_routes(app,
           chain,
           path = "/chain")

# 6. add a custom route for the root path and execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
