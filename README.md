# GenAI_learnings
2 things are also required

Virtual Environment
1. create a folder ABC --> create a virtual ENV in it (conda create -p ./genv python==3.10)
2. then in ABC folder and create requirements.txt(write all your required libraries) and create all your python/ipynb code file
3. then in terminal activate your ENV (conda activate ./genv)
4. install (pip install -r requiremnets.txt)


Dot Env(.env) file
1. create a (.evn) file in ABC folder
2. and in this file mention all the API key
  OPENAI_API_KEY    =  *key*
  LANGCHAIN_API_KEY =  *key*  (thsi is fr langsmith trackig)
  LANGCHAIN_PROJECT =  "GenAIappwithOPENAI" (langchain project will keep track of all the prompt and charges)
  HF_TOKEN          =  *key*  (huggingface)
  GROQ_API_KEY      =  *key*
