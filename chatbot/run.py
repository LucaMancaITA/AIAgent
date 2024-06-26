from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st

# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

# Streamlit framework
st.title('Langchain Demo With LLAMA2 API')
input_text=st.text_input("Search the topic u want")

# ollama model
llm = Ollama(model="codellama")

# Output parser
output_parser = StrOutputParser()

# Chain: prompt -> llm -> parser
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))