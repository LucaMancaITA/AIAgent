import os
import pandas as pd

from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.langchain import LangChainLLM
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings

from langchain_community.llms import Ollama

from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from pdf import pdf_engine


# Read the dataframe
csv_path = os.path.join("data", "dataframe.csv")
dataframe = pd.read_csv(csv_path)

# LLama2 LLM
llm = Ollama(model="codellama")
llm = LangChainLLM(llm)
Settings.llm = llm

# Pandas query engine
df_query_engine = PandasQueryEngine(
    df=dataframe,
    #llm=llm,
    verbose=True,
    instruction_str=instruction_str
)
df_query_engine.update_prompts(
    {"pandas_prompt": new_prompt}
)

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=df_query_engine,
        metadata=ToolMetadata(
            name="csv_data",
            description="this gives information a the world populations"
        )),
    QueryEngineTool(
        query_engine=pdf_engine,
        metadata=ToolMetadata(
            name="pdf_data",
            description="this gives information about Canada the country"
        ))
]

agent = ReActAgent.from_tools(
    tools,
    #llm=llm,
    verbose=True,
    context=context
)

while (prompt := input("Enter a prompt (q to quit)")) != "q":
    result = agent.query(prompt)
    print(result)
