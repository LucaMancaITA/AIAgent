import os
import pandas as pd
from llama_index.llms.langchain import LangChainLLM
from langchain_community.llms import Ollama
from llama_index.experimental.query_engine import PandasQueryEngine
from utils.prompts import new_prompt, instruction_str, context


# LLama2 LLM
llm = Ollama(model="codellama")
llm = LangChainLLM(llm)

# Read the dataframe
csv_path = os.path.join("data", "dataframe.csv")
dataframe = pd.read_csv(csv_path)

# Pandas query engine
df_query_engine = PandasQueryEngine(
    df=dataframe,
    llm=llm,
    verbose=True,
    instruction_str=instruction_str
)
df_query_engine.update_prompts(
    {"pandas_prompt": new_prompt}
)
