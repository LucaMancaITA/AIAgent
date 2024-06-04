from llama_index.llms.langchain import LangChainLLM
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings

from langchain_community.llms import Ollama

from rag.utils.prompts import new_prompt, instruction_str, context
from rag.engines.note_engine import note_engine
from rag.engines.pdf_reader import pdf_engine
from rag.engines.pandas_reader import df_query_engine


# LLama2 LLM
llm = Ollama(model="codellama")
llm = LangChainLLM(llm)

# Tools
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

# Agent
agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True,
    context=context
)

while (prompt := input("Enter a prompt (q to quit)")) != "q":
    result = agent.query(prompt)
    print(result)
