from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv

load_dotenv()

# LLM model
llm = Ollama(model="llama2", request_timeout=30.0)

# Code LLM model
code_llm = Ollama(model="codellama", request_timeout=30.0)

# Instantiate a parser
parser = LlamaParse(result_type="markdown")

# Parse the PDF files
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# Embedding model
embed_model = resolve_embed_model("local:BAAI/bge-m3")

# Instantiate the Vector Index
vector_index = VectorStoreIndex.from_documents(
    documents=documents,
    embed_model=embed_model
)

# Create the query engine
pdf_engine = vector_index.as_query_engine(llm=llm)

# Tools
tools = [
    QueryEngineTool(
        query_engine=pdf_engine,
        metadata=ToolMetadata(
            name="documentation",
            description="Reading pdf docs"
        )
    )
]

# Context
context = """
    Purpose: The primary role of this agent is to assist users by analyzing
    code. It should be able to generate code and answer questions about code
    provided."""

# ReAct agent: to integrate reasoning and action capabilities
agent = ReActAgent.from_tools(
    tools,
    llm=code_llm,
    verbose=True,
    context=context
)

result = pdf_engine.query("Which languages are spoken in Canada?")
print(result)