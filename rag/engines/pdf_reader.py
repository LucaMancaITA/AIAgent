import os
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.langchain import LangChainLLM
from langchain_community.llms import Ollama


# Codellama LLM
llm = Ollama(model="codellama")
llm = LangChainLLM(llm)
Settings.llm = llm

# LLM embedding model
hugging_face_llm_embedding = "BAAI/bge-small-en-v1.5"
embed_model = HuggingFaceEmbedding(model_name=hugging_face_llm_embedding)
Settings.embed_model = embed_model

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building_index", index_name)
        index = VectorStoreIndex.from_documents(
            data,
            embed_model=embed_model,
            show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name))

    return index


pdf_path = os.path.join("data", "document.pdf")
pdf_doc = PDFReader().load_data(file=pdf_path)
pdf_index = get_index(pdf_doc, "pdf")

pdf_engine = pdf_index.as_query_engine(llm=llm)
