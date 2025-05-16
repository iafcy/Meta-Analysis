from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from model import Bot
from prompt import genereate_char_prompt

def extract_characteristics(model: Bot, ma_title: str, columns: list[str], pdf_path: str):
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    index = VectorStoreIndex.from_documents(documents, llm=model.get_llamaindex_llm())
    query_engine = index.as_query_engine(verbose=True, similarity_top_k=8)

    prompt = genereate_char_prompt({ 'title': ma_title, 'columns': '|'.join(columns) })
    response = query_engine.query(prompt)

    sources = []
    for i, source_node in enumerate(response.source_nodes):
        sources.append(source_node.node.get_content())

    return response.response