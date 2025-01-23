import os
import chromadb
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.node_parser import Node


src_dir = "/content/codebase/src"
db_path = "/content/chroma_db"

def get_file_tree(directory):
    """
    Traverse the directory and return a structured file tree.
    Each node contains the path and type (file/directory).
    """
    file_tree = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            file_tree.append({
                "path": os.path.join(root, dir_name),
                "type": "directory"
            })
        for file_name in files:
            file_tree.append({
                "path": os.path.join(root, file_name),
                "type": "file"
            })
    return file_tree


file_tree = get_file_tree(src_dir)
print(f"Indexed {len(file_tree)} items in the file tree.")


nodes = [
    Node(
        text=f"{item['type'].capitalize()}: {item['path']}",
        metadata={"path": item["path"], "type": item["type"]}
    )
    for item in file_tree
]


Settings.embed_model = HuggingFaceEmbedding(
    model_name="thenlper/gte-base"
)

db = chromadb.PersistentClient(path=db_path)


chroma_collection = db.get_or_create_collection("file_tree_collection")


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


index = VectorStoreIndex(
    nodes, storage_context=storage_context
)
print("File tree index created and stored successfully.")
