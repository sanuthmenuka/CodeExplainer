import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter
from tree_sitter import Language, Parser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


src_dir = "/content/codebase/src"
db_path = "/content/chroma_db"


language_configs = {
    "java": {
        "exts": [".java"],
        "tree_sitter_lang": "tree-sitter-java",
    },
    "python": {
        "exts": [".py"],
        "tree_sitter_lang": "tree-sitter-python",
    },
    "csharp": {
        "exts": [".cs"],
        "tree_sitter_lang": "tree-sitter-c-sharp",
    },
}


for lang_name, config in language_configs.items():
    print(f"Processing {lang_name} files...")
    
   
    Language.build_library(
        'build/my-languages.so',
        [config["tree_sitter_lang"]]
    )
    LANG = Language('build/my-languages.so', lang_name)
    parser = Parser()
    parser.set_language(LANG)
    
    
    reader = SimpleDirectoryReader(
        input_dir=src_dir,
        required_exts=config["exts"],
        recursive=True,
    )
    docs = reader.load_data()
    print(f"Loaded {len(docs)} {lang_name} documents.")
    
    
    splitter = CodeSplitter(
        language=lang_name,
        chunk_lines=40,  # Lines per chunk
        chunk_lines_overlap=15,  # Lines overlap between chunks
        max_chars=1500,  # Max chars per chunk
        parser=parser
    )
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"Generated {len(nodes)} nodes for {lang_name}.")
    
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="thenlper/gte-base"
    )
    
    
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(f"{lang_name}_collection")
    
   
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    
    index = VectorStoreIndex(
        nodes, storage_context=storage_context
    )
    print(f"Index for {lang_name} created and stored.")

print("All languages processed successfully!")
