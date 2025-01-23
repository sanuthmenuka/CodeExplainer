# config.py

# Path to the source code directory
SRC_DIR = "/content/codebase/src"

# Path to ChromaDB storage
DB_PATH = "/content/chroma_db"

# HuggingFace model name for embeddings
EMBEDDING_MODEL = "thenlper/gte-base"

# Chunking settings for CodeSplitter
CHUNK_LINES = 40
CHUNK_LINES_OVERLAP = 15
MAX_CHARS = 1500
