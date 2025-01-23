from flask import Flask, request, jsonify
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.llms.groq import Groq

# Set up the Groq LLM
llm = Groq(model="gemma2-9b-it", api_key="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")

# Configure LlamaIndex to use Groq
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-base")

# Initialize client
db = chromadb.PersistentClient(path="./chroma_db")

# Get collection
chroma_collection = db.get_or_create_collection("quickstart")

# Assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load your index from stored vectors
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# Create a query engine
query_engine = index.as_query_engine()

# Flask API
app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    response = query_engine.query(question)
    return jsonify({"response": str(response)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
