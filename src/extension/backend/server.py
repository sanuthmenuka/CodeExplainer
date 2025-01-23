from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.llms.groq import Groq

# Initialize FastAPI app
app = FastAPI()

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

# Request model
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    response = query_engine.query(request.question)
    return {"response": str(response)}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
