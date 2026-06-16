import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from rag_pipeline import EmbeddingManager, VectorStore, RAGRetriever, get_llm, rag_simple

# Initialize pipeline once on startup
print("Initializing RAG pipeline components for server...")
embedding_manager = EmbeddingManager()
vector_store = VectorStore()
retriever = RAGRetriever(vector_store, embedding_manager)

app = FastAPI(title="Flex RAG Chatbot Service")

# Add CORS middleware to support potential client connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=3, ge=1, le=10)
    score_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    model: Optional[str] = "llama-3.1-8b-instant"

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"Received query: '{request.message}' with model={request.model}, temp={request.temperature}, top_k={request.top_k}")
        # Initialize LLM dynamically based on request parameters
        llm = get_llm(model=request.model, temperature=request.temperature)
        
        # Run RAG
        result = rag_simple(
            query=request.message,
            retriever=retriever,
            llm=llm,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def status_endpoint():
    try:
        info = vector_store.get_collection_info()
        return {
            "status": "ready",
            "collection_name": info.get("collection_name"),
            "document_count": info.get("count", 0),
            "sources": info.get("sources", []),
            "embedding_model": embedding_manager.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database status error: {str(e)}")

# Mount static files directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return {"message": "Server is running, but index.html is missing inside static/ folder."}
    return FileResponse(index_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
