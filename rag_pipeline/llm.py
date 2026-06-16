import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import List, Dict, Any, Optional

# Load environment variables from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
load_dotenv(os.path.join(project_root, ".env"))

def get_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.1) -> ChatGroq:
    """Initialize and return a ChatGroq LLM instance"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback to os.environ if .env load failed
        api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Please set it in your .env file or environment.")
    
    return ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=1024
    )

def rag_simple(query: str, retriever: Any, llm: Any, top_k: int = 3, score_threshold: float = 0.0) -> Dict[str, Any]:
    """
    Retrieve documents and generate response using the LLM.
    
    Args:
        query: User input query
        retriever: RAGRetriever instance
        llm: ChatGroq instance
        top_k: Top K passages to fetch
        score_threshold: Minimum similarity threshold
        
    Returns:
        Dict containing:
            - 'answer': generated text answer
            - 'sources': list of retrieved chunks with metadata and scores
    """
    # Retrieve context
    results = retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    if not context:
        return {
            "answer": "I'm sorry, I couldn't find any relevant context in the uploaded documents to answer your question.",
            "sources": []
        }
    
    # Prompt template for ChatGroq
    prompt = f"""Use the following context to answer the question. If the context does not contain relevant information, state that you cannot find the answer in the provided documents. Do not make up facts.
    
    Context:
    {context}

    Question: {query}

    Answer:"""
    
    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
        
    return {
        "answer": answer,
        "sources": results
    }
