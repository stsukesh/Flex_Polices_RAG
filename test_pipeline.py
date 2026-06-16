import os
from rag_pipeline import EmbeddingManager, VectorStore, RAGRetriever, get_llm, rag_simple

def main():
    print("Testing modular RAG pipeline...")
    
    # 1. Initialize embedding manager
    try:
        embedding_manager = EmbeddingManager()
        print("Embedding Manager successfully initialized.")
    except Exception as e:
        print(f"FAILED to initialize Embedding Manager: {e}")
        return
        
    # 2. Initialize vector store
    try:
        vector_store = VectorStore()
        print("Vector Store successfully initialized.")
        info = vector_store.get_collection_info()
        print(f"Collection Info: Name='{info['collection_name']}', Count={info['count']}, Sources={info['sources']}")
    except Exception as e:
        print(f"FAILED to initialize Vector Store: {e}")
        return
        
    # 3. Initialize retriever
    try:
        retriever = RAGRetriever(vector_store, embedding_manager)
        print("Retriever successfully initialized.")
    except Exception as e:
        print(f"FAILED to initialize Retriever: {e}")
        return
        
    # 4. Initialize LLM
    try:
        llm = get_llm(temperature=0.1)
        print("LLM (ChatGroq) successfully initialized.")
    except Exception as e:
        print(f"FAILED to initialize LLM: {e}")
        return
        
    # 5. Run test query
    query = "What about working Hours in Flex"
    print(f"\nRunning test query: '{query}'")
    try:
        res = rag_simple(query, retriever, llm, top_k=2)
        print(f"Answer:\n{res['answer']}\n")
        print("Retrieved Chunks:")
        for idx, doc in enumerate(res['sources']):
            source = doc['metadata'].get('source', 'unknown')
            score = doc['similarity_score']
            print(f"  [{idx+1}] Source: {source} (Score: {score:.4f})")
            print(f"      Content: {doc['content'][:150]}...")
        print("\nPipeline check completed successfully!")
    except Exception as e:
        print(f"FAILED to execute RAG pipeline query: {e}")

if __name__ == "__main__":
    main()
