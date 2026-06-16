import os
import uuid
import chromadb
import numpy as np
from typing import List, Any, Dict

# Resolve default persist directory relative to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DEFAULT_PERSIST_DIR = os.path.join(project_root, "data", "vector_store")

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = DEFAULT_PERSIST_DIR):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get details about the collection for frontend visibility"""
        try:
            count = self.collection.count()
            if count == 0:
                return {
                    "collection_name": self.collection_name,
                    "count": 0,
                    "sources": []
                }
            
            # Fetch all metadatas to extract unique sources
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            sources = set()
            for meta in metadatas:
                if meta:
                    source_path = meta.get("source", meta.get("file_path", "Unknown"))
                    if source_path:
                        filename = os.path.basename(source_path)
                        sources.add(filename)
            
            return {
                "collection_name": self.collection_name,
                "count": count,
                "sources": sorted(list(sources))
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "count": 0,
                "sources": [],
                "error": str(e)
            }
