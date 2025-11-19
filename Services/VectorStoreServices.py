from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List
from Config.settings import settings
from Services.EmbeddingServices import EmbeddingService
import os

class VectorStoreService:
    def __init__(self):
        # Get embeddings
        embedding_service = EmbeddingService()
        
        # Create directory for Chroma database if it doesn't exist
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Initialize Chroma (no Cassandra anymore!)
        self.vector_store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embedding_service.get_embeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        self.vector_store.add_documents(documents)
        self.vector_store.persist()
    
    def get_retriever(self):
        """Get retriever from vector store"""
        return self.vector_store.as_retriever()
    
    def similarity_search(self, query: str, k: int = 4):
        """Search for similar documents"""
        return self.vector_store.similarity_search(query, k=k)