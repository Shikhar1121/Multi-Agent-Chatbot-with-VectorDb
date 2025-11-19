"""
Run this script once to initialize the vector database with documents
Command: python setup.py
"""
import os

# Set USER_AGENT to avoid warning
os.environ['USER_AGENT'] = 'RAG-Chatbot/1.0'

from Config.settings import settings
from Utils.DoumentLoader import DocumentLoader
from Services.VectorStoreServices import VectorStoreService
from Data.Urls import URLS

def setup():
    print("="*60)
    print("RAG CHATBOT SETUP - Using Chroma Vector Database")
    print("="*60)
    print(f"\nChroma DB location: {settings.CHROMA_PERSIST_DIR}")
    
    # Load and split documents
    print(f"\n[1/3] Loading documents from {len(URLS)} URLs...")
    try:
        loader = DocumentLoader(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        split_docs = loader.load_and_split(URLS)
        print(f"      ✓ Loaded and split {len(split_docs)} document chunks")
    except Exception as e:
        print(f"      ❌ Document loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize vector store
    print("\n[2/3] Initializing Chroma vector store...")
    try:
        vector_store = VectorStoreService()
        print("      ✓ Vector store initialized")
    except Exception as e:
        print(f"      ❌ Vector store initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Add documents
    print("\n[3/3] Adding documents to vector store...")
    try:
        vector_store.add_documents(split_docs)
        print("      ✓ Documents added and persisted successfully")
    except Exception as e:
        print(f"      ❌ Failed to add documents: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print(f"\n📁 Database location: {settings.CHROMA_PERSIST_DIR}")
    print(f"📦 Collection name: {settings.COLLECTION_NAME}")
    print(f"📄 Document chunks: {len(split_docs)}")
    print("\n🚀 Run the app with: streamlit run app.py")
    print("="*60)

if __name__ == "__main__":
    setup()