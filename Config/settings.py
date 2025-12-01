import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    
    # Set environment variables
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
    if TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    
    # Set USER_AGENT to avoid warnings
    os.environ.setdefault('USER_AGENT', 'RAG-Chatbot/1.0')
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-5-mini-2025-08-07"
    
    # Chroma settings
    CHROMA_PERSIST_DIR = "./chroma_db"
    COLLECTION_NAME = "rag_chatbot"
    
    # Document processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 0

    # Tavily settings
    TAVILY_MAX_RESULTS = 5


settings = Settings()