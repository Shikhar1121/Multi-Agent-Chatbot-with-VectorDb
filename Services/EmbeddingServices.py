from langchain_huggingface import HuggingFaceEmbeddings
from Config.settings import settings

class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )
        return cls._instance
    
    def get_embeddings(self):
        return self.embeddings