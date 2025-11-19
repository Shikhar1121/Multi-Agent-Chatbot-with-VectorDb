from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 0):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_urls(self, urls: List[str]) -> List[Document]:
        """Load documents from URLs"""
        docs = [WebBaseLoader(url).load() for url in urls]
        doc_list = [item for sublist in docs for item in sublist]
        return doc_list
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def load_and_split(self, urls: List[str]) -> List[Document]:
        """Load from URLs and split in one go"""
        docs = self.load_urls(urls)
        return self.split_documents(docs)