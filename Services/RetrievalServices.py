from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

class RetrievalService:
    def __init__(self):
        self.wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=200
        )
        self.wiki = WikipediaQueryRun(api_wrapper=self.wiki_wrapper)
    
    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia"""
        return self.wiki.invoke({"query": query})