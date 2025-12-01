from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from Config.settings import settings

class RetrievalService:
    def __init__(self):
        self.wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=200
        )
        self.wiki = WikipediaQueryRun(api_wrapper=self.wiki_wrapper)

        self.tavily_search = TavilySearchResults(
            max_results= settings.TAVILY_MAX_RESULTS,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
    
    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia"""
        return self.wiki.invoke({"query": query})
    
    def search_tavily(self, query: str) -> str:
        """
        Search the web using Tavily for current/news information
        """
        try:
            results = self.tavily_search.invoke({"query": query})
            
            if not results:
                return "No results found."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No content')
                url = result.get('url', '')
                
                formatted_results.append(
                    f"Result {i}:\n"
                    f"Title: {title}\n"
                    f"Content: {content}\n"
                    f"URL: {url}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error during Tavily search: {str(e)}"