from typing import List
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.documents import Document
from Agents.Router import QuestionRouter
from Services.VectorStoreServices import VectorStoreService
from Services.RetrievalServices import RetrievalService

class State(TypedDict):
    question: str
    generation: str
    documents: List[Document]

class RAGGraph:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service
        self.retrieval_service = RetrievalService()
        self.router = QuestionRouter()
        self.retriever = vector_store_service.get_retriever()
        self.app = self._build_graph()
    
    def _retrieve(self, state: State):
        """Retrieve from vector store"""
        print("-----Retrieving from Vector Store-----")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def _wiki_search(self, state: State):
        """Search Wikipedia"""
        print("-----Searching Wikipedia-----")
        question = state["question"]
        docs = self.retrieval_service.search_wikipedia(question)
        wiki_results = Document(page_content=docs)
        return {"documents": [wiki_results], "question": question}
    
    def _tavily_search(self, state: State):
        """Search Tavily"""
        print("-----Searching Tavily-----")
        question = state["question"]
        docs = self.retrieval_service.search_tavily(question)
        tavily_results = Document(page_content=docs)
        return {"documents": [tavily_results], "question": question}
    
    def _route_question(self, state: State):
        """Route question to appropriate source"""
        print("-----Routing Question-----")
        question = state["question"]
        source = self.router.route(question)
        
        if source == "wikiSearch":
            print("---Routing to Wikipedia---")
            return "wikiSearch"
        elif source == "vectorStore":
            print("---Routing to RAG---")
            return "vectorStore"
        elif source == "tavilySearch":
            print("---Routing to Tavily---")
            return "tavilySearch"
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("wikiSearch", self._wiki_search)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("tavilySearch", self._tavily_search)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            START,
            self._route_question,
            {
                "wikiSearch": "wikiSearch",
                "vectorStore": "retrieve" , 
                "tavilySearch": "tavilySearch"
            },
        )
        
        # Add edges to END
        workflow.add_edge("retrieve", END)
        workflow.add_edge("wikiSearch", END)
        workflow.add_edge("tavilySearch", END)
        
        return workflow.compile()
    
    def invoke(self, question: str):
        """Invoke the graph with a question"""
        result = self.app.invoke({"question": question})
        return result