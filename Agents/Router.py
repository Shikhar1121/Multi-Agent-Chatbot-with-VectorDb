from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from Config.settings import settings

class RouteQuery(BaseModel):
    """Route the user query to the most relevant datasource"""
    dataSource: Literal["vectorStore", "wikiSearch" , "tavilySearch"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or vectorstore or tavilySearch.",
    )

class QuestionRouter:
    def __init__(self):
        self.llm = ChatOpenAI(model=settings.LLM_MODEL)
        self.structured_llm = self.llm.with_structured_output(RouteQuery)
        
        system = """You are an expert at routing a user question to a vectorstore or wikipedia or tavilySearch.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

Use the vectorstore for questions on these topics. Otherwise, use wiki-search or tavilySearch for searching the web."""

        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
        ])
        
        self.chain = self.route_prompt | self.structured_llm
    
    def route(self, question: str) -> str:
        """Route question to appropriate datasource"""
        result = self.chain.invoke({"question": question})
        return result.dataSource