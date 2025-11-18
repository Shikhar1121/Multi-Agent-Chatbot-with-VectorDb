import cassio
from dotenv import load_dotenv
import os
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.vectorstores.cassandra import Cassandra
from langchain_classic.indexes.vectorstore import VectorStoreIndexWrapper
from typing import Literal , List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel , Field
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing_extensions import TypedDict
from langchain_classic.schema import Document
from langgraph.graph import START , END , StateGraph

load_dotenv()
appToken=os.environ["ASTRA_DB_APPLICATION_TOKEN"] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
dbId = os.environ["ASTRA_DB_ID"] = os.getenv("ASTRA_DB_ID")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

cassio.init(token= appToken , database_id=dbId)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-1lm/",
]


## load url
docs = [WebBaseLoader(url).load() for url in urls]
docList = [item for sublist in docs for item in sublist]
textSplitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 500 , chunk_overlap = 0)
splitDocs= textSplitter.split_documents(docList)



embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")



astraVectorStore = Cassandra(embedding= embeddings, table_name= "Chatbot with VectorDb" , session=None , keyspace=None )

astraVectorStore.add_documents(splitDocs)

astraVectorStoreIndex = VectorStoreIndexWrapper(vectorstore=astraVectorStore)

retriever = astraVectorStore.as_retriever()


##Data model
class RouteQuery(BaseModel):
    """
    Route the user query to the most relevant datasource
    """

    dataSource : Literal["vectorStore" ,"WikiSearch" ] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or vectorstore.",
    )
llm = ChatOpenAI(model="gpt-5.1-2025-11-13")

structuredLlmRouter =  llm.with_structured_output(RouteQuery)

# Prompt

system = """You are an expert at routing a user question to a vectorstore or wikipedia.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""

routePrompt = ChatPromptTemplate.from_messages(

    [

    ("system", system),

    ("human", "{question}"),

    ]

)

questionRouter = routePrompt| structuredLlmRouter


wikiApiWrapper = WikipediaAPIWrapper(top_k_results=1 , doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wikiApiWrapper)

class State(TypedDict):
    question :str
    generation : str
    documents : List[str]

def retrieve(state : State):
    print("-----Retrieving-----")
    questions = state["question"]

    documents = retriever.invoke(questions)
    return {"documents" : documents , "questions" : questions}

def wikiSearch(state:State):
    print("----Wikipedia----")
    question = state["question"]

    docs= wiki.invoke({"query": question})
    wikiResults = docs
    wikiResults = Document(page_content = wikiResults)
    return {"documents" : wikiResults , "questions" : question}

def RouteQuestion(state : State):
    print("-----Route Question-----")
    question = state["question"]
    source = questionRouter.invoke({"question" : question})
    if source.datasource == "WikiSearch":
        print("---Routing to Wikipedia---")
        return "WikiSearch"
    elif source.datasource == "vectorStore":
        print("---Routing to RAG---")
        return "vectorStore"
    

workflow = StateGraph(StateGraph)

workflow.add_node("WikiSearch" , wikiSearch)
workflow.add_node("retrieve" , retrieve )

workflow.add_conditional_edges(
    START , 
    RouteQuestion , 
    {
        "wikiSearch" : "wikiSearch", 
        "vectorStore" : "retrieve"
    },

)

workflow.add_edge("retrieve" , END)
workflow.add_edge("wikiSearch" , END)

app = workflow.compile()


