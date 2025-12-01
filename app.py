import streamlit as st
import os
import traceback

# Set USER_AGENT
os.environ['USER_AGENT'] = 'RAG-Chatbot/1.1'

from Agents.Graph import RAGGraph
from Services.VectorStoreServices import VectorStoreService
from Config.settings import settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Enable debug mode
DEBUG = False

def log_debug(message):
    """Print debug messages"""
    if DEBUG:
        st.sidebar.write(f"🔍 {message}")
        print(f"DEBUG: {message}")

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)
##st.secrets["OPENAI_API_KEY"]
##st.secrets["LANGSMITH_API_KEY"]
##st.secrets["HF_TOKEN"]
##st.secrets["TAVILY_API_KEY"]
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "graph" not in st.session_state:
    st.session_state.graph = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Initialize components
@st.cache_resource
def initialize_system():
    """Initialize vector store and graph"""
    try:
        log_debug("Loading existing Chroma database...")
        
        # Check if database exists
        if not os.path.exists(settings.CHROMA_PERSIST_DIR):
            return None, None, "Database not found. Please run: python setup.py"
        
        # Initialize vector store (loads from disk)
        vector_store = VectorStoreService()
        log_debug("Vector store loaded from disk")
        
        # Initialize graph
        graph = RAGGraph(vector_store)
        log_debug("RAG graph initialized")
        
        return vector_store, graph, None
        
    except Exception as e:
        error_msg = f"Initialization error: {str(e)}\n{traceback.format_exc()}"
        return None, None, error_msg

# Generate response
def generate_response(question: str, documents):
    """Generate response using LLM with retrieved documents"""
    try:
        log_debug(f"Generating response for: {question}")
        
        llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0.7)
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Answer the question based on the context provided. 
            If the context doesn't contain relevant information, say so clearly.
            
            Context: {context}"""),
            ("human", "{question}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        
        return response.content
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# UI
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about AI agents, prompt engineering, and adversarial attacks!")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses:
    - **Vector Store (Chroma)**: For questions about agents, prompt engineering, adversarial attacks
    - **Wikipedia**: For general knowledge questions
    - **Tavily**: For current events and news
    
    The system automatically routes your question to the best source.
    """)
    
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Debug toggle
    debug_enabled = st.checkbox("Show Debug Info", value=DEBUG)
    if debug_enabled:
        DEBUG = True
        st.markdown("### Debug Info")
        st.write(f"Messages: {len(st.session_state.messages)}")
        st.write(f"Initialized: {st.session_state.initialized}")
        st.write(f"DB exists: {os.path.exists(settings.CHROMA_PERSIST_DIR)}")
    
    st.markdown("---")
    st.markdown("### Settings")
    st.write(f"🤖 Model: {settings.LLM_MODEL}")
    st.write(f"📊 Vector DB: Chroma")
    st.write(f"📁 Location: {settings.CHROMA_PERSIST_DIR}")

# Initialize system
if not st.session_state.initialized:
    with st.spinner("🔄 Loading vector database..."):
        vector_store, graph, error = initialize_system()
        
        if error:
            st.error(f"❌ {error}")
            
            if "Database not found" in error:
                st.info("""
                ### First Time Setup Required
                
                Run this command to initialize the database:
                ```bash
                python setup.py
                ```
                
                This will:
                1. Load documents from URLs
                2. Split them into chunks
                3. Create embeddings
                4. Save to Chroma database
                """)
            st.stop()
        else:
            st.session_state.vector_store = vector_store
            st.session_state.graph = graph
            st.session_state.initialized = True
            st.success("✅ System ready!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            st.caption(f"📚 Source: {message['source']}")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Get documents from graph
                result = st.session_state.graph.invoke(prompt)
                documents = result.get("documents", [])
                
                # Determine source
                if len(documents) > 0:
                    if hasattr(documents[0], 'metadata') and documents[0].metadata:
                        source = "Vector Store"
                    elif "wikipedia.org" in documents[0].page_content:
                        source = "Wikipedia"
                    elif "tavily" in documents[0].page_content:
                        source = "Tavily"
                else:
                    source = "Unknown"
                
                # Generate response
                response = generate_response(prompt, documents)
                
                st.markdown(response)
                st.caption(f"📚 Source: {source}")
                
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "source": source
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                if DEBUG:
                    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with LangChain, LangGraph, Chroma and Streamlit</div>",
    unsafe_allow_html=True
)