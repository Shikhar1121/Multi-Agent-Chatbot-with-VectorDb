# 🤖 RAG Chatbot with Vector Database

An intelligent chatbot that combines Retrieval-Augmented Generation (RAG) with smart routing to answer questions from a custom knowledge base and Wikipedia or Tavily Search for web results. Built with LangChain, LangGraph, Chroma, and Streamlit.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.9-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)

## 🌟 Features

- **Smart Question Routing**: Automatically routes questions to the most relevant data source
- **Dual Knowledge Base**: 
  - Vector database for specialized topics (AI agents, prompt engineering, adversarial attacks)
  - Wikipedia integration for general knowledge
  - Tavily Search for web results.
- **Interactive UI**: Clean, modern Streamlit interface with chat history
- **Source Attribution**: Every response is tagged with its data source
- **Local Vector Storage**: Uses Chroma for efficient, local vector storage
- **Agentic Workflow**: Built with LangGraph for complex routing logic

## 🏗️ Architecture

```
User Question → LLM Router → [Vector Store] or [Wikipedia] or [Tavily Search]
                                    ↓              ↓                 ↓ 
                              Retrieved Docs   Wiki Content     Web Content
                                    ↓              ↓                ↓
                                              LLM Generator
                                                   ↓
                                              Final Response
```

### Tech Stack

- **LangChain**: Orchestration framework for LLM applications
- **LangGraph**: State machine for agentic workflows
- **Chroma**: Vector database for semantic search
- **OpenAI GPT-4**: Language model for routing and generation
- **HuggingFace Embeddings**: Text embeddings for vector similarity
- **Streamlit**: Web interface
- **Wikipedia API**: External knowledge source
- **Tavily API**: External web knowledge source


## 📁 Project Structure

```
Multi-Agent-Chatbot-with-VectorDb/
├── app.py                      # Streamlit application
├── setup.py                    # Database initialization script
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── config/
│   └── settings.py            # Configuration settings
├── data/
│   └── urls.py                # Data source URLs
├── services/
│   ├── embedding_service.py   # Embedding generation
│   ├── vector_store_service.py # Vector database operations
│   └── retrieval_service.py   # Wikipedia or Tavily Search retrieval
├── agents/
│   ├── router.py              # Question routing logic
│   └── graph.py               # LangGraph workflow
├── utils/
│   └── document_loader.py     # Document loading utilities
└── chroma_db/                 # Vector database storage (created on setup)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Internet connection for initial setup

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shikhar1121/Multi-Agent-Chatbot-with-VectorDb.git
cd Multi-Agent-Chatbot-with-VectorDb
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirement.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here  # Optional
HF_TOKEN=your_huggingface_token_here       # Optional
Tavily_API_KEY your Tavily api key
```

5. **Initialize the vector database**
```bash
python setup.py
```

This will:
- Load documents from configured URLs
- Split them into chunks
- Generate embeddings
- Store in Chroma vector database

6. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💡 Usage

1. **Ask questions about the specialized topics:**
   - "What is prompt engineering?"
   - "Explain AI agents and their components"
   - "What are adversarial attacks on language models?"
   - "Latest AI news today"

2. **Ask general knowledge questions:**
   - "Who is Albert Einstein?"
   - "What is the capital of France?"
   - "Current weather in Paris"

The system automatically routes each question to the appropriate source and provides attributed responses.

## 🎯 Key Components

### Question Router
Uses gpt-5-mini with structured output to classify questions and route them to:
- **Vector Store**: For questions about AI agents, prompt engineering, adversarial attacks
- **Wikipedia**: For general knowledge queries
- **Tavily**: For Web Search.

### Vector Store Service
- Manages Chroma vector database
- Handles document ingestion and retrieval
- Provides semantic similarity search

### RAG Graph
LangGraph-based workflow that:
1. Routes questions to appropriate data source
2. Retrieves relevant documents
3. Generates context-aware responses

## 🔧 Configuration

Edit `config/settings.py` to customize:

```python
# Model settings
LLM_MODEL = "gpt-5-mini"                    # Change to gpt-5-mini for lower cost
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # HuggingFace embedding model

# Database settings
CHROMA_PERSIST_DIR = "./chroma_db"     # Vector database location
COLLECTION_NAME = "rag_chatbot"        # Collection name

# Document processing
CHUNK_SIZE = 500                       # Text chunk size
CHUNK_OVERLAP = 0                      # Overlap between chunks
```

## 📊 Data Sources

Default sources (configurable in `data/urls.py`):
- Lilian Weng's blog posts on:
  - AI Agents
  - Prompt Engineering
  - Adversarial Attacks on LLMs

To add your own sources, update `data/urls.py` and run `python setup.py` again.

## 🐛 Troubleshooting

### Common Issues

**Import errors:**
```bash
pip install -r requirement.txt --force-reinstall
```

**Database not found:**
```bash
python setup.py
```

**OpenAI API errors:**
- Check your API key in `.env`
- Verify billing is set up on OpenAI dashboard

**Slow responses:**
- Switch to `gpt-5-mini` in settings.py
- Reduce chunk size for faster retrieval

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the amazing framework
- [Streamlit](https://streamlit.io/) for the intuitive UI framework
- [Chroma](https://www.trychroma.com/) for the vector database
- [Lilian Weng](https://lilianweng.github.io/) for the excellent blog posts used as knowledge base

## 📧 Contact

Your Name - Shikhar Srivastava

Project Link: [https://github.com/Shikhar1121/Multi-Agent-Chatbot-with-VectorDb.git](https://github.com/Shikhar1121/Multi-Agent-Chatbot-with-VectorDb.git)

---

⭐ If you found this project helpful, please consider giving it a star!