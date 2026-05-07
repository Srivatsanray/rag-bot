# RAG Bot (FastAPI + Streamlit)

This working project is to built a citation based RAG Bot. The core idea is to use overlap fixed-chunking with Sparse + Dense embed with cross-encoder rerank. LLM Output utilized hard coded broad and naive query classification to route use of multi-query + HyDE approach.

## Installation

```bash
git clone https://github.com/Srivatsanray/rag-bot/
cd rag-bot-fastapi
```

Setup Virtual Environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install frontend:

```bash
cd client
pip3 install -r requirements.txt
```

Install backend:

```bash
cd ../server
pip3 install -r requirements.txt
```

## API Keys Required

- **Groq API key** from [console.groq.com](https://console.groq.com/)

Create a `.env` file:

```env
GROQ_API_KEY=your-groq-key
```

## Run the Bot

Start FastAPI backend:

```bash
# Terminal 1
cd server
uvicorn main:app --reload
```

Start Streamlit frontend:

```bash
# Terminal 2
cd client
streamlit run app.py
```

# Project Structure 

```bash
rag-bot/
├── client/                         # Streamlit Frontend
│   ├── app.py                      # Main Streamlit entrypoint
│   ├── components/                 # UI modules
│   │   ├── chat.py
│   │   ├── inspector.py
│   │   └── sidebar.py
│   ├── state/
│   │   └── session.py              # Session state manager
│   ├── utils/
│   │   ├── api.py                  # Talks to backend
│   │   ├── config.py               # API_URL and config values
│   │   └── helpers.py              # API wrappers for frontend
│   ├── requirements.txt
│   └── README.md

├── server/                         # FastAPI Backend
│   ├── api/
│   │   ├── routes.py               # API endpoints
│   │   └── schemas.py              # Pydantic schemas for I/O
│   ├── core/
│   │   ├── document_processor.py   # Handles PDF validation and chunking
│   │   ├── llm_chain_factory.py    # Builds LLM output
│   │   └── vector_database.py      # Embeddings + Qdrant + Rerank
│   ├── config/
│   │   └── settings.py             # App config, model provider setup
│   ├── utils/
│   │   └── logger.py               # Logging setup
│   ├── main.py                     # FastAPI app entrypoint
│   ├── requirements.txt
│   └── README.md

├── README.md
├── .gitignore
```

---
