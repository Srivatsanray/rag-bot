# RAG BOT - Server

This is the FastAPI backend for the RAG PDFBot. It handles PDF processing, vectorstore embedding, LLM chain execution, and API endpoints.

---

## Project Structure

```
server/
├── api/                        # FastAPI routes and schemas
├── config/                     # Environment and constants
├── core/                       # LLM logic, vectorstore, processing
├── utils/                      # Logger and helpers
├── main.py                     # App entry point
```

---

## Installation

1. **Clone the repo**

```bash
git clone
cd
```

2. **Create a virtual environment (optional)**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
cd server

pip3 install -r requirements.txt
```

---

## Configuration

Set your API keys in `config/settings.py`:

---

## Usage

Run the app:

```bash
cd rag-challenge/server

uvicorn main:app --reload
```

---

## API Endpoints

- `/upload`
- `/chat`
- `/vector_store/count/{provider}`
- `/vector_store/search`
- `/health`

## Logging

Logs are printed to the console and controlled via `utils/logger.py`.
