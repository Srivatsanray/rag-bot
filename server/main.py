import uvicorn
from fastapi import FastAPI
from api.routes import router
from core.vector_database import initialize_vectorstore
from utils.logger import logger

app = FastAPI(title="RAG PDFBot", description="Chat with multiple PDFs")
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up...")
    initialize_vectorstore()
    logger.info("Startup complete.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
