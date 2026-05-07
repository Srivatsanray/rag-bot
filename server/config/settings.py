import os
from dotenv import load_dotenv

load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") #template for setting
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEMPFILE_UPLOAD_DIRECTORY = "./temp/uploaded_files"
VECTORSTORE_DIRECTORY = os.getenv("VECTORSTORE_DIRECTORY", "./data/qdrant_store")

