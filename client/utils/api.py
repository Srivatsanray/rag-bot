from io import BytesIO

import requests
from utils.config import API_URL


def handle_response(response):
    try:
        json_data = response.json()
        if json_data["status"] == "success":
            return json_data.get("data")
        else:
            raise Exception(json_data.get("message", "Unknown error occurred."))
    except Exception as e:
        raise Exception(f"API Error: {str(e)}")


def get_vectorstore_collection_count() -> int:
    response = requests.get(f"{API_URL}/vector_store/count")
    return handle_response(response)


def get_vectorstore_similarity_search(query) -> list[dict]:
    payload = {"query": query}
    response = requests.post(f"{API_URL}/vector_store/search", json=payload)
    return handle_response(response)


def upload_and_process_pdfs(uploaded_files) -> str:
    files = []
    for file in uploaded_files:
        if hasattr(file, "data"):
            files.append(("files", (file.name, BytesIO(file.data), file.type)))
        else:
            files.append(("files", (file.name, file.read(), file.type)))

    # Send the POST request with multiple files
    response = requests.post(f"{API_URL}/upload", files=files)
    return handle_response(response)


def chat(user_input) -> str:
    payload = {"message": user_input}

    response = requests.post(f"{API_URL}/chat", json=payload)
    return handle_response(response)
