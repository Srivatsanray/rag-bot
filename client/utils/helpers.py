from utils.api import (
    chat,
    get_vectorstore_collection_count,
    get_vectorstore_similarity_search,
    upload_and_process_pdfs,
)


def process_uploaded_pdfs(uploaded_files) -> str:
    return upload_and_process_pdfs(uploaded_files)


def process_user_input(user_input) -> str:
    return chat(user_input)


def get_documents_count() -> int:
    return get_vectorstore_collection_count()


def get_similar_chunks(query) -> list[dict]:
    return get_vectorstore_similarity_search(query)
