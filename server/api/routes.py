import asyncio

from api.schemas import ChatRequest, SearchQueryRequest, StandardAPIResponse
from core.llm_chain_factory import generate_answer
from core.vector_database import (
    find_similar_chunks,
    get_collection_count,
    upsert_vectorstore_from_pdfs,
    vectorstore_exists,
)
from fastapi import APIRouter, File, HTTPException, UploadFile
from utils.logger import logger

router = APIRouter()


@router.get("/health", response_model=StandardAPIResponse)
async def health_check():
    logger.debug("Health check requested")
    return StandardAPIResponse(
        status="success", data="ok", message="Service is healthy"
    )


@router.post("/upload", response_model=StandardAPIResponse)
async def upload_and_process_pdfs(files: list[UploadFile] = File(...)):
    try:
        logger.info(f"Received {len(files)} file(s) for processing")
        await upsert_vectorstore_from_pdfs(files)  # still async — save_uploaded_file inside is async
        count = get_collection_count()  # sync now, call directly
        return StandardAPIResponse(
            status="success",
            data={"chunks_in_store": count},
            message=f"Successfully processed {len(files)} file(s).",
        )
    except ValueError as e:
        logger.warning(f"Validation error during upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during upload")
        raise HTTPException(status_code=500, detail="Failed to process uploaded files.")


@router.get("/vector_store/count", response_model=StandardAPIResponse)
async def get_vectorstore_count():
    try:
        count = get_collection_count()  # sync now, call directly
        return StandardAPIResponse(status="success", data={"chunk_count": count})
    except Exception as e:
        logger.exception("Error getting collection count")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vector_store/search", response_model=StandardAPIResponse)
async def search_vectorstore(request: SearchQueryRequest):
    # Debug endpoint — useful for inspecting retrieval quality without invoking the LLM.
    # Passes the raw query as both hyde_query and original_query since HyDE is not
    # needed for manual retrieval inspection.
    try:
        if not vectorstore_exists():  # sync now, call directly
            raise HTTPException(status_code=400, detail="No documents uploaded yet.")
        chunks = find_similar_chunks(request.query, request.query)  # sync now, call directly
        return StandardAPIResponse(status="success", data=chunks)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during similarity search")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=StandardAPIResponse)
async def chat(request: ChatRequest):
    try:
        if not vectorstore_exists():  # sync now, call directly
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded yet. Please upload a PDF first.",
            )

        result = await asyncio.to_thread(generate_answer, request.message)  # sync fn, offload to thread

        return StandardAPIResponse(
            status="success",
            data={
                "answer": result["answer"],
                "chunks": result["chunks"],
                "low_confidence": result["low_confidence"],
            },
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"ValueError in chat: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Chat endpoint encountered an error")
        raise HTTPException(status_code=500, detail="Failed to generate answer.")
