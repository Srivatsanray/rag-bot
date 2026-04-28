from pydantic import BaseModel
from typing import Any, Optional, Literal


class SearchQueryRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    message: str


class StandardAPIResponse(BaseModel):
    status: Literal["success", "error"]
    data: Optional[Any] = None
    message: Optional[str] = None
