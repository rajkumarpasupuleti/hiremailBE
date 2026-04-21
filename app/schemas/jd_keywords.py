from __future__ import annotations
from pydantic import BaseModel, Field






class JDKeywordRequest(BaseModel):
    job_description: str = Field(..., min_length=1, description="Raw JD text/string input")

    
class JDKeywordResponse(BaseModel):
    keywords: list[str]
    required_keywords: list[str]
    preferred_keywords: list[str]
    boolean_query: str