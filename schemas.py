"""
Database Schemas for Adaptive Topic Explorer

Each Pydantic model below corresponds to a MongoDB collection (lowercased class name).
- Topic -> "topic"
- Conversation -> "conversation"
- Suggestion -> "suggestion"
"""
from pydantic import BaseModel, Field
from typing import Optional, List

class Topic(BaseModel):
    name: str = Field(..., description="Topic name, e.g., Shipping")
    parent: Optional[str] = Field(None, description="Parent topic name if this is a subtopic")
    description: Optional[str] = Field(None, description="Optional description")

class Conversation(BaseModel):
    transcript: str = Field(..., description="Raw conversation text")
    topic: Optional[str] = Field(None, description="Assigned topic name")
    subtopic: Optional[str] = Field(None, description="Assigned subtopic name (child topic)")
    suggestion_id: Optional[str] = Field(None, description="Reference to suggestion if created")

class Suggestion(BaseModel):
    name: str = Field(..., description="Suggested topic or subtopic name")
    reason: Optional[str] = Field(None, description="Why this suggestion was created")
    parent: Optional[str] = Field(None, description="Optional parent topic name if this is a subtopic suggestion")
    status: str = Field("pending", description="pending | approved | rejected")
"""
Note: The Flames database viewer can read /schema. Our API will expose minimal schema metadata for reference.
"""
