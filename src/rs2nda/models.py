from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class RSQuestion(BaseModel):
    """ReproSchema Question Model"""
    id: str
    question: Dict[str, str]  # language -> question text
    category: str = "Item"
    ui: Dict[str, Any]
    response_options: str
    pref_label: Optional[Dict[str, str]] = Field(None, alias="prefLabel")
    schema_version: str = Field(..., alias="schemaVersion")

class NDAQuestion(BaseModel):
    """NDA Question Model"""
    element_name: str
    data_type: str
    size: Optional[float]
    required: str
    element_description: str
    value_range: str
    notes: Optional[str]
    aliases: Optional[str]

class QuestionMapping(BaseModel):
    """Mapping between RS and NDA questions"""
    rs_id: str
    nda_id: str
    confidence: float  # Confidence score of the mapping
    method: str  # How this mapping was determined (e.g., "id_match", "text_similarity")
