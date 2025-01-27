# src/rs2nda/models.py
from pydantic import BaseModel
from typing import List, Dict, Optional

class CDEDefinition(BaseModel):
    ElementName: str
    ElementDescription: str
    Notes: Optional[str]
    ValueRange: Optional[str]

class ReproSchemaResponse(BaseModel):
    id: str
    question: str
    response_value: str
    response_options: Optional[List[Dict[str, str]]]

class MatchedMapping(BaseModel):
    cde_element: str
    repro_id: Optional[str]