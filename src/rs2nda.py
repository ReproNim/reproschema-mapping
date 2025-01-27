"""
ReproSchema to NDA (National Data Archive) converter.
Provides functionality to map ReproSchema responses to CDE (Common Data Elements) format.
"""

import requests
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# Models
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

# Utilities
_cache = {}

def fetch_item_content(url: str) -> Dict:
    """Fetch JSON content from a URL and cache it."""
    if url not in _cache:
        response = requests.get(url)
        response.raise_for_status()
        _cache[url] = response.json()
    return _cache[url]

def get_constraints_url(is_about_url: str, response_options: Optional[str]) -> Optional[str]:
    """Construct the full URL for response options (constraints) based on the isAbout URL."""
    if not response_options or not isinstance(response_options, str):
        return None
    
    url_parts = is_about_url.split("/")
    if "items" not in url_parts:
        return None
    
    items_index = url_parts.index("items")
    base_parts = url_parts[:items_index]
    
    if response_options.startswith("../"):
        response_options = response_options[3:]
    
    return "/".join(base_parts + [response_options])

def extract_reproschema_responses(response_jsonld: List[Dict]) -> List[Dict]:
    """Extract ReproSchema responses with question details and constraints."""
    responses = []
    for entry in response_jsonld:
        if entry.get("@type") == "reproschema:Response":
            item_url = entry["isAbout"]
            item_content = fetch_item_content(item_url)
            response_options = item_content.get("responseOptions")
            options_url = get_constraints_url(item_url, response_options)
            options_content = fetch_item_content(options_url) if options_url else None
            
            choices = options_content.get("choices") if options_content else None
            responses.append({
                'id': item_content["id"],
                'question': item_content["question"]["en"],
                'response_value': entry["value"],
                'response_options': choices
            })
    return responses

def clean_string(text: str) -> str:
    """Clean and normalize a string."""
    return " ".join(text.split()).strip().lower()

def validate_url(url: str) -> bool:
    """Check if a URL is valid."""
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Core Classes
class SemanticMatcher:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def match(self, cde_definitions: pd.DataFrame, reproschema_responses: List[Dict], threshold=0.5):
        """Match CDE definitions to ReproSchema questions using semantic similarity."""
        cde_descriptions = cde_definitions["ElementDescription"].tolist()
        cde_names = cde_definitions["ElementName"].tolist()
        
        reproschema_questions = [r["question"] for r in reproschema_responses]
        reproschema_ids = [r["id"] for r in reproschema_responses]

        cde_embeddings = self.model.encode(cde_descriptions, convert_to_tensor=True)
        repro_embeddings = self.model.encode(reproschema_questions, convert_to_tensor=True)

        similarity_matrix = util.cos_sim(cde_embeddings, repro_embeddings)
        similarity_matrix = similarity_matrix.cpu().numpy()

        matches = {}
        for i, cde_name in enumerate(cde_names):
            best_match_idx = np.argmax(similarity_matrix[i])
            best_score = similarity_matrix[i][best_match_idx]
            
            if best_score >= threshold:
                matches[cde_name] = (reproschema_ids[best_match_idx], float(best_score))
            else:
                matches[cde_name] = None

        return matches

class ResponseMapper:
    def __init__(self, cde_definitions: pd.DataFrame):
        self.cde_definitions = cde_definitions
        
    def _parse_cde_notes(self, notes: str) -> Dict[str, str]:
        """Parse CDE notes string into a mapping dictionary."""
        if pd.isna(notes):
            return {}
            
        mapping = {}
        parts = notes.split(';')
        for part in parts:
            part = part.strip()
            if '=' in part:
                value, name = part.split('=', 1)
                value = value.strip()
                name = name.strip()
                mapping[name] = value
                mapping[value] = value
        return mapping

    def _find_best_match(self, response_value: Any, cde_mapping: Dict[str, str]) -> str:
        """Find the best matching CDE value for a given ReproSchema response value."""
        if response_value is None:
            return "-9"
            
        if str(response_value) in cde_mapping:
            return str(response_value)
            
        response_str = str(response_value)
        
        if response_str in cde_mapping:
            return cde_mapping[response_str]
            
        for cde_name, cde_value in cde_mapping.items():
            if response_str.lower() == cde_name.lower():
                return cde_value
                
        return "-9"

    def map_responses(self, reproschema_responses: List[Dict], matched_mapping: Dict[str, Tuple[str, float]]) -> Dict[str, str]:
        """Map ReproSchema response values to CDE values."""
        mapped_data = {}
        repro_lookup = {r["id"]: r for r in reproschema_responses}
        
        for cde_element, match_info in matched_mapping.items():
            if match_info is None:
                mapped_data[cde_element] = "-9"
                continue
                
            repro_id, _ = match_info
            response = repro_lookup.get(repro_id)
            
            if response is None:
                mapped_data[cde_element] = "-9"
                continue
                
            cde_row = self.cde_definitions[self.cde_definitions["ElementName"] == cde_element]
            if cde_row.empty:
                mapped_data[cde_element] = "-9"
                continue
                
            cde_notes = cde_row["Notes"].iloc[0]
            cde_mapping = self._parse_cde_notes(cde_notes)
            
            response_value = response.get("response_value")
            mapped_value = self._find_best_match(response_value, cde_mapping)
            mapped_data[cde_element] = mapped_value
            
        return mapped_data

    def create_template_row(self, mapped_data: Dict[str, str], template_columns: List[str]) -> List[str]:
        """Create a row for the template using mapped data."""
        return [mapped_data.get(col, "-9") for col in template_columns]