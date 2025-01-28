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
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

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

class MappingResponse(BaseModel):
    mapped_value: str = Field(description="The mapped CDE value that best matches the input response value")

class ResponseMapper:
    def __init__(self, cde_definitions: pd.DataFrame):
        self.cde_definitions = cde_definitions
        self.llm = ChatOllama(model="deepseek-r1")
        
        self.system_message = SystemMessage(content="""You are a response mapping system that matches semantically equivalent responses between scales.
            Your task is to determine if the meaning in ReproSchema matches any meaning in the CDE scale.""")
        
        self.prompt_template = """You are a response scale mapping expert. Your task is to find semantic equivalents between different ways of expressing the same level or degree.

            ReproSchema description: {response_text} (value={response_value})

            Available CDE mappings:
            {cde_mapping}

            Task: Find the CDE option that is semantically closest to the ReproSchema description.
            Consider:
            - Similar time periods (e.g., "less than a day" ≈ "once or twice")
            - Similar frequencies (e.g., "multiple times" ≈ "several times")
            - Similar intensities (e.g., "very intensely" ≈ "severe intensity")

            After your analysis, respond with ONLY a JSON object in this format: {{"mapped_value": "value"}}
            Where "value" is the matching CDE value or "-9" if no good match exists.

            Example responses:
            {{"mapped_value": "1"}}  # When a good match is found and the value is 1
            {{"mapped_value": "-9"}}  # When no semantic match exists"""

    def _get_response_text(self, response_value: Any, response_options: List[Dict]) -> Optional[str]:
        """Get the text description for a response value."""
        if not response_options:
            return None
            
        for option in response_options:
            if option.get('value') == response_value:
                # Extract the English text from the name dictionary
                name_dict = option.get('name', {})
                return name_dict.get('en')
        return None

    async def _find_best_match(self, response_value: Any, cde_mapping: Dict[str, str], response_options: Optional[List[Dict]]) -> str:
        """Find the best matching CDE value."""
        if response_value is None:
            return "-9"
            
        # Try exact value match first
        str_value = str(response_value)
        if str_value in cde_mapping:
            return str_value
        
        # Get the text meaning of this value in ReproSchema
        response_text = self._get_response_text(response_value, response_options)
        
        # Try direct text match
        if response_text and response_text in cde_mapping:
            return cde_mapping[response_text]
        
        # Use LLM for semantic matching
        try:
            prompt = self.prompt_template.format(
                response_value=response_value,
                response_text=response_text or "No description available",
                cde_mapping=json.dumps(cde_mapping, indent=2)
            )
            
            messages = [
                self.system_message,
                HumanMessage(content=prompt)
            ]
            
            result = await self.llm.ainvoke(messages)
            result_content = result.content.strip()
            
            # Extract JSON from the response
            try:
                # Try to find JSON-like string in the output
                import re
                json_match = re.search(r'\{.*\}', result_content)
                if json_match:
                    json_str = json_match.group()
                    json_data = json.loads(json_str)
                    mapped_value = json_data.get('mapped_value', '-9')
                    
                    # Validate mapped value
                    if mapped_value in cde_mapping.values() or mapped_value == "-9":
                        return mapped_value
                        
                print(f"Warning: Could not extract valid JSON from: {result_content}")
                return "-9"
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw LLM output: {result_content}")
                return "-9"
                
        except Exception as e:
            print(f"Error in LLM mapping: {e}")
            print(f"Context: value={response_value}, text={response_text}")
            return "-9"
    
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

    async def map_responses(self, reproschema_responses: List[Dict], matched_mapping: Dict[str, Tuple[str, float]]) -> Dict[str, str]:
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
            response_options = response.get("response_options")
            
            mapped_value = await self._find_best_match(response_value, cde_mapping, response_options)
            mapped_data[cde_element] = mapped_value
            
        return mapped_data

    def create_template_row(self, mapped_data: Dict[str, str], template_columns: List[str]) -> List[str]:
        """Create a row for the template using mapped data."""
        return [mapped_data.get(col, "-9") for col in template_columns]

# Example of how to use it in the mapping.py script:
async def map_schema(cde_csv_path: str, reproschema_response_path: str, cde_template_path: str):
    # Load CDE definitions and template
    cde_definitions = pd.read_csv(cde_csv_path)
    template_df = pd.read_csv(cde_template_path, header=1)
    template_columns = template_df.columns.tolist()

    # Load and extract ReproSchema responses
    with open(reproschema_response_path) as f:
        response_data = json.load(f)
    reproschema_responses = extract_reproschema_responses(response_data)

    # Get matches using semantic matcher
    semantic_matcher = SemanticMatcher()
    matched_mapping = semantic_matcher.match(
        cde_definitions=cde_definitions,
        reproschema_responses=reproschema_responses
    )

    # Map responses using enhanced mapper
    response_mapper = ResponseMapper(cde_definitions)
    mapped_data = await response_mapper.map_responses(reproschema_responses, matched_mapping)

    # Create template row
    template_row = response_mapper.create_template_row(mapped_data, template_columns)

    # Create and save output DataFrame
    df = pd.DataFrame([template_row], columns=template_columns)
    df.to_csv("output_template.csv", index=False)
    return df