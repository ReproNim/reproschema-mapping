"""
ReproSchema to NDA (National Data Archive) converter.
Provides functionality to map ReproSchema responses to CDE (Common Data Elements) format.
"""
import aiohttp
import asyncio
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import requests
import torch
from langchain_ollama import ChatOllama
from langchain.schema.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MappingConfig:
    """Configuration settings for mapping process"""
    similarity_threshold: float = 0.5
    batch_size: int = 10
    cache_size: int = 1000
    missing_numeric: str = "-9"
    llm_model: str = "deepseek-r1"

config = MappingConfig()

# Custom Exceptions
class MappingError(Exception):
    """Base exception for mapping errors"""
    pass

class DataValidationError(MappingError):
    """Exception for data validation errors"""
    pass

# Pydantic Models
class CDEDefinition(BaseModel):
    """Model for CDE definition data"""
    ElementName: str
    DataType: str
    ElementDescription: str
    Notes: Optional[str] = None
    ValueRange: Optional[str] = None
    Aliases: Optional[str] = None

class ReproSchemaResponse(BaseModel):
    """Model for ReproSchema response data"""
    id: str
    question: str
    response_value: Any
    isAbout: str = ""  # Add this field
    response_options: Optional[List[Dict[str, Any]]] = None

class CDEMapping(BaseModel):
    """Model for CDE mapping data"""
    numeric_to_string: Dict[str, str] = Field(default_factory=dict)
    string_to_numeric: Dict[str, str] = Field(default_factory=dict)
    valid_values: Set[str] = Field(default_factory=set)
    value_range: Optional[str] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None

    def is_valid_value(self, value: str) -> bool:
        # Special case for numeric ranges
        if self.min_value is not None and self.max_value is not None:
            try:
                val = int(float(value))
                return self.min_value <= val <= self.max_value
            except (ValueError, TypeError):
                pass
        return value in self.valid_values

class MappingResult(BaseModel):
    """Model for mapping result data"""
    cde_element: str
    repro_id: str
    confidence: float

class URLCache:
    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self.max_size = max_size
        self._session = None
    
    async def _get_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def fetch_item_content(self, url: str) -> Dict:
        if url in self._cache:
            return self._cache[url]
            
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                text = await response.text()
                content = json.loads(text)
                
                if len(self._cache) >= self.max_size:
                    # Remove oldest item
                    self._cache.pop(next(iter(self._cache)))
                    
                self._cache[url] = content
                return content
                
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            raise
    
    async def close(self):
        if self._session:
            await self._session.close()

url_cache = URLCache()

def get_constraints_url(is_about_url: str, response_options: Optional[str]) -> Optional[str]:
    """
    Construct constraints URL from isAbout URL and response options path
    
    Args:
        is_about_url: URL of the item content
        response_options: Path to response options (e.g. "../valueConstraintsFirst19")
        
    Returns:
        Full URL to the constraints content or None if cannot be constructed
    """
    if not response_options or not isinstance(response_options, str):
        return None
    
    try:
        # Split URL into parts
        url_parts = is_about_url.split("/")
        
        # Find the items directory in the path
        if "items" not in url_parts:
            return None
            
        items_index = url_parts.index("items")
        base_parts = url_parts[:items_index]
        
        # Handle relative paths
        options_path = response_options
        if options_path.startswith("../"):
            options_path = options_path[3:]  # Remove ../ prefix
            
        # Construct full URL
        constraints_url = "/".join(base_parts + [options_path])
        logger.debug(f"Constructed constraints URL: {constraints_url}")
        return constraints_url
        
    except Exception as e:
        logger.error(f"Error constructing constraints URL: {str(e)}")
        return None

async def extract_reproschema_responses(response_jsonld: List[Dict]) -> List[Dict]:
    """Extract responses from ReproSchema JSON-LD including questions and response options"""
    logger = logging.getLogger(__name__)
    responses = []
    
    for entry in response_jsonld:
        try:
            if entry.get("@type") == "reproschema:Response":
                logger.debug(f"Processing response entry: {json.dumps(entry, indent=2)}")
                
                # Get the isAbout URL
                is_about = entry.get("isAbout", "")
                logger.info(f"Found isAbout URL: {is_about}")
                
                # Fetch content from isAbout URL to get ID and question
                response_id = ""
                question = ""
                response_options = None
                
                if is_about:
                    try:
                        item_content = await url_cache.fetch_item_content(is_about)
                        # Get ID from the isAbout content
                        response_id = item_content.get("id", "")
                        question = item_content.get("question", {}).get("en", "")
                        logger.debug(f"Found response ID: {response_id}")
                        logger.debug(f"Found question: {question}")
                        
                        # First try to get response options directly from item content
                        direct_options = item_content.get("responseOptions", {})
                        if direct_options:
                            direct_choices = direct_options.get("choices", [])
                            if direct_choices:
                                logger.debug("Found response options directly in item content")
                                response_options = direct_choices
                            else:
                                logger.debug("No direct choices found in responseOptions")
                        
                        # If no direct options found, try getting from URL
                        if not response_options:
                            response_options_path = item_content.get("responseOptions")
                            if isinstance(response_options_path, str):
                                logger.debug(f"Found response options path: {response_options_path}")
                                constraints_url = get_constraints_url(is_about, response_options_path)
                                if constraints_url:
                                    try:
                                        constraints_content = await url_cache.fetch_item_content(constraints_url)
                                        choices = constraints_content.get("choices", [])
                                        if choices:
                                            logger.debug("Found response options from constraints URL")
                                            response_options = choices
                                    except Exception as e:
                                        logger.error(f"Error fetching constraints content: {str(e)}")
                                else:
                                    logger.warning("Could not construct constraints URL")
                            else:
                                logger.debug("Response options path is not a string")
                        
                        if response_options:
                            logger.debug(f"Final response options: {json.dumps(response_options, indent=2)}")
                        else:
                            logger.warning(f"No response options found for {response_id}")
                            
                    except Exception as e:
                        logger.error(f"Error fetching item content: {str(e)}")
                
                # Construct response
                response = ReproSchemaResponse(
                    id=response_id, 
                    question=question,
                    response_value=entry.get("value"),
                    response_options=response_options,
                    isAbout=is_about
                )
                
                responses.append(response.model_dump())
                logger.debug(f"Added response: {json.dumps(response.model_dump(), indent=2)}")
                
        except Exception as e:
            logger.error(f"Error extracting response: {str(e)}")
            continue
            
    return responses

def clean_string(text: str) -> str:
    """Clean and normalize text string"""
    return " ".join(text.split()).strip().lower()

# Question Matching
class QuestionMatcher:
    """Matches ReproSchema questions to CDE elements using multiple strategies"""

    _model = None
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.logger = logging.getLogger(__name__)
        if QuestionMatcher._model is None:
            self.logger.info("Initializing SentenceTransformer model")
            QuestionMatcher._model = SentenceTransformer(model_name)
        self.model = QuestionMatcher._model
        self.llm = ChatOllama(model=config.llm_model)
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get cached embedding or compute new one"""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.model.encode(text, convert_to_tensor=True)
        return self._embedding_cache[text]

    def _exact_id_match(self, reproschema_ids: List[str], cde_names: List[str]) -> Tuple[Dict[str, Tuple[str, float]], Set[str]]:
        """Find exact matches between ReproSchema IDs and CDE names"""
        matches = {}
        used_repro_ids = set()
        
        try:
            for cde_name in cde_names:
                if cde_name in reproschema_ids:
                    matches[cde_name] = (cde_name, 1.0)  # Perfect match
                    used_repro_ids.add(cde_name)
                    
            return matches, used_repro_ids
            
        except Exception as e:
            logger.error(f"Error in exact ID matching: {str(e)}")
            return {}, set()

    def _alias_match(self, reproschema_ids: List[str], cde_definitions: pd.DataFrame, 
                    used_repro_ids: Set[str]) -> Tuple[Dict[str, Tuple[str, float]], Set[str]]:
        """Find matches through aliases"""
        matches = {}
        newly_used_ids = set()

        try:
            # Process rows with non-null Aliases
            alias_rows = cde_definitions[cde_definitions['Aliases'].notna()]
            for _, row in alias_rows.iterrows():
                element_name = row['ElementName']
                aliases = set(alias.strip() for alias in str(row['Aliases']).split(','))
                
                # Find matching reproschema ID
                for repro_id in reproschema_ids:
                    if repro_id in used_repro_ids:
                        continue
                        
                    if repro_id in aliases:
                        matches[element_name] = (repro_id, 1.0)
                        newly_used_ids.add(repro_id)
                        break

            return matches, newly_used_ids
            
        except Exception as e:
            logger.error(f"Error in alias matching: {str(e)}")
            return {}, set()

    def _similarity_match(self, cde_definitions: pd.DataFrame, reproschema_responses: List[Dict],
                         used_repro_ids: Set[str], threshold: float = 0.5) -> Dict[str, Tuple[str, float]]:
        """Find matches based on semantic similarity"""
        try:
            # Filter out already matched responses
            remaining_responses = [r for r in reproschema_responses if r["id"] not in used_repro_ids]
            
            if not remaining_responses:
                return {}

            # Prepare texts for comparison
            cde_descriptions = cde_definitions["ElementDescription"].tolist()
            cde_names = cde_definitions["ElementName"].tolist()
            reproschema_questions = [r["question"] for r in remaining_responses]
            reproschema_ids = [r["id"] for r in remaining_responses]

            # Compute embeddings and similarities in batches
            matches = {}
            batch_size = config.batch_size
            
            for i in range(0, len(cde_descriptions), batch_size):
                cde_batch = cde_descriptions[i:i + batch_size]
                cde_embeddings = torch.stack([self._get_embedding(desc) for desc in cde_batch])
                
                for j in range(0, len(reproschema_questions), batch_size):
                    repro_batch = reproschema_questions[j:j + batch_size]
                    repro_embeddings = torch.stack([self._get_embedding(q) for q in repro_batch])
                    
                    # Compute similarity matrix for current batch
                    similarity_matrix = util.cos_sim(cde_embeddings, repro_embeddings)
                    similarity_matrix = similarity_matrix.cpu().numpy()
                    
                    # Find best matches in current batch
                    for batch_idx, cde_idx in enumerate(range(i, min(i + batch_size, len(cde_names)))):
                        best_match_idx = j + np.argmax(similarity_matrix[batch_idx])
                        best_score = similarity_matrix[batch_idx][best_match_idx - j]
                        
                        if best_score >= threshold:
                            matches[cde_names[cde_idx]] = (reproschema_ids[best_match_idx], float(best_score))

            return matches
            
        except Exception as e:
            logger.error(f"Error in similarity matching: {str(e)}")
            return {}

    async def _agent_match(self, unmatched_responses: List[Dict], cde_definitions: pd.DataFrame) -> Dict[str, Tuple[str, float]]:
        """Use LLM to match remaining items"""
        matches = {}
        
        system_prompt = """You are a data mapping expert. Analyze the question from ReproSchema and the ElementDescription 
        from CDE to determine if they correspond to each other. Return your response as a JSON with 'matched_element' and 'confidence'."""
        
        try:
            for response in unmatched_responses:
                prompt = f"""
                ReproSchema Question: {response['question']}
                ReproSchema ID: {response['id']}
                
                Available CDE Elements:
                {cde_definitions[['ElementName', 'ElementDescription']].to_string()}
                
                Which ElementName best matches this question? Return as JSON: {{"matched_element": "element_name", "confidence": 0.0-1.0}}
                """
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ]
                
                result = await self.llm.ainvoke(messages)
                try:
                    # Extract JSON from response
                    json_match = re.search(r'\{.*\}', result.content.strip())
                    if json_match:
                        json_str = json_match.group()
                        json_data = json.loads(json_str)
                        if json_data['confidence'] > 0.7:
                            matches[json_data['matched_element']] = (response['id'], json_data['confidence'])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing LLM response: {str(e)}")
                    continue
                    
            return matches
            
        except Exception as e:
            logger.error(f"Error in agent matching: {str(e)}")
            return {}

    async def match(self, cde_definitions: pd.DataFrame, reproschema_responses: List[Dict],
            threshold: float = config.similarity_threshold) -> Dict[str, Tuple[str, float]]:
        """Main matching function that coordinates all matching steps"""
        try:
            # Extract IDs and URLs from input data
            for response in reproschema_responses:
                self.logger.info(f"Raw response data: {json.dumps(response, indent=2)}")
            
            # Extract IDs and URLs
            reproschema_items = [(r.get("id", ""), r.get("isAbout", "")) for r in reproschema_responses]
            self.logger.info("=== Processing Items ===")
            
            matches = {}
            used_ids = set()

            # Match based on both ID and URL
            for repro_id, about_url in reproschema_items:
                if not about_url:  # Skip if URL is empty
                    self.logger.warning(f"Empty URL for ID: {repro_id}")
                    continue
                    
                self.logger.info(f"Checking URL: {about_url}")
                
                # Get item name from URL
                if "items/interview_age" in about_url:
                    self.logger.info(f"Found interview_age match with ID: {repro_id}")
                    matches["interview_age"] = (repro_id, 1.0)
                    used_ids.add(repro_id)
                elif "items/sex" in about_url:
                    self.logger.info(f"Found sex match with ID: {repro_id}")
                    matches["sex"] = (repro_id, 1.0)
                    used_ids.add(repro_id)

            # Log mapping results
            self.logger.info("=== Initial Matches ===")
            self.logger.info(f"Matches found: {matches}")
            self.logger.info(f"Used IDs: {used_ids}")
                
            # Step 2: Exact ID matching for remaining items
            reproschema_ids = [r[0] for r in reproschema_items if r[0] not in used_ids]
            cde_names = cde_definitions["ElementName"].tolist()
            exact_matches, newly_used_ids = self._exact_id_match(reproschema_ids, cde_names)
            matches.update(exact_matches)
            used_ids.update(newly_used_ids)
            
            self.logger.debug(f"After exact matching: {matches}")

            # Step 3: Alias matching
            alias_matches, newly_used_ids = self._alias_match(reproschema_ids, cde_definitions, used_ids)
            matches.update(alias_matches)
            used_ids.update(newly_used_ids)
            
            # Step 4: Similarity matching for remaining items
            unmatched_responses = [r for r in reproschema_responses if r["id"] not in used_ids]
            if unmatched_responses:
                similarity_matches = self._similarity_match(cde_definitions, unmatched_responses, used_ids, threshold)
                matches.update(similarity_matches)
                used_ids.update(match[0] for match in similarity_matches.values() if match is not None)
            
            # Step 5: Agent matching as last resort
            final_unmatched = [r for r in reproschema_responses if r["id"] not in used_ids]
            if final_unmatched:
                agent_matches = await self._agent_match(final_unmatched, cde_definitions)
                matches.update(agent_matches)
            
            return matches

        except Exception as e:
            logger.error(f"Error in matching process: {str(e)}")
            raise MappingError(f"Failed to complete matching process: {str(e)}")
        
    # Response Mapping
class ResponseMapper:
    """Maps ReproSchema responses to CDE format"""
    SEX_MAPPING = {
        "http://schema.org/Female": "F",
        "http://schema.org/Male": "M",
        "Female": "F",
        "Male": "M",
        "F": "F",
        "M": "M"
    }

    def __init__(self, cde_definitions: pd.DataFrame):
        self.logger = logging.getLogger(__name__)
        required_columns = ["ElementName", "DataType", "ElementDescription", "Notes", "ValueRange"]
        missing_columns = [col for col in required_columns if col not in cde_definitions.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CDE definitions: {missing_columns}")
        self.cde_definitions = cde_definitions.copy()
        self.llm = ChatOllama(model=config.llm_model)
        self._cde_mappings_cache: Dict[str, CDEMapping] = {}
        
        self.system_message = SystemMessage(content="""You are a response mapping system that matches semantically equivalent responses between scales.
            Your task is to determine if the meaning in ReproSchema matches any meaning in the CDE scale.""")
        
        self.prompt_template = """You are a response scale mapping expert. Your task is to find semantic equivalents between different ways of expressing the same level or degree.

            ReproSchema description: {response_text} (value={response_value})

            Available CDE mappings:
            {cde_mapping}

            Task: Find the CDE option that is semantically closest to the ReproSchema description.

            Instructions:
            1. Look for direct semantic equivalence between ReproSchema and CDE value descriptions
            2. Map reproschema value to the CDE value that best preserves the meaning of the response
            3. If CDE value is string (e.g., "M" or "F" in sex), keep the string value
            4. Only use -9 if no semantic equivalent exists or reproschema value is null

            Consider:
            - Similar time periods (e.g., "less than a day" ≈ "once or twice")
            - Similar frequencies (e.g., "multiple times" ≈ "several times")
            - Similar intensities (e.g., "very intensely" ≈ "severe intensity")

            After your analysis, respond with ONLY a JSON object in this format: {{"mapped_value": "value"}}
            Where "value" is the matching CDE value or "-9" if no good match exists.
            DO NOT include any other text in your response."""

    def _parse_cde_notes(self, notes: str, value_range: Optional[str] = None) -> CDEMapping:
        """Parse CDE notes and value range into structured mapping format"""
        mapping = CDEMapping()

        try:
            if pd.notna(value_range):
                # Split value range on ';' to separate numeric ranges and extra values
                parts = value_range.split(";")
                range_part = parts[0].strip()  # First part should be the range (if any)
                extra_values = {p.strip() for p in parts[1:]}  # Additional values like -9

                # Handle numeric range
                if "::" in range_part:
                    try:
                        min_val, max_val = map(int, range_part.split("::"))
                        mapping.value_range = range_part
                        mapping.min_value = min_val
                        mapping.max_value = max_val
                        mapping.valid_values.update(str(i) for i in range(min_val, max_val + 1))
                    except ValueError:
                        logger.warning(f"Could not parse numeric range: {range_part}")

                # Add any extra values (e.g., -9)
                mapping.valid_values.update(extra_values)

            # Parse notes for mappings
            if pd.notna(notes):
                for part in notes.split(";"):
                    part = part.strip()
                    if not part:
                        continue

                    if "=" in part:
                        value, description = part.split("=", 1)
                        value, description = value.strip(), description.strip()
                        mapping.numeric_to_string[value] = description
                        mapping.string_to_numeric[description] = value
                        mapping.valid_values.add(value)
                    else:
                        mapping.valid_values.add(part)

            return mapping

        except Exception as e:
            logger.error(f"Error parsing CDE definition: {str(e)}")
            return CDEMapping()

    def _get_cde_mapping(self, cde_element: str) -> Optional[CDEMapping]:
        if cde_element not in self._cde_mappings_cache:
            cde_row = self.cde_definitions[self.cde_definitions["ElementName"] == cde_element]
            if cde_row.empty:
                return None
                
            notes = cde_row["Notes"].iloc[0]
            value_range = cde_row["ValueRange"].iloc[0] if "ValueRange" in cde_row.columns else None
            self._cde_mappings_cache[cde_element] = self._parse_cde_notes(notes, value_range)
                
        return self._cde_mappings_cache[cde_element]

    def _get_response_text(self, response_value: Any, response_options: List[Dict]) -> Optional[str]:
        """Get text description for a response value"""
        if not response_options:
            return None
            
        try:
            for option in response_options:
                if option.get('value') == response_value:
                    name_dict = option.get('name', {})
                    return name_dict.get('en')
            return None
            
        except Exception as e:
            logger.error(f"Error getting response text: {str(e)}")
            return None

    def _validate_and_convert_value(self, value: str, data_type: str, cde_mapping: CDEMapping) -> str:
        """Validate and convert value based on CDE DataType"""
        if value == "-9":
            return "NR" if data_type == "String" else "-9"
                
        try:
            # Simplified sex field handling - check if it's a sex value that needs conversion
            if value in self.SEX_MAPPING:
                return self.SEX_MAPPING[value]
                    
            # Handle integer fields
            if data_type == "Integer":
                try:
                    int_value = int(float(str(value)))
                    if cde_mapping.is_valid_value(str(int_value)):
                        return str(int_value)
                    return "-9"
                except (ValueError, TypeError):
                    return "-9"
                        
            # String fields - try direct validation first
            if data_type == "String":
                if cde_mapping.is_valid_value(value):
                    return value
                if value in cde_mapping.string_to_numeric:
                    return cde_mapping.string_to_numeric[value]
                if value in cde_mapping.numeric_to_string:
                    return cde_mapping.numeric_to_string[value]
                return "NR"
                
            return value
                        
        except Exception as e:
            logger.error(f"Error converting value {value} to {data_type}: {str(e)}")
            return "NR" if data_type == "String" else "-9"

    async def _find_best_match(self, response_value: Any, cde_mapping: CDEMapping, 
                     response_options: Optional[List[Dict]]) -> str:
        """Find best matching CDE value for a response"""
        if response_value is None:
            return "-9"
            
        try:
            # Convert response_value to string for comparison
            str_value = str(response_value)
            
            # Try exact value match first
            if str_value in cde_mapping.valid_values:
                return str_value
            
            # Get text description if available
            response_text = self._get_response_text(response_value, response_options)
            
            # Try direct text match if we have response text
            if response_text and response_text in cde_mapping.string_to_numeric:
                return cde_mapping.string_to_numeric[response_text]
            
            # Use LLM for semantic matching
            prompt = self.prompt_template.format(
                response_value=str_value,
                response_text=response_text or "No description available",
                cde_mapping=json.dumps({
                    "numeric_to_string": cde_mapping.numeric_to_string,
                    "string_to_numeric": cde_mapping.string_to_numeric,
                    "valid_values": list(cde_mapping.valid_values)
                }, indent=2)
            )
            
            messages = [
                self.system_message,
                HumanMessage(content=prompt)
            ]
            
            result = await self.llm.ainvoke(messages)
            result_content = result.content.strip()
            
            try:
                # Try to find JSON in the response
                json_match = re.search(r'\{[^}]+\}', result_content)
                if json_match:
                    json_str = json_match.group()
                    json_str = json_str.replace("'", '"')
                    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                    
                    json_data = json.loads(json_str)
                    mapped_value = json_data.get('mapped_value', '-9')
                    
                    if mapped_value in cde_mapping.valid_values:
                        return mapped_value
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response: {result_content}")
            
            return "-9"
                
        except Exception as e:
            logger.error(f"Error finding best match: {str(e)}")
            return "-9"

    async def map_responses(self, reproschema_responses: List[Dict], 
                       matched_mapping: Dict[str, Tuple[str, float]]) -> Dict[str, str]:
        """Map ReproSchema responses to CDE values"""
        mapped_data = {}
        repro_lookup = {r["id"]: r for r in reproschema_responses}
        
        try:
            self.logger.info("=== Start Response Mapping ===")
            self.logger.info(f"Matched mapping: {matched_mapping}")
            
            for cde_element, match_info in matched_mapping.items():
                self.logger.info(f"\nProcessing CDE element: {cde_element}")
                
                if match_info is None:
                    self.logger.warning(f"No match info for {cde_element}")
                    mapped_data[cde_element] = "-9"
                    continue
                    
                repro_id, confidence = match_info
                response = repro_lookup.get(repro_id)
                
                if response is None:
                    self.logger.warning(f"No response found for ID: {repro_id}")
                    mapped_data[cde_element] = "-9"
                    continue
                    
                self.logger.info(f"Found response: {response}")
                
                # Get CDE information
                cde_row = self.cde_definitions[self.cde_definitions["ElementName"] == cde_element]
                if cde_row.empty:
                    self.logger.warning(f"No CDE definition found for {cde_element}")
                    mapped_data[cde_element] = "-9"
                    continue
                    
                data_type = cde_row["DataType"].iloc[0]
                value_range = cde_row["ValueRange"].iloc[0] if "ValueRange" in cde_row.columns else None
                self.logger.info(f"Data type: {data_type}")
                self.logger.info(f"Value range: {value_range}")
                
                cde_mapping = self._get_cde_mapping(cde_element)
                if cde_mapping is None:
                    self.logger.warning(f"Could not get CDE mapping for {cde_element}")
                    mapped_data[cde_element] = "-9"
                    continue
                
                # Get semantic mapping
                raw_value = await self._find_best_match(
                    response.get("response_value"),
                    cde_mapping,
                    response.get("response_options")
                )
                self.logger.info(f"Raw value after best match: {raw_value}")
                
                # Validate and convert
                final_value = self._validate_and_convert_value(raw_value, data_type, cde_mapping)
                self.logger.info(f"Final converted value: {final_value}")
                
                mapped_data[cde_element] = final_value
            
            self.logger.info("\n=== Final Mapped Data ===")
            self.logger.info(mapped_data)
            return mapped_data
                
        except Exception as e:
            self.logger.error(f"Error mapping responses: {str(e)}")
            raise MappingError(f"Failed to map responses: {str(e)}")

    def create_template_row(self, mapped_data: Dict[str, str], template_columns: List[str]) -> List[str]:
        """Create a row for the template using mapped data"""
        try:
            return [mapped_data.get(col, "-9") for col in template_columns]
        except Exception as e:
            logger.error(f"Error creating template row: {str(e)}")
            return ["-9"] * len(template_columns)