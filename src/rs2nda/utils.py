# src/rs2nda/utils.py
import requests
import json
from typing import Dict, List, Optional

# Cache for storing fetched item content
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
    
    # Split the isAbout URL into parts
    url_parts = is_about_url.split("/")
    
    # Remove the filename (last part) and the "items" directory
    if "items" not in url_parts:
        return None
    
    # Find the index of "items" and go up one level
    items_index = url_parts.index("items")
    base_parts = url_parts[:items_index]  # Everything before "items"
    
    # Remove the "../" prefix from responseOptions
    if response_options.startswith("../"):
        response_options = response_options[3:]
    
    # Construct the full URL
    return "/".join(base_parts + [response_options])

def extract_reproschema_responses(response_jsonld: List[Dict]) -> List[Dict]:
    """Extract ReproSchema responses with question details and constraints."""
    responses = []
    for entry in response_jsonld:
        if entry.get("@type") == "reproschema:Response":
            item_url = entry["isAbout"]
            item_content = fetch_item_content(item_url)
            # Fetch response options (constraints)
            response_options = item_content.get("responseOptions")
            options_url = get_constraints_url(item_url, response_options)
            options_content = fetch_item_content(options_url) if options_url else None
            
            # Extract only the choices from response options if available
            choices = options_content.get("choices") if options_content else None
            responses.append({
                'id': item_content["id"],
                'question': item_content["question"]["en"],  # Assuming English text
                'response_value': entry["value"],
                'response_options': choices  # Now only storing the choices
            })
    return responses

def clean_string(text: str) -> str:
    """Clean and normalize a string (e.g., remove extra spaces, lowercase)."""
    return " ".join(text.split()).strip().lower()

def validate_url(url: str) -> bool:
    """Check if a URL is valid."""
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.RequestException:
        return False