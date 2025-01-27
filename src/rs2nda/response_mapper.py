# src/rs2nda/response_mapper.py
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import re

class ResponseMapper:
    def __init__(self, cde_definitions: pd.DataFrame):
        self.cde_definitions = cde_definitions
        
    def _parse_cde_notes(self, notes: str) -> Dict[str, str]:
        """
        Parse CDE notes string into a mapping dictionary.
        Example: "0 = Not at all; 1 = Rare, less than a day or two" ->
                {'Not at all': '0', 'Rare, less than a day or two': '1'}
        """
        if pd.isna(notes):
            return {}
            
        mapping = {}
        # Split by semicolon and process each mapping
        parts = notes.split(';')
        for part in parts:
            part = part.strip()
            if '=' in part:
                value, name = part.split('=', 1)
                value = value.strip()
                name = name.strip()
                # Store mapping in both directions for flexibility
                mapping[name] = value
                mapping[value] = value  # Allow direct value mapping
        return mapping

    def _find_best_match(self, response_value: Any, cde_mapping: Dict[str, str]) -> str:
        """
        Find the best matching CDE value for a given ReproSchema response value.
        """
        if response_value is None:
            return "-9"  # Missing value
            
        # If response_value is already a number and in the mapping, return it
        if str(response_value) in cde_mapping:
            return str(response_value)
            
        # Convert response_value to string for comparison
        response_str = str(response_value)
        
        # Try direct mapping first
        if response_str in cde_mapping:
            return cde_mapping[response_str]
            
        # Look for exact matches in the mapping keys
        for cde_name, cde_value in cde_mapping.items():
            if response_str.lower() == cde_name.lower():
                return cde_value
                
        return "-9"  # Default to missing value if no match found

    def map_responses(self, 
                     reproschema_responses: List[Dict], 
                     matched_mapping: Dict[str, Tuple[str, float]]) -> Dict[str, str]:
        """
        Map ReproSchema response values to CDE values.
        
        Args:
            reproschema_responses: List of ReproSchema responses
            matched_mapping: Mapping of CDE ElementNames to (ReproSchema ID, similarity score)
        
        Returns:
            Dictionary mapping CDE ElementNames to their corresponding values
        """
        mapped_data = {}
        
        # Create a lookup dictionary for reproschema responses
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
                
            # Get CDE notes for this element
            cde_row = self.cde_definitions[self.cde_definitions["ElementName"] == cde_element]
            if cde_row.empty:
                mapped_data[cde_element] = "-9"
                continue
                
            cde_notes = cde_row["Notes"].iloc[0]
            cde_mapping = self._parse_cde_notes(cde_notes)
            
            # Get response value
            response_value = response.get("response_value")
            mapped_value = self._find_best_match(response_value, cde_mapping)
            mapped_data[cde_element] = mapped_value
            
        return mapped_data

    def create_template_row(self, mapped_data: Dict[str, str], template_columns: List[str]) -> List[str]:
        """
        Create a row for the template using mapped data.
        
        Args:
            mapped_data: Dictionary of mapped values
            template_columns: List of column names in order
            
        Returns:
            List of values in template column order
        """
        return [mapped_data.get(col, "-9") for col in template_columns]