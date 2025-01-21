from pathlib import Path
import json
import pandas as pd
import requests
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import re
from .models import RSQuestion, NDAQuestion, QuestionMapping

class QuestionMapper:
    def __init__(self, nda_dir: Path):
        self.nda_dir = nda_dir
        self.rs_questions: Dict[str, RSQuestion] = {}
        self.nda_questions: Dict[str, NDAQuestion] = {}
        self.mappings: List[QuestionMapping] = []

    def load_rs_questions(self, items_url: str) -> None:
        """Load ReproSchema questions from GitHub URL"""
        # Convert GitHub URL to raw content URL
        raw_url = items_url.replace("github.com", "raw.githubusercontent.com")
        raw_url = raw_url.replace("/tree/", "/")
        
        # Fetch items directory
        response = requests.get(f"{raw_url}/items")
        if response.status_code == 200:
            items_content = response.json()
            
            # Load each item file
            for item in items_content:
                if item['name'].endswith('.json'):
                    item_response = requests.get(f"{raw_url}/items/{item['name']}")
                    if item_response.status_code == 200:
                        question_data = item_response.json()
                        question = RSQuestion(**question_data)
                        self.rs_questions[question.id] = question

    def load_nda_definitions(self, questionnaire: str) -> None:
        """Load NDA definitions from CSV"""
        file_path = self.nda_dir / f"{questionnaire}_definitions.csv"
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            question = NDAQuestion(
                element_name=row['ElementName'],
                data_type=row['DataType'],
                size=row['Size'],
                required=row['Required'],
                element_description=row['ElementDescription'],
                value_range=row['ValueRange'],
                notes=row['Notes'],
                aliases=row.get('Aliases', None)
            )
            self.nda_questions[question.element_name] = question

    def _clean_text(self, text: str) -> str:
        """Clean text for comparison"""
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower().strip()
        return text

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        text1 = self._clean_text(text1)
        text2 = self._clean_text(text2)
        return SequenceMatcher(None, text1, text2).ratio()

    def _find_matching_question(self, rs_question: RSQuestion) -> Tuple[str, float, str]:
        """Find matching NDA question for given RS question"""
        best_match = None
        best_score = 0
        method = "none"

        # First try ID match
        rs_id_clean = self._clean_text(rs_question.id)
        for nda_id, nda_question in self.nda_questions.items():
            # Try direct ID match
            if rs_id_clean in self._clean_text(nda_id):
                return nda_id, 1.0, "id_match"

            # Check aliases if available
            if nda_question.aliases:
                aliases = str(nda_question.aliases).split(';')
                for alias in aliases:
                    if rs_id_clean in self._clean_text(alias):
                        return nda_id, 0.9, "alias_match"

            # Compare question text
            rs_text = rs_question.question.get('en', '')  # Using English text
            nda_text = nda_question.element_description
            similarity = self._calculate_text_similarity(rs_text, nda_text)
            
            if similarity > best_score:
                best_score = similarity
                best_match = nda_id
                method = "text_similarity"

        return best_match, best_score, method

    def map_questions(self, confidence_threshold: float = 0.7) -> List[QuestionMapping]:
        """Map RS questions to NDA questions"""
        self.mappings = []
        
        for rs_id, rs_question in self.rs_questions.items():
            nda_id, confidence, method = self._find_matching_question(rs_question)
            
            if nda_id and confidence >= confidence_threshold:
                mapping = QuestionMapping(
                    rs_id=rs_id,
                    nda_id=nda_id,
                    confidence=confidence,
                    method=method
                )
                self.mappings.append(mapping)

        return self.mappings

    def export_mappings(self, output_file: Path) -> None:
        """Export mappings to YAML file"""
        mappings_dict = {
            "question_mappings": {
                mapping.rs_id: {
                    "nda_id": mapping.nda_id,
                    "confidence": mapping.confidence,
                    "method": mapping.method
                }
                for mapping in self.mappings
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(mappings_dict, f, indent=2)

def create_question_mapping(nda_dir: Path, 
                          rs_url: str, 
                          questionnaire: str, 
                          output_file: Path):
    """Create mapping for a questionnaire"""
    mapper = QuestionMapper(nda_dir)
    
    # Load questions from both sources
    mapper.load_rs_questions(rs_url)
    mapper.load_nda_definitions(questionnaire)
    
    # Create mappings
    mappings = mapper.map_questions()
    
    # Export results
    mapper.export_mappings(output_file)
    
    return mappings