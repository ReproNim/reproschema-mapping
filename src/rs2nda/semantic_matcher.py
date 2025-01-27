# src/rs2nda/semantic_matcher.py
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

class SemanticMatcher:
    def __init__(self, model_name="all-mpnet-base-v2"):  # Using a more powerful model
        self.model = SentenceTransformer(model_name)

    def match(self, cde_definitions: pd.DataFrame, reproschema_responses: List[Dict], threshold=0.5):  # Lower threshold
        """
        Match CDE definitions to ReproSchema questions using semantic similarity.
        
        Args:
            cde_definitions (pd.DataFrame): DataFrame containing CDE definitions with ElementName and ElementDescription
            reproschema_responses (List[Dict]): List of ReproSchema response objects
            threshold (float): Minimum similarity score for a match
        
        Returns:
            Dict[str, Tuple[str, float]]: Mapping of CDE ElementNames to (ReproSchema ID, similarity score)
        """
        # Extract descriptions and questions
        cde_descriptions = cde_definitions["ElementDescription"].tolist()
        cde_names = cde_definitions["ElementName"].tolist()
        
        reproschema_questions = [r["question"] for r in reproschema_responses]
        reproschema_ids = [r["id"] for r in reproschema_responses]

        # Generate embeddings
        cde_embeddings = self.model.encode(cde_descriptions, convert_to_tensor=True)
        repro_embeddings = self.model.encode(reproschema_questions, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_matrix = util.cos_sim(cde_embeddings, repro_embeddings)
        similarity_matrix = similarity_matrix.cpu().numpy()

        # Find best matches
        matches = {}
        for i, cde_name in enumerate(cde_names):
            best_match_idx = np.argmax(similarity_matrix[i])
            best_score = similarity_matrix[i][best_match_idx]
            
            if best_score >= threshold:
                matches[cde_name] = (reproschema_ids[best_match_idx], float(best_score))
            else:
                matches[cde_name] = None

        return matches