from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel

class MappingConfig(BaseModel):
    nda_name: str
    rs_url: str
    mapping_rules: Dict[str, str]  # rs_id -> nda_id

class BaseMapper:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        
    def load_config(self) -> MappingConfig:
        """Load mapping configuration"""
        pass