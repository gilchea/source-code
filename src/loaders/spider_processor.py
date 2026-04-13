import json
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SpiderProcessor:
    """Processes Spider dataset and partitions it into clients based on db_id."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.tables_file = os.path.join(data_dir, "tables.json")
        self.train_files = ["train_spider.json", "train_others.json"]
        self.dev_file = "dev.json"
        
        self.schemas = self._load_schemas()

    def _load_json(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_schemas(self) -> Dict[str, Any]:
        """Loads all schemas from tables.json."""
        data = self._load_json(self.tables_file)
        if not data:
            return {}
        return {item['db_id']: item for item in data}

    def get_federated_data(self, split: str = 'train') -> Dict[str, List[Dict[str, Any]]]:
        """Groups samples by db_id to simulate clients."""
        if split == 'train':
            files = [os.path.join(self.data_dir, f) for f in self.train_files]
        else:
            files = [os.path.join(self.data_dir, self.dev_file)]
            
        federated_data = {}
        
        for file_path in files:
            data = self._load_json(file_path)
            if not data:
                continue
                
            for sample in data:
                db_id = sample['db_id']
                if db_id not in federated_data:
                    federated_data[db_id] = []
                
                # Attach schema info to each client effectively
                federated_data[db_id].append(sample)
                
        return federated_data

    def get_schema_for_client(self, db_id: str) -> Dict[str, Any]:
        """Returns the schema metadata for a specific database."""
        return self.schemas.get(db_id, {})

    def format_schema_as_text(self, db_id: str) -> str:
        """Converts schema metadata into a textual description for the LLM."""
        schema = self.get_schema_for_client(db_id)
        if not schema:
            return ""
            
        table_names = schema['table_names_original']
        column_names = schema['column_names_original'] # [table_idx, col_name]
        
        text_parts = []
        for i, table in enumerate(table_names):
            cols = [col[1] for col in column_names if col[0] == i]
            text_parts.append(f"Table {table}, columns = [{', '.join(cols)}]")
            
        return " | ".join(text_parts)
