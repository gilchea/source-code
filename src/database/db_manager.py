import os
import sqlite3
import logging
from sqlalchemy import create_engine, Engine, text
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

class DBManager:
    """Manages SQLite connections for Spider databases."""
    
    def __init__(self, db_root_path: str):
        self.db_root_path = db_root_path
        self.engines = {}

    def get_engine(self, db_id: str) -> Engine:
        """Returns a SQLAlchemy engine for a specific db_id."""
        if db_id not in self.engines:
            db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file not found at {db_path}")
            
            db_url = f'sqlite:///{db_path}'
            self.engines[db_id] = create_engine(db_url)
        return self.engines[db_id]

    def execute_query(self, db_id: str, query: str) -> Tuple[bool, List[Any]]:
        """Executes a query and returns (success, result)."""
        engine = self.get_engine(db_id)
        try:
            with engine.connect() as connection:
                result = connection.execute(text(query))
                return True, [row for row in result]
        except Exception as e:
            logger.error(f"Error executing query on {db_id}: {e}")
            return False, str(e)

    def validate_sql(self, db_id: str, predicted_sql: str, golden_sql: str) -> bool:
        """Compares the results of two SQL queries."""
        success_p, res_p = self.execute_query(db_id, predicted_sql)
        success_g, res_g = self.execute_query(db_id, golden_sql)
        
        if not success_p or not success_g:
            return False
            
        # Comparison of results (set-based to ignore order if not specified)
        return set(res_p) == set(res_g)

    def close(self):
        """Disposes of all cached engines to free file handles."""
        for engine in self.engines.values():
            engine.dispose()
        self.engines.clear()
