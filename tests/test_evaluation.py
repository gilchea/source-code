import unittest
from unittest.mock import MagicMock, patch
import os
import sqlite3
import shutil
from src.federated.client import VirtualClient
from src.database.db_manager import DBManager
from src.nlp.prompt import PromptBuilder

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_eval_db"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_id = "eval_db"
        self.db_folder = os.path.join(self.test_dir, self.db_id)
        os.makedirs(self.db_folder, exist_ok=True)
        self.db_path = os.path.join(self.db_folder, f"{self.db_id}.sqlite")
        
        # Setup mock db
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE T1 (id INT, val TEXT)")
        cursor.execute("INSERT INTO T1 VALUES (1, 'A')")
        conn.commit()
        conn.close()

        # Components
        self.db_manager = DBManager(self.test_dir)
        self.engine = MagicMock()
        self.prompt_builder = PromptBuilder()
        self.retriever = MagicMock()
        self.retriever.retrieve.return_value = [] # No few-shot for simplicity

        self.client = VirtualClient(self.db_id, self.engine, self.db_manager, self.prompt_builder, self.retriever)
        
        # Sample data
        self.samples = [
            {"question": "get all", "query": "SELECT * FROM T1"}
        ]
        self.client.setup(self.samples, {
            "table_names_original": ["T1"],
            "column_names_original": [[0, "id"], [0, "val"]]
        })

    def tearDown(self):
        self.db_manager.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_evaluate_correct_sql(self):
        # Mock engine to return the correct SQL
        self.engine.generate.return_value = "SELECT * FROM T1"
        
        results = self.client.evaluate()
        self.assertEqual(results['execution_accuracy'], 1.0)
        self.assertEqual(results['sample_count'], 1)

    def test_evaluate_incorrect_sql(self):
        # Mock engine to return incorrect SQL
        self.engine.generate.return_value = "SELECT val FROM T1"
        
        results = self.client.evaluate()
        self.assertEqual(results['execution_accuracy'], 0.0)

if __name__ == '__main__':
    unittest.main()
