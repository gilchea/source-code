import unittest
import os
import json
import shutil
from src.loaders.spider_processor import SpiderProcessor

class TestSpiderProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_spider_data"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy tables.json
        self.tables = [
            {
                "db_id": "db1",
                "table_names_original": ["Table1"],
                "column_names_original": [[-1, "*"], [0, "Col1"], [0, "Col2"]]
            }
        ]
        with open(os.path.join(self.test_dir, "tables.json"), 'w') as f:
            json.dump(self.tables, f)
            
        # Create dummy train_spider.json
        self.train_data = [
            {"db_id": "db1", "question": "q1", "query": "query1"},
            {"db_id": "db1", "question": "q2", "query": "query2"}
        ]
        with open(os.path.join(self.test_dir, "train_spider.json"), 'w') as f:
            json.dump(self.train_data, f)
            
        # Create empty train_others.json to avoid loading errors if processor expects it
        with open(os.path.join(self.test_dir, "train_others.json"), 'w') as f:
            json.dump([], f)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_federated_data(self):
        processor = SpiderProcessor(self.test_dir)
        fed_data = processor.get_federated_data(split='train')
        
        self.assertIn("db1", fed_data)
        self.assertEqual(len(fed_data["db1"]), 2)

    def test_format_schema_as_text(self):
        processor = SpiderProcessor(self.test_dir)
        schema_text = processor.format_schema_as_text("db1")
        self.assertIn("Table Table1", schema_text)
        self.assertIn("Col1", schema_text)

if __name__ == '__main__':
    unittest.main()
