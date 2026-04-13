import unittest
import torch
from src.nlp.retriever import SchemaRetriever

class TestSchemaRetriever(unittest.TestCase):
    def setUp(self):
        # Using cpu for testing
        self.retriever = SchemaRetriever(device="cpu")
        self.examples = [
            {"question": "How many users are there?", "query": "SELECT count(*) FROM users"},
            {"question": "What is the name of user 1?", "query": "SELECT name FROM users WHERE id=1"},
            {"question": "List all tables", "query": "SELECT name FROM sqlite_master WHERE type='table'"}
        ]
        self.retriever.build_index(self.examples)

    def test_retrieve_exact_match(self):
        results = self.retriever.retrieve("How many users are there?", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['question'], "How many users are there?")

    def test_retrieve_similar(self):
        results = self.retriever.retrieve("Show count of users", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['question'], "How many users are there?")

    def test_retrieve_multiple(self):
        results = self.retriever.retrieve("user info", k=2)
        self.assertEqual(len(results), 2)

if __name__ == '__main__':
    unittest.main()
