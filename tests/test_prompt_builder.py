import unittest
from src.nlp.prompt import PromptBuilder

class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder()

    def test_build_prompt_basic(self):
        schema = "Table users, columns = [id, name]"
        question = "What are the names of all users?"
        prompt = self.builder.build(schema, question)
        
        self.assertIn("Table users", prompt)
        self.assertIn(question, prompt)
        self.assertIn("SQL assistant", prompt)

    def test_build_prompt_with_examples(self):
        schema = "Table users, columns = [id, name]"
        question = "What are the names of all users?"
        examples = [{"question": "count users", "query": "SELECT count(*) FROM users"}]
        prompt = self.builder.build(schema, question, examples)
        
        self.assertIn("### Examples:", prompt)
        self.assertIn("count users", prompt)
        self.assertIn("SELECT count(*) FROM users", prompt)

    def test_extract_sql(self):
        raw_output = "```sql\nSELECT name FROM users;\n```"
        extracted = self.builder.extract_sql(raw_output)
        self.assertEqual(extracted, "SELECT name FROM users;")
        
        raw_output_2 = "Here is the query: SELECT * FROM T1; Hope it helps"
        extracted_2 = self.builder.extract_sql(raw_output_2)
        # It takes the part before the first semicolon
        self.assertIn("SELECT * FROM T1;", extracted_2)

if __name__ == '__main__':
    unittest.main()
