import unittest
import os
import sqlite3
import shutil
from src.database.db_manager import DBManager

class TestDBManager(unittest.TestCase):
    def setUp(self):
        # WARNING: Always use a temporary directory for tests to avoid data loss.
        # DO NOT set this to 'spider_data/' as tearDown will delete it.
        self.test_dir = "test_temp_db"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_id = "test_db"
        self.db_folder = os.path.join(self.test_dir, self.db_id)
        os.makedirs(self.db_folder, exist_ok=True)
        self.db_path = os.path.join(self.db_folder, f"{self.db_id}.sqlite")
        
        # Create a sample database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS users")
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("INSERT INTO users (name) VALUES ('Alice')")
        cursor.execute("INSERT INTO users (name) VALUES ('Bob')")
        
        # Verify tables (as you intended)
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall() # Correctly fetch results
        
        if tables:
            table_name = tables[0][0]
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample = cursor.fetchall()
            print(f"\n[Test Setup] Created table '{table_name}' with data: {sample}")
            
        conn.commit()
        conn.close()
        
        self.db_manager = DBManager(self.test_dir)

    def tearDown(self):
        # Important: close engines to release file locks on Windows
        if hasattr(self, 'db_manager'):
            self.db_manager.close()
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
        except Exception as e:
            # Just log it, don't fail the test
            print(f"Cleanup warning: {e}")

    def test_execute_query(self):
        success, result = self.db_manager.execute_query(self.db_id, "SELECT name FROM users WHERE id=1")
        self.assertTrue(success)
        self.assertEqual(result[0][0], 'Alice')

    def test_validate_sql_correct(self):
        pred_sql = "SELECT name FROM users WHERE id=1"
        gold_sql = "SELECT name FROM users LIMIT 1"
        is_correct = self.db_manager.validate_sql(self.db_id, pred_sql, gold_sql)
        self.assertTrue(is_correct)

    def test_validate_sql_incorrect(self):
        pred_sql = "SELECT name FROM users WHERE id=2"
        gold_sql = "SELECT name FROM users WHERE id=1"
        is_correct = self.db_manager.validate_sql(self.db_id, pred_sql, gold_sql)
        self.assertFalse(is_correct)

if __name__ == '__main__':
    unittest.main()
