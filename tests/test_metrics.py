import unittest
import os
import pandas as pd
import shutil
from src.utils.metrics import MetricsTracker

class TestMetricsTracker(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_metrics_logs"
        self.tracker = MetricsTracker(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_log_round(self):
        metrics1 = {"accuracy": 0.8, "loss": 0.5}
        self.tracker.log_round(1, metrics1)
        
        metrics2 = {"accuracy": 0.85, "loss": 0.3}
        self.tracker.log_round(2, metrics2)
        
        self.assertEqual(len(self.tracker.history), 2)
        self.assertEqual(self.tracker.history[0]['round'], 1)
        
        # Check if CSV is created
        csv_path = os.path.join(self.test_dir, "federated_metrics.csv")
        self.assertTrue(os.path.exists(csv_path))
        
        # Verify content
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]['accuracy'], 0.85)

    def test_get_summary(self):
        self.tracker.log_round(1, {"accuracy": 0.5})
        self.tracker.log_round(2, {"accuracy": 1.0})
        summary = self.tracker.get_summary()
        
        # summary is result of df.describe()
        self.assertEqual(summary.loc['mean', 'accuracy'], 0.75)

if __name__ == '__main__':
    unittest.main()
