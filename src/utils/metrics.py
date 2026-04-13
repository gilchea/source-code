import os
import pandas as pd
import json
import logging
from typing import List, Dict

class MetricsTracker:
    """Tracks and saves federated learning metrics across rounds."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = []

    def log_round(self, round_idx: int, metrics: Dict[str, float]):
        """Records metrics for a specific round."""
        metrics['round'] = round_idx
        self.history.append(metrics)
        
        # Save to CSV continuously
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.log_dir, "federated_metrics.csv"), index=False)
        
    def get_summary(self):
        """Returns the final summary of the experiment."""
        if not self.history:
            return "No data recorded."
        df = pd.DataFrame(self.history)
        return df.describe()
