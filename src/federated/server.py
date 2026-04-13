import torch
import copy
from typing import List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class FederatedServer:
    """Coordinates the global training process and weight aggregation."""
    
    def __init__(self, engine, aggregator_type: str = "fedavg"):
        self.engine = engine
        self.aggregator_type = aggregator_type
        # In Federated LoRA, the "global model" is the state_dict of the LoRA layers
        self.global_weights = self._get_initial_weights()
        self.round = 0

    def _get_initial_weights(self) -> Dict[str, torch.Tensor]:
        """Extracts initial LoRA weights from the engine."""
        state_dict = self.engine.model.state_dict()
        return {k: v.cpu() for k, v in state_dict.items() if "lora_" in k}

    def select_clients(self, client_ids: List[str], fraction: float = 0.1) -> List[str]:
        """Randomly selects a fraction of clients for participation."""
        num_to_select = max(1, int(len(client_ids) * fraction))
        return list(np.random.choice(client_ids, num_to_select, replace=False))

    def aggregate(self, updates: List[Dict[str, torch.Tensor]], sample_counts: List[int]):
        """Aggregates local updates into the global model."""
        if self.aggregator_type == "fedavg":
            self.global_weights = self._fed_avg(updates, sample_counts)
        elif self.aggregator_type == "fedopt":
            # Simple placeholder for FedOpt/FedAdam logic
            self.global_weights = self._fed_avg(updates, sample_counts)
            logger.info("FedOpt selected (using FedAvg as base aggregator)")
        
        # Load aggregated weights back to the server engine for evaluation if needed
        self.engine.model.load_state_dict(self.global_weights, strict=False)

    def _fed_avg(self, updates: List[Dict[str, torch.Tensor]], sample_counts: List[int]) -> Dict[str, torch.Tensor]:
        """Standard Weighted Federated Averaging."""
        total_samples = sum(sample_counts)
        # Deep copy to avoid modifying original weights
        aggregated_weights = copy.deepcopy(updates[0])
        
        for k in aggregated_weights.keys():
            # Initial weighted value
            aggregated_weights[k] = aggregated_weights[k] * (sample_counts[0] / total_samples)
            # Add weighted updates from other clients
            for i in range(1, len(updates)):
                aggregated_weights[k] += updates[i][k] * (sample_counts[i] / total_samples)
                
        return aggregated_weights

    def save_checkpoint(self, path: str):
        """Saves current global weights."""
        torch.save(self.global_weights, path)
        logger.info(f"Global model saved to {path}")

    def load_checkpoint(self, path: str):
        """Loads weights from a checkpoint."""
        self.global_weights = torch.load(path)
        self.engine.model.load_state_dict(self.global_weights, strict=False)
        logger.info(f"Global model loaded from {path}")
