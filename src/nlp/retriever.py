import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Any

class SchemaRetriever:
    """Retrieves similar examples (few-shot) from a local dataset pool."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu", shared_model = None  ):
        # self.model = SentenceTransformer(model_name, device=device)
        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = SentenceTransformer(model_name, device=device)

        self.device = device
        self.index = None
        self.pool = []

    def build_index(self, examples: List[Dict[str, Any]]):
        """Encodes all local examples and builds a search index."""
        self.pool = examples
        questions = [ex['question'] for ex in examples]
        
        # Compute embeddings
        embeddings = self.model.encode(questions, convert_to_tensor=True, device=self.device)
        self.index = embeddings

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Finds top-k similar examples for the given query."""
        if self.index is None or len(self.pool) == 0:
            return []
            
        # Encode query
        query_emb = self.model.encode(query, convert_to_tensor=True, device=self.device)
        
        # Compute cosine similarity
        scores = util.cos_sim(query_emb, self.index)[0]
        
        # Get top-k indices
        top_k_indices = torch.topk(scores, min(k, len(self.pool))).indices
        
        return [self.pool[idx] for idx in top_k_indices]
