import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any
import logging
# from transformers import AdamW
from torch.optim import AdamW

logger = logging.getLogger(__name__)

class SQLDataset(Dataset):
    """Simple dataset for SQL fine-tuning."""
    def __init__(self, samples: List[Dict[str, Any]], prompt_builder, schema_text: str):
        self.samples = samples
        self.prompt_builder = prompt_builder
        self.schema_text = schema_text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # We don't use few-shot during actual training to avoid overfitting on examples 
        # but the prompt structure should be consistent.
        prompt = self.prompt_builder.build(self.schema_text, item['question'], examples=[])
        return {
            "prompt": prompt,
            "target": item['query']
        }

class VirtualClient:
    """Simulates a Federated Learning client with a private database."""
    
    def __init__(self, client_id: str, engine, db_manager, prompt_builder, retriever):
        self.client_id = client_id
        self.engine = engine
        self.db_manager = db_manager
        self.prompt_builder = prompt_builder
        self.retriever = retriever
        
        self.local_data = []
        self.schema_text = ""
        self.dataset = None

    def setup(self, samples: List[Dict[str, Any]], schema_meta: Dict[str, Any]):
        """Initializes client data and retriever."""
        self.local_data = samples
        # Convert schema to text
        table_names = schema_meta.get('table_names_original', [])
        column_names = schema_meta.get('column_names_original', [])
        text_parts = [f"Table {t}, columns=[{', '.join([c[1] for c in column_names if c[0] == i])}]" 
                      for i, t in enumerate(table_names)]
        self.schema_text = " | ".join(text_parts)
        
        # Build local retrieval index
        self.retriever.build_index(samples)
        self.dataset = SQLDataset(samples, self.prompt_builder, self.schema_text)

    def set_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Sets the local model weights from global weights."""
        self.engine.model.load_state_dict(state_dict, strict=False)

    def get_weights(self, clip_threshold: float = 1.0, noise_multiplier: float = 0.01, 
                    top_k_ratio: float = 1.0, use_quantization: bool = False) -> Dict[str, torch.Tensor]:
        """Returns the current LoRA weights with optional DP and Efficiency applied."""
        from src.privacy.dp_engine import DPEngine
        
        # 1. Extract raw LoRA weights
        state_dict = self.engine.model.state_dict()
        weights = {k: v.cpu() for k, v in state_dict.items() if "lora_" in k}
        
        # 2. Apply Differential Privacy (Clipping + Noise)
        weights = DPEngine.apply_dp(weights, clip_threshold, noise_multiplier)
        
        # 3. Apply Sparsification (Top-K)
        if top_k_ratio < 1.0:
            weights = DPEngine.apply_sparsification(weights, top_k_ratio)
            
        # 4. Apply Quantization (FP16)
        if use_quantization:
            weights = DPEngine.apply_quantization(weights)
            
        return weights

    # def local_train(self, epochs: int = 1, lr: float = 5e-5, batch_size: int = 4):
    #     """Performs local fine-tuning on client data."""
    #     self.engine.model.train()
    #     optimizer = AdamW(self.engine.model.parameters(), lr=lr)
        
    #     # Simple colate for text
    #     dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
    #     for epoch in range(epochs):
    #         total_loss = 0
    #         for batch in dataloader:
    #             optimizer.zero_grad()
                
    #             # Tokenize batch
    #             inputs = self.engine.tokenizer(
    #                 [p + t for p, t in zip(batch['prompt'], batch['target'])],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.engine.device)
                
    #             # Mask prompt from loss calculation (optional but standard)
    #             outputs = self.engine.model(**inputs, labels=inputs["input_ids"])
    #             loss = outputs.loss
                
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
                
    #         logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
    def local_train(self, epochs: int = 1, lr: float = 5e-5, batch_size: int = 4):
        """Performs local fine-tuning on client data."""
        self.engine.model.train()
        # Di chuyển optimizer vào trong để đảm bảo nó được khởi tạo mới cho mỗi client
        # (Standard Federated Learning)
        optimizer = AdamW(self.engine.model.parameters(), lr=lr)
        
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                inputs = self.engine.tokenizer(
                    [p + t for p, t in zip(batch['prompt'], batch['target'])],
                    padding=True,
                    truncation=True,
                    max_length=512, # Giới hạn max_length để tránh OOM
                    return_tensors="pt"
                ).to(self.engine.device)
                
                outputs = self.engine.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
        
        # --- QUAN TRỌNG: Giải phóng bộ nhớ sau khi train xong một client ---
        del optimizer
        torch.cuda.empty_cache()
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluates model performance on local data using Execution Accuracy."""
        self.engine.model.eval()
        correct = 0
        total = len(self.local_data)
        
        for item in self.local_data:
            # 1. Retrieve local few-shot examples
            examples = self.retriever.retrieve(item['question'], k=3)
            # 2. Build prompt
            prompt = self.prompt_builder.build(self.schema_text, item['question'], examples)
            # 3. Generate SQL
            raw_sql = self.engine.generate(prompt)
            pred_sql = self.prompt_builder.extract_sql(raw_sql)
            
            # 4. Valid Execution
            is_correct = self.db_manager.validate_sql(self.client_id, pred_sql, item['query'])
            if is_correct:
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        return {
            "execution_accuracy": accuracy,
            "sample_count": total
        }
