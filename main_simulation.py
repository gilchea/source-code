import os
import logging
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

from src.loaders.spider_processor import SpiderProcessor
from src.database.db_manager import DBManager
from src.models.engine import SLMEngine
from src.nlp.prompt import PromptBuilder
from src.nlp.retriever import SchemaRetriever
from src.federated.server import FederatedServer
from src.federated.client import VirtualClient
from src.utils.metrics import MetricsTracker

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
CONFIG = {
    "model_id": "microsoft/Phi-3-mini-4k-instruct",
    "data_dir": "/content/source-code/spider_data",
    "num_rounds": 10,
    "fraction": 0.05, # % of clients per round
    "local_epochs": 1,
    "lr": 5e-5,
    "batch_size": 2,
    "dp_epsilon": 1.0, # Clipping threshold
    "dp_noise": 0.01,
    "top_k_ratio": 0.5, # Sparsification
    "use_quantization": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def main():
    # 1. Load Data
    logger.info("--- Initializing LightFed-SQL Simulation ---")
    processor = SpiderProcessor(CONFIG["data_dir"])
    train_fed_data = processor.get_federated_data(split='train')
    dev_fed_data = processor.get_federated_data(split='dev')
    
    # 1. Khởi tạo model embedding duy nhất một lần
    shared_embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=CONFIG["device"])

    # 2. Setup Central Components
    db_manager = DBManager(os.path.join(CONFIG["data_dir"], "database"))
    engine = SLMEngine(CONFIG["model_id"], device=CONFIG["device"])
    engine.apply_lora(r=16) # Initialize LoRA
    
    server = FederatedServer(engine, aggregator_type="fedavg")
    tracker = MetricsTracker("./results")
    prompt_builder = PromptBuilder()

    # # 3. Initialize Virtual Clients
    logger.info(f"Setting up {len(train_fed_data)} virtual clients...")
    clients = {}
    for db_id, samples in train_fed_data.items():
        # DÙNG CHUNG shared_embedding_model
        local_retriever = SchemaRetriever(device=CONFIG["device"], shared_model=shared_embedding_model)
        client = VirtualClient(db_id, engine, db_manager, prompt_builder, local_retriever)
        client.setup(samples, processor.get_schema_for_client(db_id))
        clients[db_id] = client

    # 4. Federated Learning Loop
    for r in range(1, CONFIG["num_rounds"] + 1):
        logger.info(f"\n--- ROUND {r}/{CONFIG['num_rounds']} ---")
        
        # Select clients
        selected_client_ids = server.select_clients(list(clients.keys()), fraction=CONFIG["fraction"])
        logger.info(f"Selected clients: {selected_client_ids}")
        
        updates = []
        sample_counts = []

        # # Server Aggregation
        # server.aggregate(updates, sample_counts)
        # torch.cuda.empty_cache() # GIẢI PHÓNG BỘ NHỚ
        
        # Local Training
        for cid in tqdm(selected_client_ids, desc="Local Training"):
            client = clients[cid]
            # Sync with global model
            client.set_weights(server.global_weights)
            # Train
            client.local_train(epochs=CONFIG["local_epochs"], lr=CONFIG["lr"], batch_size=CONFIG["batch_size"])
            # Extract weights with DP and Sparsification
            noisy_weights = client.get_weights(
                clip_threshold=CONFIG["dp_epsilon"],
                noise_multiplier=CONFIG["dp_noise"],
                top_k_ratio=CONFIG["top_k_ratio"],
                use_quantization=CONFIG["use_quantization"]
            )
            updates.append(noisy_weights)
            sample_counts.append(len(client.local_data))

        # Server Aggregation
        server.aggregate(updates, sample_counts)
        torch.cuda.empty_cache() # GIẢI PHÓNG BỘ NHỚ
        
        # Evaluation (every round or every few rounds)
        # Evaluation
        if r % 1 == 0:
            logger.info("Evaluating global model...")
            test_clients = list(dev_fed_data.keys())[:10]
            for t_cid in test_clients:
                if t_cid not in clients:
                    # LƯU Ý: Vẫn dùng shared_embedding_model ở đây
                    retriever = SchemaRetriever(device=CONFIG["device"], shared_model=shared_embedding_model)
                    client = VirtualClient(t_cid, engine, db_manager, prompt_builder, retriever)
                    client.setup(dev_fed_data[t_cid], processor.get_schema_for_client(t_cid))
                    clients[t_cid] = client
                    
    # Final Save
    server.save_checkpoint("./results/global_model_final.pt")
    logger.info("Simulation completed!")

if __name__ == "__main__":
    main()
