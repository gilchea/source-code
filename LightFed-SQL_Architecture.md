# LightFed-SQL: Federated Fine-tuning SLMs for NL2SQL with In-Context Learning

## 1. Project Overview
**LightFed-SQL** is a federated learning (FL) framework designed to fine-tune Small Language Models (SLMs) for Natural Language to SQL (NL2SQL) tasks. It specifically addresses the challenges of cross-domain database schemas while preserving data privacy at the client level.

The core strategy combines **Federated Fine-tuning (PeFT/LoRA)** and **In-Context Learning (ICL)** with local semantic retrieval.

## 2. Technical Philosophy
- **Model-Centric**: Fine-tuning SLMs (e.g., Phi-3-mini) to be more sensitive to SQL syntax and schema mapping.
- **Privacy-First**: Using Federated Learning to keep databases local and Differential Privacy (DP) to secure gradient updates.
- **Efficiency**: Utilizing Parameter-Efficient Fine-Tuning (LoRA) to minimize communication costs.
- **Context-Aware**: Enhancing few-shot prompts using localized retrievers unique to each client's domain.

## 3. System Architecture (Modular)

### A. Federated Module (`src.federated`)
- **Server**: Manages the global model, orchestrates training rounds, and performs aggregation using **FedAvg** or **Adaptive FedOpt** (FedAdam/FedAdagrad).
- **Client**: Performs local training on private data, applies local DP constraints, and computes local updates.
- **Aggregator**: Abstracted mathematical logic for weight merging.

### B. NLP & Model Module (`src.models` & `src.nlp`)
- **SLM Engine**: Wrapper for HuggingFace Transformers (Phi-3).
- **LoRA Adapter**: Manages trainable low-rank adapters.
- **Prompt Builder**: Constructs ICL prompts including schema information and few-shot examples.
- **Local Retriever**: Implements semantic search (e.g., using BGE embeddings) to find similar $(Question, SQL)$ pairs from the client's **internal** historical data.

### C. Database & Execution (`src.database`)
- **Local Storage**: Each client manages its own **SQLite** or **PostgreSQL** instance.
- **SQL Validator**: Executes generated SQL against the local DB to calculate **Execution Accuracy**.

### D. Privacy & Security (`src.privacy`)
- **DP Engine**: Implements Differential Privacy (Clipping + Gaussian Noise) at the client-side training step to ensure $\epsilon, \delta$-privacy.

## 4. Federated Learning Flow
1. **Broadcast**: Server sends the global LoRA weights to selected clients.
2. **Local Retrieval**: Each client retrieves relevant $(Q, SQL)$ examples from its local "synthetic" or historical dataset.
3. **Local Training**: Clients perform fine-tuning (e.g., 1-5 epochs) using their local database questions.
4. **Privacy Protection**: Local updates are clipped and noise is added (DP).
5. **Upload**: Clients send only the **noisy LoRA adapters** back to the server.
6. **Aggregation**: Server merges adapters into the global model.

## 5. Performance Metrics
- **Execution Accuracy (EX)**: Percentage of generated queries that return the correct result from the DB.
- **Communication Cost**: Total MB transferred per round (optimized by LoRA).
- **Convergence Speed**: Number of FL rounds to reach target accuracy.
- **Inference Latency**: Time taken for the SLM to generate SQL (ms).
- **Privacy Budget ($\epsilon$)**: Tracking the cumulative privacy loss.

## 6. Directory Structure
```
LightFed-SQL/
├── spider_data/                   # Chứa dataset Spider theo domain
├── config/                 # Cấu hình hệ thống (fl_config.yaml, model_config.yaml)
├── src/
│   ├── federated/          # Logic của Federated Learning
│   │   ├── server.py       # Quản lý Server trung tâm (FedAvg/FedOpt)
│   │   ├── client.py       # Logic xử lý tại Client cá nhân
│   │   └── aggregator.py   # Các phương thức cộng dồn trọng số (FedAvg, Adam, SGD)
│   ├── models/             # Quản lý SLM (Phi-3)
│   │   ├── adapter.py      # Cấu hình PEFT/LoRA (để giảm chi phí truyền tải)
│   │   └── engine.py       # Wrapper cho mô hình Transformers
│   ├── nlp/                # Xử lý ngôn ngữ tự nhiên và SQL
│   │   ├── prompt.py       # Xây dựng Prompt ICL linh hoạt
│   │   ├── retriever.py    # Truy xuất ví dụ mẫu NỘI BỘ client
│   │   └── parser.py       # Hiệu chỉnh Schema và kiểm tra cú pháp
│   ├── privacy/            # Bảo mật dữ liệu
│   │   └── dp_engine.py    # Cơ chế Differential Privacy (Clipping, Noising)
│   ├── database/           # Quản lý Database cục bộ của mỗi client
│   │   └── db_manager.py   # Kết nối SQLite/PostgreSQL để validator SQL
│   ├── utils/              # Tiện ích ghi log, đo lường metrics
│   │   └── metrics.py      # Tính Exec Acc, Comm. Cost, Latency...
├── results/                # Lưu trữ kết quả thí nghiệm, đồ thị hội tụ
├── main.py                 # File chạy simulation chính
└── requirements.txt
```

## 7. Next Steps Plan
1. **Phase 1**: Setup the simulation environment and local Database Manager (SQLite).
2. **Phase 2**: Implement the LoRA-based local training loop with Phi-3.
3. **Phase 3**: Integrate FedAvg and verify cross-domain accuracy improvements.
4. **Phase 4**: Add DP-SGD and analyze the accuracy-privacy trade-off.
