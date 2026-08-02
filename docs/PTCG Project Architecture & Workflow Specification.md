# **Pokémon TCG AI Challenge: Repository Architecture & MLOps Blueprint**

**Document Status:** Approved for Engineering Setup  
**Target Goal:** Professional Python repository structure for end-to-end data scraping, self-play data generation, behavioral cloning / RL training, model checkpoint management, local ladder evaluation, and Kaggle submission packaging.

## **1\. Directory Tree Overview**

Below is the production-grade directory layout designed to isolate data processing, core model code, training loops, local evaluation arenas, and deployment artifacts.  
ptcg-ai/  
├── README.md  
├── requirements.txt  
├── pyproject.toml  
├── .gitignore  
│  
├── configs/                          \# Experiment & Hyperparameter Configurations  
│   ├── env\_config.yaml               \# Kaggle environment & engine settings  
│   ├── feature\_config.yaml           \# Card pool & lookup table dimensions  
│   ├── model/  
│   │   ├── mlp\_encoder.yaml          \# Baseline MLP Board Encoder params  
│   │   └── transformer\_encoder.yaml  \# Transformer Board Encoder params  
│   └── train/  
│       ├── bc\_trainer.yaml           \# Behavioral Cloning training hyperparameters  
│       └── rl\_trainer.yaml           \# Reinforcement Learning / PPO settings  
│  
├── data/                             \# Data Storage (Git Ignored except static card CSV)  
│   ├── raw/  
│   │   └── EN\_Card\_Data.csv          \# Base card properties CSV (tracked)  
│   ├── processed/  
│   │   └── static\_feature\_table.pt   \# Precomputed (1267, 32\) tensor lookup  
│   ├── replays/                      \# Raw game logs from expert games / heuristics  
│   │   ├── random\_vs\_heuristic/  
│   │   └── self\_play\_v1/  
│   └── datasets/                     \# Preprocessed PyTorch Tensor Datasets  
│       ├── train\_bc\_v1.pt  
│       └── val\_bc\_v1.pt  
│  
├── models/                           \# Trained Checkpoints & Export Registry  
│   ├── checkpoints/                  \# Intermediary training checkpoints (.pt)  
│   │   └── bc\_mlp\_epoch10.pt  
│   └── registry/                     \# Validated model weights ready for deployment  
│       └── best\_policy\_v1.pt  
│  
├── notebooks/                        \# Research & Feature Analysis Notebooks  
│   ├── 01\_card\_dataset\_eda.ipynb  
│   └── 02\_trajectory\_inspection.ipynb  
│  
├── src/                              \# Core Source Module  
│   ├── \_\_init\_\_.py  
│   ├── card\_db/                      \# Static Card Feature Precomputing & Parsing  
│   │   ├── \_\_init\_\_.py  
│   │   ├── parser.py                 \# Parses EN\_Card\_Data.csv  
│   │   └── static\_table.py           \# Builds StaticCardFeatureTable tensor  
│   │  
│   ├── env/                          \# Environment Interface & Wrapper Layer  
│   │   ├── \_\_init\_\_.py  
│   │   ├── cabt\_wrapper.py           \# Standardized wrapper for Kaggle cabt engine  
│   │   └── state\_extractor.py        \# Converts obs.current \-\> feature dict  
│   │  
│   ├── models/                       \# PyTorch Neural Architectures  
│   │   ├── \_\_init\_\_.py  
│   │   ├── card\_encoder.py           \# Hybrid Card Encoder (Static \+ ID \+ Dynamic)  
│   │   ├── state\_mlp.py              \# Baseline MLP State Encoder  
│   │   ├── state\_transformer.py      \# Transformer State Encoder  
│   │   ├── action\_encoder.py         \# Action Matrix MLP Encoder  
│   │   └── policy\_net.py             \# Joint Scorer & Dynamic Action Scorer  
│   │  
│   ├── dataset/                      \# Data Loading & Trajectory Storage  
│   │   ├── \_\_init\_\_.py  
│   │   ├── trajectory\_buffer.py      \# Self-play experience collector  
│   │   └── dataset.py                \# PyTorch Dataset for BC / Imitation Learning  
│   │  
│   ├── training/                     \# Training Engine & Optimization  
│   │   ├── \_\_init\_\_.py  
│   │   ├── trainer\_bc.py             \# Behavioral Cloning Loss Loop  
│   │   └── trainer\_rl.py             \# PPO / Self-Play Training Loop  
│   │  
│   └── evaluation/                   \# Local Ladder & Agent Testing  
│       ├── \_\_init\_\_.py  
│       ├── arena.py                  \# Multi-threaded bot vs. bot local tournament  
│       └── heuristics.py             \# Rule-based baseline agents (Random, Greedy)  
│  
├── scripts/                          \# Pipeline Execution Entrypoints  
│   ├── build\_static\_features.py      \# Step 1: Precompute static card lookup table  
│   ├── generate\_replays.py           \# Step 2: Run heuristic bots to generate log datasets  
│   ├── process\_dataset.py            \# Step 3: Format logs into tensor training datasets  
│   ├── train\_bc.py                   \# Step 4: Run Behavioral Cloning training  
│   ├── run\_arena.py                  \# Step 5: Test candidate model against baselines  
│   └── package\_submission.py         \# Step 6: Package model \+ code into submission bundle  
│  
└── submission/                       \# Kaggle Deployment Sandbox  
    ├── build/                        \# Staging area for zip/tar bundle  
    └── submission.tar.gz             \# Final upload bundle submitted to Kaggle

## **2\. Pipeline Architecture & Data Management Workflow**

To scale from heuristic datasets to deep reinforcement learning, data management is divided into 4 sequential stages:  
 \[ Raw Card CSV \] ──► \[ Step 1: build\_static\_features.py \] ──► \[ static\_feature\_table.pt \]  
                                                                        │  
 \[ Engine & Heuristics \] ──► \[ Step 2: generate\_replays.py \] ──► \[ Raw Game Replays \]  
                                                                        │  
                                                                        ▼  
                                                   \[ Step 3: process\_dataset.py \]  
                                                                        │  
                                                                        ▼  
                                                   \[ PyTorch Datasets (.pt) \]  
                                                                        │  
                                                                        ▼  
                                                       \[ Step 4: train\_bc.py \]  
                                                                        │  
                                                                        ▼  
                                                  \[ Model Registry (best\_model.pt) \]  
                                                                        │  
                                                                        ▼  
                                                  \[ Step 5: package\_submission.py \]  
                                                                        │  
                                                                        ▼  
                                                     \[ submission.tar.gz \]

### **Stage A: Static Card Data Precomputing (src/card\_db/)**

> * **Input:** data/raw/EN\_Card\_Data.csv ($1,267$ unique cards).  
> * **Script:** python scripts/build\_static\_features.py  
> * **Output:** data/processed/static\_feature\_table.pt of shape $(1267, 32)$.  
> * **Purpose:** Ensures static features are built once offline so the agent does not parse strings or CSV rows during live competition turns.

### **Stage B: Data Generation & Replay Parsing (src/dataset/)**

> * **Input:** Heuristic agents playing matches in the local cabt engine.  
> * **Script:** python scripts/generate\_replays.py \--num\_games 1000 \--bot1 greedy \--bot2 random  
> * **Output:** Serialized JSON/MsgPack trajectory files in data/replays/.  
> * **Script:** python scripts/process\_dataset.py  
> * **Output:** Cleaned PyTorch tensor datasets stored in data/datasets/train\_bc\_v1.pt.

## **3\. Training & Experiment Tracking Architecture**

### **Config-Driven Experiments**

All training runs inherit parameters from configs/ using dataclasses or PyYAML. This eliminates hardcoded numbers in code:  
\# configs/train/bc\_trainer.yaml  
experiment\_name: "bc\_transformer\_baseline\_v1"  
seed: 42  
device: "cuda" \# Training happens on GPU; inference is CPU-optimized

data:  
  train\_path: "data/datasets/train\_bc\_v1.pt"  
  val\_path: "data/datasets/val\_bc\_v1.pt"  
  batch\_size: 64

model:  
  type: "transformer" \# Options: \["mlp", "transformer"\]  
  state\_embed\_dim: 256  
  action\_embed\_dim: 64  
  num\_transformer\_layers: 2  
  num\_heads: 4

optimizer:  
  lr: 0.0003  
  weight\_decay: 0.0001  
  epochs: 25

checkpoint:  
  save\_dir: "models/checkpoints/"  
  registry\_dir: "models/registry/"

### **Model Registry & Artifact Management**

> * Every training run generates checkpoint files in models/checkpoints/ containing:  
  * Model weights (state\_dict).  
  * Feature configuration metadata ($N\_{\\text{cards}}$, embedding dims).  
  * Validation accuracy/loss metrics.  
> * When a model beats the current benchmark in local evaluation (run\_arena.py), the checkpoint is promoted to models/registry/best\_policy.pt.

## **4\. Agent Submission Pipeline & Packaging (submission/)**

Kaggle requires a standalone submission bundle (submission.tar.gz or single file) that executes offline within strict memory and CPU time constraints.

### **The Self-Contained Deployment Strategy**

The scripts/package\_submission.py script automatically bundles all necessary dependencies into a deployment-ready directory without exposing full training code or raw replay datasets.  
 submission/build/  
 ├── main.py                    \# Entrypoint function agent(obs, config)  
 ├── static\_table.pt            \# Precomputed static card lookup tensor (1267, 32\)  
 ├── best\_model.pt              \# Trained PyTorch model weights (\~3 MB)  
 └── standalone\_agent.py        \# Inlined inference model code (No external dependencies)

### **Automated Submission Build Script (scripts/package\_submission.py)**

This script validates model size and inference execution speed before generating the archive:  
import os  
import tarfile  
import shutil  
import torch

BUILD\_DIR \= "submission/build"  
OUTPUT\_TAR \= "submission/submission.tar.gz"

def build\_package():  
    os.makedirs(BUILD\_DIR, exist\_ok=True)  
      
    \# 1\. Copy Model Registry Artifact  
    shutil.copy("models/registry/best\_policy.pt", os.path.join(BUILD\_DIR, "best\_model.pt"))  
      
    \# 2\. Copy Static Lookup Table  
    shutil.copy("data/processed/static\_feature\_table.pt", os.path.join(BUILD\_DIR, "static\_table.pt"))  
      
    \# 3\. Generate main.py Submission Entrypoint  
    with open(os.path.join(BUILD\_DIR, "main.py"), "w") as f:  
        f.write('''import os  
import torch  
from standalone\_agent import StandalonePTCGAgent

\# Global Agent Singleton (Loaded once on startup)  
AGENT \= None

def agent(obs, config):  
    global AGENT  
    if AGENT is None:  
        model\_path \= os.path.join(os.path.dirname(\_\_file\_\_), "best\_model.pt")  
        static\_table\_path \= os.path.join(os.path.dirname(\_\_file\_\_), "static\_table.pt")  
        AGENT \= StandalonePTCGAgent(model\_path, static\_table\_path)  
          
    return AGENT.act(obs)  
''')

    \# 4\. Copy Standalone Agent Source Code  
    shutil.copy("src/models/standalone\_agent.py", os.path.join(BUILD\_DIR, "standalone\_agent.py"))

    \# 5\. Compress into tar.gz  
    with tarfile.open(OUTPUT\_TAR, "w:gz") as tar:  
        tar.add(BUILD\_DIR, arcname=os.path.basename(BUILD\_DIR))  
          
    print(f"\[SUCCESS\] Submission packaged: {OUTPUT\_TAR} ({os.path.getsize(OUTPUT\_TAR) / 1024 / 1024:.2f} MB)")

if \_\_name\_\_ \== "\_\_main\_\_":  
    build\_package()

## **5\. Developer Command Reference Cheat Sheet**

Developers working on this repository should follow these standard command workflows:

| Stage | Command | Description |
| :---- | :---- | :---- |
| **1\. Setup** | pip install \-r requirements.txt | Install engine dependencies and PyTorch. |
| **2\. Feature Precompute** | python scripts/build\_static\_features.py | Generate static\_feature\_table.pt from CSV. |
| **3\. Data Generation** | python scripts/generate\_replays.py \--games 5000 | Run self-play / heuristic matches to log moves. |
| **4\. Process Dataset** | python scripts/process\_dataset.py | Format raw logs into PyTorch tensor training splits. |
| **5\. Train Model** | python scripts/train\_bc.py \--config configs/train/bc\_trainer.yaml | Train Behavioral Cloning model on GPU. |
| **6\. Arena Evaluation** | python scripts/run\_arena.py \--candidate models/checkpoints/bc\_epoch10.pt \--baseline heuristic | Evaluate win-rate of model against heuristics. |
| **7\. Package Agent** | python scripts/package\_submission.py | Build and verify submission.tar.gz for Kaggle upload. |

## **6\. Key Design Rules for Developers**

> 1. **Strict CPU Latency Budget:** All model forward passes in src/models/standalone\_agent.py must run on CPU in $\< 10\\text{ ms}$ per turn. Avoid heavy matrix allocations inside agent(obs, config).  
> 2. **Deterministic Precomputing:** Never parse EN\_Card\_Data.csv inside the live Kaggle turn loop. Read directly from static\_feature\_table.pt.  
> 3. **Modular Encoder Swapping:** Ensure state\_mlp.py and state\_transformer.py implement the exact same PyTorch interface: forward(obs\_dict) \-\> Tensor\[1, 256\].  
> 4. **Standalone Inference:** The code inside submission/build/standalone\_agent.py must be completely self-contained so it does not rely on local relative imports outside its deployment folder.