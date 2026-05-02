# LightGCN Recommender System

A highly optimized PyTorch Geometric implementation of **LightGCN** (Graph Convolutional Networks for Recommendation), designed to predict user-item interactions. 

This project trains a collaborative filtering model on the MovieLens-1M dataset. It accurately implements the simplified graph convolution matrix and base-embedding L2 regularization specified in the original LightGCN paper, optimized with batched tensor operations and asynchronous multiprocessing to run efficiently on modern hardware.

## Benchmark Results
The model was evaluated using a strict 5-core interaction filter. To ensure statistical significance and stability, the pipeline was run 5 independent times. A more comprehensive comparison can be found at the repo [ml-rec-sys](https://github.com/tuananhlevan/ml-rec-sys)

| Metrics | Result |
| :---: | :---: |
| NDCG@20 | 0.3578 ± 0.0011 |
| Recall@20 | 0.2604 ± 0.0012 |
| MRR@20 | 0.6077 ± 0.0022 |
| Precision@20 | 0.2630 ± 0.0009 |

## Features
* **Vectorized BPR Data Pipeline:** Uses PyTorch DataLoaders with multiprocessing to fetch negative samples continuously without blocking the GPU (Only available on Linux-based operating systems)
* **Memory-Safe Evaluation:** Computes NDCG@K, Recall@K, Precision@K, and MRR@K in chunks, preventing Out-Of-Memory (OOM) crashes on large datasets
* **Pre-computed Graph Normalization:** Pre-computes the undirected bipartite graph prior to the training loop to save GPU compute cycles
* **Data Integrity Filtering:** Implements a strict $k$-core filter (minimum 5 interactions) to safely construct robust training and test splits
* **Clean CLI:** Simple command-line interface for both training the model and generating human-readable inference lists

## Project Structure
```text
├── data/
│   └── get_csv.py           # Convert original ml-1m .dat files to the corresponding .csv files
├── ckpt/                    # Directory for saved model weights
├── src/
│   ├── __init__.py
│   ├── config.py            # Global hyperparameters
│   ├── data_pipeline.py     # Dataset class and splitting logic
│   ├── evaluate.py          # Batched metric calculations
│   ├── inference.py         # Generation of Top-K recommendations
│   ├── model.py             # LightGCN architecture using PyTorch Geometric
│   └── train.py             # Main training loop
├── main.py                  # CLI entry point
├── environment.yml          # Configuration file for the required conda environment
└── README.md
```

## Environment
You can install the environment needed to run this repo using the commands below (You can skip this step and use the parent repo environment if installed)
```bash
conda env create -f environment.yml
conda activate lightgcn
```

## Prepare Data
Run the below command to download and preprocess the dataset MovieLens-1M
```bash
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
python data/get_csv.py --source ml-1m/ --output data/
```

## Usage
This project is controlled via a central command-line interface

### 1. Training the Model
To initialize the training pipeline from scratch, use the `--train` flag. This will load the data, build the graph, and begin the epoch loop, automatically saving the best model state to the `ckpt/` directory
```bash
python main.py --train
```

### 2. Running inference
To generate recommendations for a specific user, use the `--infer` flag followed by the User ID. The system will load the best checkpoint, compute the user's affinity against all items, mask their historical training interactions, and output a ranked list
```bash
python main.py --infer 105
```
You can optionally control how many recommendations are returned using the `--top_k` flag (defaults to 5)
```bash
python main.py --infer 105 --top_k 10
```