import os

class Config:
    # --- File Paths ---
    DATA_PATH = "data/ratings.csv"
    CKPT_DIR = "ckpt/"

    # --- Data & Graph Parameters ---
    TEST_SIZE = 0.2
    SPLIT_TYPE = "random"   # "temporal" for time-sequential split, "random" for random split
    
    # --- Model Hyperparameters ---
    EMB_DIM = 64
    NUM_LAYERS = 3

    # --- Training Hyperparameters ---
    EPOCHS = 300
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0
    EVAL_EPOCH = 20         # How often we calculate Recall@20
    SAVE_EPOCH = 50         # How often do we save model routinely
    
    # --- Evaluate Hyperparameters ---
    K = 20                  # Top K

    # Ensure output directories exist when config is loaded
    os.makedirs(CKPT_DIR, exist_ok=True)