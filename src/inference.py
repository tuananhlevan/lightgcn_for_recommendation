import os
import torch
import pandas as pd

from config import Config
from model import SimpleLightGCN
from data_pipeline import load_and_prep_movielens

def get_movie_title_mapping(movies_path='data/movies.csv'):
    """Loads the MovieLens movies.csv to map raw movieId to Titles."""
    try:
        movies_df = pd.read_csv(movies_path)
        return dict(zip(movies_df['movieId'], movies_df['title']))
    except FileNotFoundError:
        print(f"Warning: {movies_path} not found. Returning raw IDs instead of titles.")
        return {}

def recommend_for_user(raw_user_id, top_k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Reload the data to get the exact same graph and mappings
    train_edge_index, _, num_users, num_items = load_and_prep_movielens()
    train_edge_index = train_edge_index.to(device)
    
    # Exact DataFrames to map raw IDs -> model indices -> raw IDs
    df = pd.read_csv(Config.DATA_PATH)
    df = df[df['rating'] >= 3.0].copy()
    user_uniques = pd.factorize(df['userId'])[1]
    item_uniques = pd.factorize(df['movieId'])[1]
    
    # Check if the user exists
    if raw_user_id not in user_uniques:
        print(f"User {raw_user_id} not found in training data.")
        return
        
    model_user_idx = user_uniques.get_loc(raw_user_id)
    movie_titles = get_movie_title_mapping()
    
    # 2. Load the best model checkpoint
    model = SimpleLightGCN(num_users, num_items).to(device)
    ckpt_path = os.path.join(Config.CKPT_DIR, "best_model.pth")
    
    if not os.path.exists(ckpt_path):
        print("No checkpoint found. Please train the model first.")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Perform Inference
    with torch.no_grad():
        # Do the graph hops to get final embeddings
        user_embs, item_embs = model(train_edge_index)
        
        # Grab our target user
        target_user_vector = user_embs[model_user_idx]
        
        # Matrix multiplication to score all movies
        all_item_scores = torch.matmul(item_embs, target_user_vector)
        
        # Mask out history (movies they already watched)
        user_history_indices = train_edge_index[1][train_edge_index[0] == model_user_idx]
        all_item_scores[user_history_indices] = -float('inf')
        
        # Get Top K
        scores, top_k_indices = torch.topk(all_item_scores, top_k)
        
    # 4. Map back to human-readable format
    print(f"\n--- Top {top_k} Recommendations for User {raw_user_id} ---")
    for rank, (score, model_item_idx) in enumerate(zip(scores, top_k_indices)):
        raw_movie_id = item_uniques[model_item_idx.item()]
        title = movie_titles.get(raw_movie_id, f"Unknown Movie ID {raw_movie_id}")
        print(f"{rank + 1}. {title} (Match Score: {score.item():.4f})")