import os
import torch
import pandas as pd

from config import Config
from model import SimpleLightGCN
from data_pipeline import load_and_prep_movielens

def get_movie_info_mapping(movies_path='data/movies.csv'):
    """Loads the MovieLens movies.csv to map raw movieId to Titles and Genres."""
    try:
        movies_df = pd.read_csv(movies_path)
        # Create a dictionary where Key = movieId, Value = {'title': ..., 'genres': ...}
        return movies_df.set_index('movieId')[['title', 'genres']].to_dict('index')
    except FileNotFoundError:
        print(f"Warning: {movies_path} not found. Returning raw IDs instead.")
        return {}
    except KeyError:
        print("Warning: 'genres' column not found in movies.csv. Are you using the correct file?")
        return {}

def recommend_for_user(raw_user_id, top_k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Rebuild the graph
    print("Rebuilding graph for inference...")
    train_edge_index, _, num_users, num_items = load_and_prep_movielens(Config.SPLIT_TYPE)
    train_edge_index = train_edge_index.to(device)
    
    # Exact DataFrame mapping
    df = pd.read_csv(Config.DATA_PATH)
    df = df[df['rating'] >= 3.0].copy()
    user_uniques = pd.factorize(df['userId'])[1]
    item_uniques = pd.factorize(df['movieId'])[1]
    
    if raw_user_id not in user_uniques:
        print(f"User {raw_user_id} not found in training data.")
        return
        
    model_user_idx = user_uniques.get_loc(raw_user_id)
    
    # Use our new mapping function
    movie_info = get_movie_info_mapping()
    
    # 2. Load Checkpoint
    model = SimpleLightGCN(num_users, num_items).to(device)
    ckpt_path = os.path.join(Config.CKPT_DIR, "best_model.pth")
    
    if not os.path.exists(ckpt_path):
        print("No checkpoint found. Please train the model first.")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Inference
    with torch.no_grad():
        bipartite_edges = model.get_graph(train_edge_index)
        user_embs, item_embs, _ = model(bipartite_edges)
        target_user_vector = user_embs[model_user_idx]
        
        all_item_scores = torch.matmul(item_embs, target_user_vector)
        
        user_history_indices = train_edge_index[1][train_edge_index[0] == model_user_idx]
        all_item_scores[user_history_indices] = -float('inf')
        
        scores, top_k_indices = torch.topk(all_item_scores, top_k)
        
    # 4. Print Output with Genres
    print(f"\n--- Top {top_k} Recommendations for User {raw_user_id} ---")
    for rank, (score, model_item_idx) in enumerate(zip(scores, top_k_indices)):
        raw_movie_id = item_uniques[model_item_idx.item()]
        
        # Safely fetch the title and genre
        info = movie_info.get(raw_movie_id, {'title': f"Unknown ID {raw_movie_id}", 'genres': 'Unknown'})
        title = info['title']
        genres = info['genres']
        
        # Format the print to look clean in the terminal
        print(f"{rank + 1:02d}. {title[:40]:<42} | Genres: {genres:<35} | Score: {score.item():.4f}")