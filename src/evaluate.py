import torch
import numpy as np
from collections import defaultdict

def evaluate_metrics_at_k(model, train_edge_index, test_edge_index, k=20):
    """
    Evaluates the model and returns Recall@K, NDCG@K, MRR@K, and Precision@K.
    """
    model.eval()
    
    test_user_dict = defaultdict(list)
    for i in range(test_edge_index.shape[1]):
        u = test_edge_index[0, i].item()
        item = test_edge_index[1, i].item()
        test_user_dict[u].append(item)
        
    with torch.no_grad():
        user_embs, item_embs = model(train_edge_index)
        all_scores = torch.matmul(user_embs, item_embs.T)
        
        train_users = train_edge_index[0]
        train_items = train_edge_index[1]
        all_scores[train_users, train_items] = -float('inf')
        
        _, top_k_indices = torch.topk(all_scores, k, dim=1)
        top_k_indices = top_k_indices.cpu().numpy()
        
        # Lists to store the metrics for each user
        recalls = []
        ndcgs = []
        mrrs = []
        precisions = []
        
        for u, target_items in test_user_dict.items():
            user_top_k = top_k_indices[u]
            
            hits = 0
            dcg = 0.0
            first_hit_rank = None
            
            for rank, rec_item in enumerate(user_top_k):
                if rec_item in target_items:
                    hits += 1
                    dcg += 1.0 / np.log2(rank + 2)
                    
                    # Capture the rank of the VERY FIRST hit for MRR
                    # (rank + 1) because enumerate starts at 0, but position 1 is rank 1
                    if first_hit_rank is None:
                        first_hit_rank = rank + 1
                        
            # 1. Recall@K: Out of the user's hidden test items, how many did we find?
            recall = hits / min(len(target_items), k)
            recalls.append(recall)
            
            # 2. Precision@K: Out of the K items we recommended, how many were good?
            precision = hits / k
            precisions.append(precision)
            
            # 3. MRR@K: 1 / (Rank of first correct guess). 0 if no correct guesses.
            mrr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
            mrrs.append(mrr)
            
            # 4. NDCG@K
            idcg = 0.0
            num_ideal_hits = min(len(target_items), k)
            for i in range(num_ideal_hits):
                idcg += 1.0 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
            
    # Return the mean scores
    return np.mean(recalls), np.mean(ndcgs), np.mean(mrrs), np.mean(precisions)