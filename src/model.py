import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
from torch_geometric.utils import to_undirected

from config import Config

class SimpleLightGCN(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.num_layers = Config.NUM_LAYERS
        
        self.embedding = nn.Embedding(num_users + num_items, Config.EMB_DIM)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
        self.conv = LGConv()

    def forward(self, edge_index):
        # Offset the item IDs
        users = edge_index[0]
        items = edge_index[1] + self.num_users
        bipartite_edges = torch.stack([users, items], dim=0)
        
        # Make the graph undirected
        bipartite_edges = to_undirected(bipartite_edges)
        
        x = self.embedding.weight
        embs = [x]
        
        for _ in range(self.num_layers):
            x = self.conv(x, bipartite_edges)
            embs.append(x)
            
        out = torch.mean(torch.stack(embs, dim=0), dim=0)
        user_embs, item_embs = torch.split(out, [self.num_users, self.num_items])
        
        return user_embs, item_embs