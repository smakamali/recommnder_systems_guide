"""
Training module for GraphSAGE with BPR loss.

Implements Bayesian Personalized Ranking (BPR) loss for training GraphSAGE
on implicit feedback (ratings) with negative sampling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) loss.
    
    Formula: L = -log(σ(r_ui - r_uj)) + λ||θ||²
    where:
    - r_ui: score for positive pair (user i, item u)
    - r_uj: score for negative pair (user i, item j)
    - σ: sigmoid function
    - λ: regularization parameter
    
    Reference: GNN guide lines 379-389 (BPR Loss section).
    """
    
    def __init__(self, reg_lambda=0.01):
        """
        Initialize BPR loss.
        
        Args:
            reg_lambda: L2 regularization parameter (default: 0.01)
        """
        super(BPRLoss, self).__init__()
        self.reg_lambda = reg_lambda
    
    def forward(self, pos_scores, neg_scores, model_params=None):
        """
        Compute BPR loss.
        
        Args:
            pos_scores: Scores for positive pairs (batch_size,)
            neg_scores: Scores for negative pairs (batch_size,)
            model_params: Optional model parameters for regularization
            
        Returns:
            torch.Tensor: BPR loss value
        """
        # BPR loss: -log(σ(pos_score - neg_score))
        diff = pos_scores - neg_scores
        loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))
        
        # Add L2 regularization if model parameters provided
        if model_params is not None and self.reg_lambda > 0:
            l2_reg = sum(p.pow(2.0).sum() for p in model_params)
            loss += self.reg_lambda * l2_reg
        
        return loss


def train_graphsage_model(model, graph_data, trainset, user_id_to_idx, item_id_to_idx,
                          num_epochs=20, batch_size=512, learning_rate=0.001,
                          num_negatives=1, device='cpu', verbose=True):
    """
    Train GraphSAGE model with BPR loss.
    
    Args:
        model: GraphSAGERecommender instance
        graph_data: PyTorch Geometric Data object
        trainset: Surprise Trainset
        user_id_to_idx: Dict mapping user_id -> node index
        item_id_to_idx: Dict mapping item_id -> node index
        num_epochs: Number of training epochs (default: 20)
        batch_size: Batch size for training (default: 512)
        learning_rate: Learning rate (default: 0.001)
        num_negatives: Number of negative samples per positive (default: 1)
        device: Device to train on (default: 'cpu')
        verbose: Print training progress (default: True)
        
    Returns:
        dict: Training history with losses per epoch
    """
    # Move model and data to device
    model = model.to(device)
    graph_data = graph_data.to(device)
    
    # Build positive pairs and negative sampling pool
    positive_pairs = []
    user_positive_items = defaultdict(set)
    
    for inner_uid, inner_iid, rating in trainset.all_ratings():
        uid = trainset.to_raw_uid(inner_uid)
        iid = trainset.to_raw_iid(inner_iid)
        
        if uid not in user_id_to_idx or iid not in item_id_to_idx:
            continue
        
        user_idx = user_id_to_idx[uid]
        item_idx = item_id_to_idx[iid]
        
        positive_pairs.append((user_idx, item_idx))
        user_positive_items[user_idx].add(item_idx)
    
    # Get all item indices for negative sampling
    all_item_indices = list(item_id_to_idx.values())
    num_items = len(all_item_indices)
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = BPRLoss(reg_lambda=0.01)
    
    # Training history
    history = {
        'train_loss': [],
        'epoch': []
    }
    
    if verbose:
        print(f"\nTraining GraphSAGE model...")
        print(f"  Positive pairs: {len(positive_pairs):,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Device: {device}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle positive pairs
        indices = np.random.permutation(len(positive_pairs))
        
        # Process in batches
        for batch_start in range(0, len(positive_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(positive_pairs))
            batch_indices = indices[batch_start:batch_end]
            
            # Get batch of positive pairs
            batch_users = []
            batch_pos_items = []
            batch_neg_items = []
            
            for idx in batch_indices:
                user_idx, pos_item_idx = positive_pairs[idx]
                
                # Sample negative items
                neg_samples = []
                attempts = 0
                while len(neg_samples) < num_negatives and attempts < 100:
                    neg_item_idx = np.random.choice(all_item_indices)
                    if neg_item_idx not in user_positive_items[user_idx]:
                        neg_samples.append(neg_item_idx)
                    attempts += 1
                
                # If couldn't find enough negatives, use random (may include positives)
                while len(neg_samples) < num_negatives:
                    neg_samples.append(np.random.choice(all_item_indices))
                
                batch_users.append(user_idx)
                batch_pos_items.append(pos_item_idx)
                batch_neg_items.extend(neg_samples[:num_negatives])
            
            # Forward pass: get embeddings
            user_emb, item_emb = model(graph_data)
            
            # Get scores for positive and negative pairs
            batch_users_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            batch_pos_items_tensor = torch.tensor(batch_pos_items, dtype=torch.long, device=device)
            batch_neg_items_tensor = torch.tensor(batch_neg_items, dtype=torch.long, device=device)
            
            # Expand users for multiple negatives
            if num_negatives > 1:
                batch_users_tensor = batch_users_tensor.repeat_interleave(num_negatives)
            
            pos_scores = model.predict(user_emb, item_emb, batch_users_tensor, batch_pos_items_tensor)
            neg_scores = model.predict(user_emb, item_emb, batch_users_tensor, batch_neg_items_tensor)
            
            # Compute loss
            loss = criterion(pos_scores, neg_scores, model.parameters())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        history['train_loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
    
    if verbose:
        print("\nTraining completed!")
    
    return history


if __name__ == "__main__":
    # Test training
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gnn.graph_data_loader import build_bipartite_graph
    from gnn.graphsage_model import GraphSAGERecommender
    from common.data_loader import (load_movielens_100k, get_train_test_split,
                                    load_user_features, load_item_features)
    
    print("Loading data...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    user_features = load_user_features()
    item_features = load_item_features()
    
    print("Building graph...")
    graph_data, preprocessor, user_id_to_idx, item_id_to_idx = build_bipartite_graph(
        trainset, user_features, item_features
    )
    
    print("Initializing model...")
    user_feat_dim = graph_data.x[graph_data.node_type == 0].size(1)
    item_feat_dim = graph_data.x[graph_data.node_type == 1].size(1)
    
    model = GraphSAGERecommender(
        num_users=graph_data.num_users,
        num_items=graph_data.num_items,
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        hidden_dim=64,
        num_layers=2
    )
    
    print("Training model (2 epochs for testing)...")
    history = train_graphsage_model(
        model, graph_data, trainset, user_id_to_idx, item_id_to_idx,
        num_epochs=2, batch_size=256, learning_rate=0.001,
        device='cpu', verbose=True
    )
    
    print(f"\nFinal loss: {history['train_loss'][-1]:.4f}")
    print("Training test successful!")
