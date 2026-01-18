"""
Alternating Least Squares (ALS) Matrix Factorization using Implicit Library.

This module implements ALS-based matrix factorization using the implicit library,
which provides an efficient ALS implementation for collaborative filtering.

The ALS algorithm alternates between fixing item factors Q and optimizing user
factors P, then fixing P and optimizing Q. This converges to a solution that
minimizes the regularized squared error loss.

Reference: Guide Section 2.3 - Matrix Factorization, ALS Algorithm (lines 368-399)
"""

import numpy as np
from scipy.sparse import csr_matrix
import implicit


class ALSMatrixFactorization:
    """
    Alternating Least Squares Matrix Factorization using implicit library.
    
    Wrapper around implicit.als.AlternatingLeastSquares that works with
    Surprise Trainset format for consistency with other modules.
    
    Implements the ALS algorithm as described in the guide:
    - Loss: L = sum((r_ui - p_u^T q_i)^2) + λ(||p_u||^2 + ||q_i||^2)
    - Alternates between optimizing P (with Q fixed) and Q (with P fixed)
    """
    
    def __init__(self, n_factors=50, reg=0.1, n_iter=50, random_state=42):
        """
        Initialize ALS Matrix Factorization.
        
        Args:
            n_factors (int): Number of latent factors k (default: 50)
            reg (float): Regularization parameter λ (default: 0.1)
            n_iter (int): Number of iterations (default: 50)
            random_state (int): Random seed for reproducibility (default: 42)
        """
        self.n_factors = n_factors
        self.reg = reg
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.trainset = None
        self.user_mapping = None  # Maps Surprise inner_uid -> implicit user_idx
        self.item_mapping = None  # Maps Surprise inner_iid -> implicit item_idx
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        
    def fit(self, trainset):
        """
        Train the ALS model on the training set.
        
        Args:
            trainset: Surprise Trainset object
        """
        self.trainset = trainset
        np.random.seed(self.random_state)
        
        # Initialize implicit ALS model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.reg,
            iterations=self.n_iter,
            random_state=self.random_state,
            use_gpu=False  # Set to True if GPU available
        )
        
        # Build sparse user-item matrix from trainset
        n_users = trainset.n_users
        n_items = trainset.n_items
        
        # Create mappings between Surprise inner IDs and implicit indices
        # (They should match, but we'll be explicit)
        self.user_mapping = {i: i for i in range(n_users)}
        self.item_mapping = {i: i for i in range(n_items)}
        self.reverse_user_mapping = {i: i for i in range(n_users)}
        self.reverse_item_mapping = {i: i for i in range(n_items)}
        
        # Build sparse matrix: rows = users, columns = items
        row_indices = []
        col_indices = []
        values = []
        
        for uid, iid, rating in trainset.all_ratings():
            u = trainset.to_inner_uid(uid)
            i = trainset.to_inner_iid(iid)
            
            row_indices.append(u)
            col_indices.append(i)
            # implicit library works with confidence/preference values
            # For explicit ratings, we can use the rating directly
            values.append(float(rating))
        
        # Create CSR (Compressed Sparse Row) matrix
        user_item_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_users, n_items)
        )
        
        # Train the model
        print(f"  Training with {user_item_matrix.nnz} ratings...")
        self.model.fit(user_item_matrix)
        
    def predict(self, uid, iid, verbose=False):
        """
        Predict rating for a user-item pair.
        
        Args:
            uid: Raw user id (as in dataset)
            iid: Raw item id (as in dataset)
            verbose: If True, return prediction details
            
        Returns:
            float: Predicted rating
        """
        if self.model is None or self.trainset is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to inner ids
        try:
            u_inner = self.trainset.to_inner_uid(uid)
            i_inner = self.trainset.to_inner_iid(iid)
        except ValueError:
            # Cold start - return global mean or default
            return self.trainset.global_mean
        
        # Get user and item factors from the model
        user_factors = self.model.user_factors[u_inner]
        item_factors = self.model.item_factors[i_inner]
        
        # Prediction: r_hat = p_u^T * q_i (guide line 344)
        prediction = np.dot(user_factors, item_factors)
        
        # Clip to rating scale (typically 1-5 for MovieLens)
        prediction = np.clip(prediction, 1.0, 5.0)
        
        return float(prediction)
    
    def get_user_factors(self, uid):
        """Get latent factors for a user."""
        if self.model is None or self.trainset is None:
            raise ValueError("Model must be trained first")
        try:
            u_inner = self.trainset.to_inner_uid(uid)
            return self.model.user_factors[u_inner].copy()
        except ValueError:
            return None
    
    def get_item_factors(self, iid):
        """Get latent factors for an item."""
        if self.model is None or self.trainset is None:
            raise ValueError("Model must be trained first")
        try:
            i_inner = self.trainset.to_inner_iid(iid)
            return self.model.item_factors[i_inner].copy()
        except ValueError:
            return None


def train_als_model(trainset, n_factors=50, reg=0.1, n_iter=50, random_state=42):
    """
    Convenience function to train an ALS model using implicit library.
    
    Args:
        trainset: Surprise Trainset object
        n_factors (int): Number of latent factors
        reg (float): Regularization parameter
        n_iter (int): Number of iterations
        random_state (int): Random seed
        
    Returns:
        ALSMatrixFactorization: Trained model
    """
    print(f"Training ALS Matrix Factorization (using implicit library)...")
    print(f"  Latent factors (k): {n_factors}")
    print(f"  Regularization (λ): {reg}")
    print(f"  Iterations: {n_iter}")
    
    model = ALSMatrixFactorization(
        n_factors=n_factors,
        reg=reg,
        n_iter=n_iter,
        random_state=random_state
    )
    model.fit(trainset)
    
    print("Training completed!")
    return model


if __name__ == "__main__":
    # Example usage
    from data_loader import load_movielens_100k, get_train_test_split
    
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data)
    
    # Train ALS model
    model = train_als_model(trainset, n_factors=50, reg=0.1, n_iter=50)
    
    # Test prediction
    sample_uid = testset[0][0]
    sample_iid = testset[0][1]
    sample_rating = testset[0][2]
    
    prediction = model.predict(sample_uid, sample_iid)
    print(f"\nSample prediction:")
    print(f"  User: {sample_uid}, Item: {sample_iid}")
    print(f"  True rating: {sample_rating:.2f}")
    print(f"  Predicted rating: {prediction:.2f}")
    print(f"  Error: {abs(sample_rating - prediction):.2f}")
