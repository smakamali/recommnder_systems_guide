"""
Educational ALS Matrix Factorization Implementation from Scratch.

This module implements the ALS algorithm directly from the guide's pseudocode
(lines 370-399 in README.md) using only NumPy for educational purposes.

This implementation helps understand the core mathematics behind ALS without
the abstraction of libraries.

Reference: Guide Section 2.3 - Matrix Factorization, ALS Algorithm (lines 370-399)
Loss Function: L = sum((r_ui - p_u^T q_i)^2) + λ(||p_u||^2 + ||q_i||^2)
"""

import numpy as np
from data_loader import load_movielens_100k, get_train_test_split


def als_matrix_factorization(trainset, k=50, lambda_reg=0.1, iterations=50, 
                             random_state=42, verbose=True):
    """
    ALS Matrix Factorization implementation directly from guide pseudocode.
    
    This is the exact algorithm described in the guide (lines 372-399).
    
    Algorithm:
    1. Initialize P (k × num_users) and Q (k × num_items) randomly
    2. For each iteration:
       a. Fix Q, optimize P (solve for each user u)
       b. Fix P, optimize Q (solve for each item i)
    
    Args:
        trainset: Surprise Trainset object
        k (int): Number of latent factors (default: 50)
        lambda_reg (float): Regularization parameter λ (default: 0.1)
        iterations (int): Number of iterations (default: 50)
        random_state (int): Random seed (default: 42)
        verbose (bool): Print progress (default: True)
        
    Returns:
        tuple: (P, Q) where P is k × n_users, Q is k × n_items
    """
    np.random.seed(random_state)
    
    num_users = trainset.n_users
    num_items = trainset.n_items
    
    # Initialize user and item factors randomly (guide lines 374-375)
    # P: k × num_users, Q: k × num_items
    P = np.random.normal(0, 0.1, size=(k, num_users))
    Q = np.random.normal(0, 0.1, size=(k, num_items))
    
    # Build efficient data structures for access
    # user_items[u] = [(item_id, rating), ...]
    # item_users[i] = [(user_id, rating), ...]
    user_items = {}
    item_users = {}
    
    for uid, iid, rating in trainset.all_ratings():
        u = trainset.to_inner_uid(uid)
        i = trainset.to_inner_iid(iid)
        
        if u not in user_items:
            user_items[u] = []
        user_items[u].append((i, rating))
        
        if i not in item_users:
            item_users[i] = []
        item_users[i].append((u, rating))
    
    # ALS iterations (guide lines 377-396)
    for iteration in range(iterations):
        # Fix Q, optimize P (guide lines 378-386)
        for u in range(num_users):
            if u not in user_items:
                continue
            
            # Get items rated by user u (guide lines 380-381)
            rated_items = [item_rating[0] for item_rating in user_items[u]]
            r_u = np.array([item_rating[1] for item_rating in user_items[u]])
            
            # Q_u: k × |rated_items| (guide line 381)
            Q_u = Q[:, rated_items]
            
            # Solve: (Q_u @ Q_u^T + λI) * p_u = Q_u @ r_u (guide lines 383-386)
            # This is the closed-form solution for p_u when Q is fixed
            P[:, u] = np.linalg.solve(
                Q_u @ Q_u.T + lambda_reg * np.eye(k),
                Q_u @ r_u
            )
        
        # Fix P, optimize Q (guide lines 388-396)
        for i in range(num_items):
            if i not in item_users:
                continue
            
            # Get users who rated item i (guide lines 390-391)
            users_rated = [user_rating[0] for user_rating in item_users[i]]
            r_i = np.array([user_rating[1] for user_rating in item_users[i]])
            
            # P_i: k × |users_rated| (guide line 391)
            P_i = P[:, users_rated]
            
            # Solve: (P_i @ P_i^T + λI) * q_i = P_i @ r_i (guide lines 393-396)
            # This is the closed-form solution for q_i when P is fixed
            Q[:, i] = np.linalg.solve(
                P_i @ P_i.T + lambda_reg * np.eye(k),
                P_i @ r_i
            )
        
        if verbose and (iteration + 1) % 10 == 0:
            # Calculate current loss for monitoring
            loss = calculate_loss(trainset, P, Q, lambda_reg)
            print(f"  Iteration {iteration + 1}/{iterations}, Loss: {loss:.4f}")
    
    if verbose:
        print(f"  Final loss: {calculate_loss(trainset, P, Q, lambda_reg):.4f}")
    
    return P, Q


def calculate_loss(trainset, P, Q, lambda_reg):
    """
    Calculate the regularized squared error loss.
    
    Loss = sum((r_ui - p_u^T q_i)^2) + λ(||p_u||^2 + ||q_i||^2)
    (matching guide line 363)
    
    Args:
        trainset: Surprise Trainset object
        P: User factors (k × n_users)
        Q: Item factors (k × n_items)
        lambda_reg: Regularization parameter
        
    Returns:
        float: Total loss
    """
    error_sum = 0.0
    
    for uid, iid, rating in trainset.all_ratings():
        u = trainset.to_inner_uid(uid)
        i = trainset.to_inner_iid(iid)
        
        # Predicted rating: p_u^T * q_i (guide line 344)
        prediction = P[:, u].dot(Q[:, i])
        
        # Squared error
        error_sum += (rating - prediction) ** 2
    
    # Regularization terms
    reg_term = lambda_reg * (np.sum(P ** 2) + np.sum(Q ** 2))
    
    return error_sum + reg_term


def predict_rating(P, Q, trainset, uid, iid):
    """
    Predict rating for a user-item pair.
    
    Prediction: r_hat = p_u^T * q_i (guide line 344)
    
    Args:
        P: User factors (k × n_users)
        Q: Item factors (k × n_items)
        trainset: Surprise Trainset object
        uid: Raw user id
        iid: Raw item id
        
    Returns:
        float: Predicted rating
    """
    try:
        u = trainset.to_inner_uid(uid)
        i = trainset.to_inner_iid(iid)
        prediction = P[:, u].dot(Q[:, i])
        # Clip to valid rating range
        return np.clip(prediction, 1.0, 5.0)
    except ValueError:
        # Cold start - return global mean
        return trainset.global_mean


class ALSFromScratch:
    """
    Educational wrapper class for ALS from scratch.
    """
    
    def __init__(self, k=50, lambda_reg=0.1, iterations=50, random_state=42):
        """
        Initialize ALS from scratch.
        
        Args:
            k (int): Number of latent factors
            lambda_reg (float): Regularization parameter
            iterations (int): Number of iterations
            random_state (int): Random seed
        """
        self.k = k
        self.lambda_reg = lambda_reg
        self.iterations = iterations
        self.random_state = random_state
        self.P = None
        self.Q = None
        self.trainset = None
    
    def fit(self, trainset, verbose=True):
        """
        Train the model.
        
        Args:
            trainset: Surprise Trainset object
            verbose (bool): Print progress
        """
        self.trainset = trainset
        print(f"Training ALS from scratch (guide pseudocode implementation)...")
        print(f"  Latent factors (k): {self.k}")
        print(f"  Regularization (lambda): {self.lambda_reg}")
        print(f"  Iterations: {self.iterations}")
        
        self.P, self.Q = als_matrix_factorization(
            trainset,
            k=self.k,
            lambda_reg=self.lambda_reg,
            iterations=self.iterations,
            random_state=self.random_state,
            verbose=verbose
        )
        
        print("Training completed!")
    
    def predict(self, uid, iid):
        """
        Predict rating for a user-item pair.
        
        Args:
            uid: Raw user id
            iid: Raw item id
            
        Returns:
            float: Predicted rating
        """
        if self.P is None or self.Q is None:
            raise ValueError("Model must be trained before making predictions")
        
        return predict_rating(self.P, self.Q, self.trainset, uid, iid)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Educational ALS Matrix Factorization from Scratch")
    print("Following guide pseudocode (README.md lines 370-399)")
    print("=" * 60)
    
    print("\nLoading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data)
    
    # Train model
    model = ALSFromScratch(k=50, lambda_reg=0.1, iterations=50)
    model.fit(trainset, verbose=True)
    
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

