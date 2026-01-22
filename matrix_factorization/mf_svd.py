"""
Singular Value Decomposition (SVD) Matrix Factorization.

This module implements SVD-based matrix factorization using Surprise library's
SVD class, which uses SGD to optimize the matrix factorization objective.

SVD decomposes the rating matrix into lower-dimensional user and item factors,
enabling efficient prediction of missing ratings.

Reference: Guide Section 2.3 - Matrix Factorization
"""

from surprise import SVD

def train_svd_model(trainset, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, 
                   random_state=42, verbose=True):
    """
    Train SVD-based Matrix Factorization model.
    
    The model minimizes the same loss function as ALS:
    L = sum((r_ui - p_u^T q_i)^2) + λ(||p_u||^2 + ||q_i||^2)
    
    Uses Singular Value Decomposition via Surprise's SVD class.
    
    Args:
        trainset: Surprise Trainset object
        n_factors (int): Number of latent factors k (default: 50)
        n_epochs (int): Number of training epochs (default: 20)
        lr_all (float): Learning rate for all parameters (default: 0.005)
        reg_all (float): Regularization term for all parameters (default: 0.02)
        random_state (int): Random seed for reproducibility (default: 42)
        verbose (bool): Whether to print progress (default: True)
        
    Returns:
        SVD: Trained SVD model
    """
    if verbose:
        print("Training SVD Matrix Factorization...")
        print(f"  Latent factors (k): {n_factors}")
        print(f"  Learning rate: {lr_all}")
        print(f"  Regularization (lambda): {reg_all}")
        print(f"  Epochs: {n_epochs}")
    
    # Surprise's SVD uses SGD by default
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
        verbose=verbose
    )
    
    model.fit(trainset)
    
    if verbose:
        print("Training completed!")
    
    return model


class SVDMatrixFactorization:
    """
    Wrapper class for SVD-based matrix factorization.
    
    This is essentially a convenience wrapper around Surprise's SVD class.
    """
    
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, 
                 random_state=42):
        """
        Initialize SVD Matrix Factorization.
        
        Args:
            n_factors (int): Number of latent factors
            n_epochs (int): Number of training epochs
            lr_all (float): Learning rate
            reg_all (float): Regularization parameter
            random_state (int): Random seed
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.model = None
        
    def fit(self, trainset, verbose=True):
        """
        Train the SVD model.
        
        Args:
            trainset: Surprise Trainset object
            verbose (bool): Whether to print progress
        """
        self.model = train_svd_model(
            trainset,
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
            verbose=verbose
        )
    
    def predict(self, uid, iid):
        """
        Predict rating for a user-item pair.
        
        Args:
            uid: Raw user id
            iid: Raw item id
            
        Returns:
            float: Predicted rating
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        prediction = self.model.predict(uid, iid)
        return prediction.est
    
    def test(self, testset):
        """Make predictions on a test set."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.test(testset)


if __name__ == "__main__":
    # Example usage
    from data_loader import load_movielens_100k, get_train_test_split
    
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data)
    
    # Train SVD model
    print("\n" + "=" * 60)
    svd_model = train_svd_model(trainset, n_factors=50, n_epochs=20)
    
    # Sample prediction
    sample_uid = testset[0][0]
    sample_iid = testset[0][1]
    sample_rating = testset[0][2]
    
    svd_pred = svd_model.predict(sample_uid, sample_iid)
    
    print(f"\nSample prediction:")
    print(f"  User: {sample_uid}, Item: {sample_iid}")
    print(f"  True rating: {sample_rating:.2f}")
    print(f"  Predicted rating: {svd_pred:.2f}")
    print(f"  Error: {abs(sample_rating - svd_pred):.2f}")
