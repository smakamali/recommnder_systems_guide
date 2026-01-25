"""
Factorization Machines for Feature-Rich Recommendation.

Implements FM using myFM library with support for user and item features.
Handles feature engineering, training, prediction, and evaluation.

Reference: feature_extensions.md lines 70-193
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from surprise import Prediction
import scipy.sparse as sp


class FactorizationMachineModel:
    """
    Factorization Machine wrapper for recommendation.
    
    Uses myFM's Bayesian FM implementation with Gibbs sampling for training.
    """
    
    def __init__(self, n_factors=50, learning_rate=0.1, reg_lambda=0.01, 
                 n_epochs=30, task='reg', metric='rmse', random_seed=42):
        """
        Initialize FM model with hyperparameters.
        
        Args:
            n_factors: Number of latent factors (default: 50)
            learning_rate: Not used (kept for compatibility, myFM uses Gibbs sampling)
            reg_lambda: Not directly used (kept for compatibility, myFM has internal regularization)
            n_epochs: Number of training iterations (default: 30)
            task: Task type - 'reg' for regression (default: 'reg')
            metric: Evaluation metric - 'rmse' or 'mae' (default: 'rmse', not used in training)
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate  # Kept for compatibility
        self.reg_lambda = reg_lambda  # Kept for compatibility
        self.n_epochs = n_epochs
        self.task = task
        self.metric = metric
        self.random_seed = random_seed
        self.model = None
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Train FM on sparse matrix data.
        
        Args:
            X_train: scipy.sparse matrix of shape (n_samples, n_features)
            y_train: numpy array of ratings/targets
            X_valid: Optional validation sparse matrix (default: None)
            y_valid: Optional validation targets (default: None)
        """
        try:
            import myfm
        except ImportError:
            raise ImportError(
                "myFM is not installed. Install with: pip install myfm"
            )
        except Exception as e:
            raise ImportError(
                f"myFM library cannot be loaded: {str(e)}\n"
                "Install with: pip install myfm"
            ) from e
        
        try:
            # Create model - use MyFMRegressor for regression
            if self.task == 'reg':
                self.model = myfm.MyFMRegressor(
                    rank=self.n_factors,
                    random_seed=self.random_seed
                )
            else:
                raise ValueError(f"Task type '{self.task}' not supported. Use 'reg' for regression.")
            
            # Train model
            # n_kept_samples controls how many samples to keep for prediction
            # Using same value as n_iter for simplicity
            self.model.fit(
                X_train, 
                y_train,
                n_iter=self.n_epochs,
                n_kept_samples=self.n_epochs
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to train myFM model: {str(e)}"
            ) from e
        
    def predict(self, X_test):
        """
        Generate predictions for test data.
        
        Args:
            X_test: scipy.sparse matrix of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        try:
            predictions = self.model.predict(X_test)
            return np.array(predictions)
        except Exception as e:
            raise RuntimeError(f"Failed to generate predictions: {str(e)}") from e


class FMRecommender:
    """
    High-level recommender interface using FM.
    
    Integrates with existing pipeline (trainset, testset format).
    """
    
    def __init__(self, feature_preprocessor, fm_model):
        """
        Initialize with preprocessor and FM model.
        
        Args:
            feature_preprocessor: FeaturePreprocessor instance
            fm_model: FactorizationMachineModel instance
        """
        self.preprocessor = feature_preprocessor
        self.fm_model = fm_model
        self.user_features_dict = None
        self.item_features_dict = None
        
    def fit(self, trainset, user_features, item_features):
        """
        Fit FM on training data with features.
        
        Args:
            trainset: Surprise Trainset
            user_features: DataFrame with user features
            item_features: DataFrame with item features
        """
        # Transform features
        self.user_features_dict = self.preprocessor.transform_user_features(user_features)
        self.item_features_dict = self.preprocessor.transform_item_features(item_features)
        
        # Convert to sparse matrix format
        X_train, y_train = self.preprocessor.to_sparse_matrix(
            trainset, 
            self.user_features_dict, 
            self.item_features_dict
        )
        
        # Train model
        self.fm_model.train(X_train, y_train)
        
    def predict(self, user_id, item_id):
        """
        Predict rating for a single user-item pair.
        
        Args:
            user_id: User ID (string)
            item_id: Item ID (string)
            
        Returns:
            float: Predicted rating
        """
        if self.user_features_dict is None or self.item_features_dict is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get user and item features
        user_feat = self.user_features_dict.get(str(user_id), {})
        item_feat = self.item_features_dict.get(str(item_id), {})
        
        # Build single-row sparse matrix
        rows = []
        cols = []
        data = []
        
        # User ID (one-hot)
        if str(user_id) in self.preprocessor.user_id_map:
            user_idx = self.preprocessor.feature_offset['user_id'] + \
                      self.preprocessor.user_id_map[str(user_id)]
            rows.append(0)
            cols.append(user_idx)
            data.append(1.0)
        
        # Item ID (one-hot)
        if str(item_id) in self.preprocessor.item_id_map:
            item_idx = self.preprocessor.feature_offset['item_id'] + \
                      self.preprocessor.item_id_map[str(item_id)]
            rows.append(0)
            cols.append(item_idx)
            data.append(1.0)
        
        # User features
        for feat_idx, feat_val in user_feat.items():
            rows.append(0)
            cols.append(feat_idx)
            data.append(feat_val)
        
        # Item features
        for feat_idx, feat_val in item_feat.items():
            rows.append(0)
            cols.append(feat_idx)
            data.append(feat_val)
        
        # Build sparse matrix
        X_single = sp.csr_matrix(
            (data, (rows, cols)), 
            shape=(1, self.preprocessor.total_features)
        )
        
        # Predict
        predictions = self.fm_model.predict(X_single)
        
        return float(predictions[0])
    
    def test(self, testset):
        """
        Test on testset (returns Surprise-compatible predictions).
        
        Args:
            testset: List of (uid, iid, true_rating) tuples
            
        Returns:
            list: List of Prediction objects compatible with Surprise
        """
        if self.user_features_dict is None or self.item_features_dict is None:
            raise ValueError("Model must be fitted before testing")
        
        # Convert testset to sparse matrix format
        X_test, _ = self.preprocessor.to_sparse_matrix(
            testset,
            self.user_features_dict,
            self.item_features_dict
        )
        
        # Get predictions
        predictions = self.fm_model.predict(X_test)
        
        # Convert to Surprise Prediction format
        surprise_predictions = []
        for i, (uid, iid, true_r) in enumerate(testset):
            pred_r = predictions[i]
            # Clip to valid rating range [1, 5]
            pred_r = max(1.0, min(5.0, pred_r))
            surprise_predictions.append(
                Prediction(uid, iid, true_r, pred_r, {})
            )
        
        return surprise_predictions


def train_fm_model(trainset, user_features, item_features, 
                   n_factors=50, learning_rate=0.1, reg_lambda=0.01, 
                   n_epochs=30, verbose=True):
    """
    Convenience function to train FM model (similar to other train_*_model functions).
    
    Args:
        trainset: Surprise Trainset
        user_features: DataFrame with user features
        item_features: DataFrame with item features
        n_factors: Number of latent factors (default: 50)
        learning_rate: Learning rate (default: 0.1)
        reg_lambda: Regularization parameter (default: 0.01)
        n_epochs: Number of epochs (default: 30)
        verbose: Print progress (default: True)
        
    Returns:
        FMRecommender: Trained FM recommender
    """
    from common.data_loader import FeaturePreprocessor
    
    if verbose:
        print("Preprocessing features...")
    
    # Preprocess features
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(user_features, item_features)
    
    if verbose:
        print(f"Total features: {preprocessor.total_features}")
        print("Training Factorization Machine...")
    
    # Create and train model
    fm_model = FactorizationMachineModel(
        n_factors=n_factors,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        n_epochs=n_epochs
    )
    
    recommender = FMRecommender(preprocessor, fm_model)
    recommender.fit(trainset, user_features, item_features)
    
    if verbose:
        print("FM training completed!")
    
    return recommender


if __name__ == "__main__":
    # Example usage
    from common.data_loader import (load_movielens_100k, get_train_test_split,
                             load_user_features, load_item_features)
    
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    
    print("Loading features...")
    user_features = load_user_features()
    item_features = load_item_features()
    
    print("Training FM model...")
    fm_model = train_fm_model(
        trainset, user_features, item_features,
        n_factors=50, learning_rate=0.1, reg_lambda=0.01, n_epochs=30
    )
    
    print("\nTesting predictions...")
    sample_user, sample_item, true_rating = testset[0]
    pred_rating = fm_model.predict(sample_user, sample_item)
    print(f"User {sample_user}, Item {sample_item}:")
    print(f"  True rating: {true_rating:.2f}")
    print(f"  Predicted: {pred_rating:.2f}")
    print(f"  Error: {abs(true_rating - pred_rating):.2f}")
