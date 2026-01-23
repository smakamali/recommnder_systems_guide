"""
Unit tests for mf_als.py (ALS using implicit library).

Tests for:
- ALSMatrixFactorization class
- train_als_model() function
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from scipy.sparse import csr_matrix

from mf_als import ALSMatrixFactorization, train_als_model


class TestALSMatrixFactorization:
    """Tests for ALSMatrixFactorization class."""
    
    def test_als_matrix_factorization_initialization(self):
        """Test that ALSMatrixFactorization can be instantiated."""
        model = ALSMatrixFactorization(
            n_factors=50, reg=0.1, n_iter=50, random_state=42
        )
        
        assert model.n_factors == 50
        assert model.reg == 0.1
        assert model.n_iter == 50
        assert model.random_state == 42
        assert model.model is None
        assert model.trainset is None
    
    def test_als_fit_trains_model(self, mock_trainset):
        """Test that fit() trains the model."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        assert model.model is not None
        assert model.trainset is not None
        assert hasattr(model.model, 'user_factors')
        assert hasattr(model.model, 'item_factors')
    
    def test_als_fit_creates_mappings(self, mock_trainset):
        """Test that fit() creates user and item mappings."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        assert model.user_mapping is not None
        assert model.item_mapping is not None
        assert model.reverse_user_mapping is not None
        assert model.reverse_item_mapping is not None
    
    def test_als_predict_before_fit_raises_error(self):
        """Test that predict() raises error if model not trained."""
        model = ALSMatrixFactorization()
        
        with pytest.raises(ValueError, match="must be trained"):
            model.predict('1', '1')
    
    def test_als_predict_returns_float(self, mock_trainset):
        """Test that predict() returns a float."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        prediction = model.predict('1', '1')
        
        assert isinstance(prediction, (float, np.floating))
    
    def test_als_predict_in_range(self, mock_trainset):
        """Test that predictions are clipped to valid rating range."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        prediction = model.predict('1', '1')
        
        assert 1.0 <= prediction <= 5.0
    
    def test_als_predict_handles_cold_start_user(self, mock_trainset):
        """Test that predict() handles cold start users."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        # User not in trainset
        prediction = model.predict('999', '1')
        
        # Should return global mean
        assert prediction == mock_trainset.global_mean
    
    def test_als_predict_handles_cold_start_item(self, mock_trainset):
        """Test that predict() handles cold start items."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        # Item not in trainset
        prediction = model.predict('1', '999')
        
        # Should return global mean
        assert prediction == mock_trainset.global_mean
    
    def test_als_get_user_factors(self, mock_trainset):
        """Test that get_user_factors() returns user factors."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        factors = model.get_user_factors('1')
        
        assert factors is not None
        assert isinstance(factors, np.ndarray)
        assert len(factors) == 10
    
    def test_als_get_user_factors_cold_start(self, mock_trainset):
        """Test that get_user_factors() returns None for cold start users."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        factors = model.get_user_factors('999')
        
        assert factors is None
    
    def test_als_get_item_factors(self, mock_trainset):
        """Test that get_item_factors() returns item factors."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        factors = model.get_item_factors('1')
        
        assert factors is not None
        assert isinstance(factors, np.ndarray)
        assert len(factors) == 10
    
    def test_als_get_item_factors_cold_start(self, mock_trainset):
        """Test that get_item_factors() returns None for cold start items."""
        model = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model.fit(mock_trainset)
        
        factors = model.get_item_factors('999')
        
        assert factors is None
    
    def test_als_get_factors_before_fit_raises_error(self):
        """Test that get_user_factors/get_item_factors raise error if not trained."""
        model = ALSMatrixFactorization()
        
        with pytest.raises(ValueError, match="must be trained"):
            model.get_user_factors('1')
        
        with pytest.raises(ValueError, match="must be trained"):
            model.get_item_factors('1')
    
    def test_als_reproducible_with_same_seed(self, mock_trainset):
        """Test that model is reproducible with same random_state."""
        model1 = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model1.fit(mock_trainset)
        
        model2 = ALSMatrixFactorization(
            n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        model2.fit(mock_trainset)
        
        # Factors should be similar (may not be identical due to implicit library internals)
        pred1 = model1.predict('1', '1')
        pred2 = model2.predict('1', '1')
        
        # Predictions should be very close
        assert abs(pred1 - pred2) < 0.01
    
    def test_als_with_different_n_factors(self, mock_trainset):
        """Test that different n_factors produce different factor dimensions."""
        model1 = ALSMatrixFactorization(
            n_factors=5, reg=0.1, n_iter=5, random_state=42
        )
        model1.fit(mock_trainset)
        
        model2 = ALSMatrixFactorization(
            n_factors=20, reg=0.1, n_iter=5, random_state=42
        )
        model2.fit(mock_trainset)
        
        factors1 = model1.get_user_factors('1')
        factors2 = model2.get_user_factors('1')
        
        assert len(factors1) == 5
        assert len(factors2) == 20


class TestTrainALSModel:
    """Tests for train_als_model() function."""
    
    def test_train_als_model_returns_model(self, mock_trainset):
        """Test that function returns trained model."""
        model = train_als_model(
            mock_trainset, n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        
        assert isinstance(model, ALSMatrixFactorization)
        assert model.model is not None
    
    def test_train_als_model_can_predict(self, mock_trainset):
        """Test that returned model can make predictions."""
        model = train_als_model(
            mock_trainset, n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        
        prediction = model.predict('1', '1')
        assert isinstance(prediction, (float, np.floating))
        assert 1.0 <= prediction <= 5.0
    
    def test_train_als_model_with_custom_params(self, mock_trainset):
        """Test that function accepts custom parameters."""
        model = train_als_model(
            mock_trainset, n_factors=20, reg=0.5, n_iter=10, random_state=123
        )
        
        assert model.n_factors == 20
        assert model.reg == 0.5
        assert model.n_iter == 10
        assert model.random_state == 123
    
    def test_train_als_model_verbose_output(self, mock_trainset, capsys):
        """Test that function prints training progress."""
        model = train_als_model(
            mock_trainset, n_factors=10, reg=0.1, n_iter=5, random_state=42
        )
        
        captured = capsys.readouterr()
        assert 'Training ALS' in captured.out or 'Training with' in captured.out
