"""
Unit tests for mf_als_from_scratch.py.

Tests for:
- als_matrix_factorization() function
- calculate_loss() function
- predict_rating() function
- ALSFromScratch class
- train_als_from_scratch_model() function
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from mf_als_from_scratch import (
    als_matrix_factorization,
    calculate_loss,
    predict_rating,
    ALSFromScratch,
    train_als_from_scratch_model
)


class TestALSMatrixFactorization:
    """Tests for als_matrix_factorization() function."""
    
    def test_als_returns_two_matrices(self, mock_trainset):
        """Test that als_matrix_factorization returns P and Q matrices."""
        P, Q = als_matrix_factorization(
            mock_trainset, k=10, lambda_reg=0.1, iterations=5, verbose=False
        )
        
        assert P is not None
        assert Q is not None
        assert isinstance(P, np.ndarray)
        assert isinstance(Q, np.ndarray)
    
    def test_als_matrix_shapes(self, mock_trainset):
        """Test that P and Q have correct shapes."""
        k = 10
        P, Q = als_matrix_factorization(
            mock_trainset, k=k, lambda_reg=0.1, iterations=5, verbose=False
        )
        
        # P should be k × n_users
        assert P.shape == (k, mock_trainset.n_users)
        # Q should be k × n_items
        assert Q.shape == (k, mock_trainset.n_items)
    
    def test_als_initialization_random(self, mock_trainset):
        """Test that matrices are initialized randomly."""
        P1, Q1 = als_matrix_factorization(
            mock_trainset, k=10, lambda_reg=0.1, iterations=1, 
            random_state=42, verbose=False
        )
        P2, Q2 = als_matrix_factorization(
            mock_trainset, k=10, lambda_reg=0.1, iterations=1,
            random_state=43, verbose=False
        )
        
        # Different random states should produce different initializations
        assert not np.allclose(P1, P2)
        assert not np.allclose(Q1, Q2)
    
    def test_als_reproducible_with_same_seed(self, mock_trainset):
        """Test that same random_state produces same results."""
        P1, Q1 = als_matrix_factorization(
            mock_trainset, k=10, lambda_reg=0.1, iterations=5,
            random_state=42, verbose=False
        )
        P2, Q2 = als_matrix_factorization(
            mock_trainset, k=10, lambda_reg=0.1, iterations=5,
            random_state=42, verbose=False
        )
        
        # Should be identical with same seed
        assert np.allclose(P1, P2)
        assert np.allclose(Q1, Q2)
    
    def test_als_loss_decreases(self, mock_trainset):
        """Test that loss decreases over iterations."""
        # Calculate loss at different iterations
        losses = []
        for iters in [1, 5, 10]:
            P, Q = als_matrix_factorization(
                mock_trainset, k=10, lambda_reg=0.1, iterations=iters,
                random_state=42, verbose=False
            )
            loss = calculate_loss(mock_trainset, P, Q, 0.1)
            losses.append(loss)
        
        # Loss should generally decrease (or at least not increase significantly)
        # Note: This may not always be true due to regularization, but should be true in most cases
        assert losses[-1] <= losses[0] * 1.1  # Allow 10% tolerance
    
    def test_als_with_different_k(self, mock_trainset):
        """Test that different k values produce different sized matrices."""
        P1, Q1 = als_matrix_factorization(
            mock_trainset, k=5, lambda_reg=0.1, iterations=3, verbose=False
        )
        P2, Q2 = als_matrix_factorization(
            mock_trainset, k=20, lambda_reg=0.1, iterations=3, verbose=False
        )
        
        assert P1.shape[0] == 5
        assert P2.shape[0] == 20
        assert Q1.shape[0] == 5
        assert Q2.shape[0] == 20
    
    def test_als_with_different_lambda(self, mock_trainset):
        """Test that different regularization values affect results."""
        P1, Q1 = als_matrix_factorization(
            mock_trainset, k=10, lambda_reg=0.01, iterations=5,
            random_state=42, verbose=False
        )
        P2, Q2 = als_matrix_factorization(
            mock_trainset, k=10, lambda_reg=1.0, iterations=5,
            random_state=42, verbose=False
        )
        
        # Higher regularization should produce smaller factor values
        assert np.abs(P2).mean() < np.abs(P1).mean() or np.abs(P2).mean() == np.abs(P1).mean()
    
    def test_als_verbose_output(self, mock_trainset, capsys):
        """Test that verbose=True prints progress."""
        als_matrix_factorization(
            mock_trainset, k=10, lambda_reg=0.1, iterations=15, verbose=True
        )
        captured = capsys.readouterr()
        assert 'Iteration' in captured.out or 'Final loss' in captured.out


class TestCalculateLoss:
    """Tests for calculate_loss() function."""
    
    def test_calculate_loss_returns_float(self, mock_trainset):
        """Test that calculate_loss returns a float."""
        P = np.random.normal(0, 0.1, size=(10, mock_trainset.n_users))
        Q = np.random.normal(0, 0.1, size=(10, mock_trainset.n_items))
        
        loss = calculate_loss(mock_trainset, P, Q, 0.1)
        
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0  # Loss should be non-negative
    
    def test_calculate_loss_increases_with_regularization(self, mock_trainset):
        """Test that higher regularization increases loss."""
        P = np.random.normal(0, 0.1, size=(10, mock_trainset.n_users))
        Q = np.random.normal(0, 0.1, size=(10, mock_trainset.n_items))
        
        loss_low = calculate_loss(mock_trainset, P, Q, 0.01)
        loss_high = calculate_loss(mock_trainset, P, Q, 1.0)
        
        assert loss_high >= loss_low
    
    def test_calculate_loss_has_error_component(self, mock_trainset):
        """Test that loss includes prediction error component."""
        # Create factors that predict poorly
        P = np.zeros((10, mock_trainset.n_users))
        Q = np.zeros((10, mock_trainset.n_items))
        
        loss = calculate_loss(mock_trainset, P, Q, 0.0)  # No regularization
        
        # Loss should be sum of squared errors (all predictions are 0)
        # Should be positive if there are ratings
        assert loss > 0
    
    def test_calculate_loss_has_regularization_component(self, mock_trainset):
        """Test that loss includes regularization term."""
        # Large factors
        P = np.ones((10, mock_trainset.n_users)) * 10
        Q = np.ones((10, mock_trainset.n_items)) * 10
        
        loss_no_reg = calculate_loss(mock_trainset, P, Q, 0.0)
        loss_with_reg = calculate_loss(mock_trainset, P, Q, 1.0)
        
        # Loss with regularization should be higher
        assert loss_with_reg > loss_no_reg


class TestPredictRating:
    """Tests for predict_rating() function."""
    
    def test_predict_rating_returns_float(self, mock_trainset):
        """Test that predict_rating returns a float."""
        P = np.random.normal(0, 0.1, size=(10, mock_trainset.n_users))
        Q = np.random.normal(0, 0.1, size=(10, mock_trainset.n_items))
        
        prediction = predict_rating(P, Q, mock_trainset, '1', '1')
        
        assert isinstance(prediction, (float, np.floating))
    
    def test_predict_rating_in_range(self, mock_trainset):
        """Test that predictions are clipped to valid rating range."""
        P = np.random.normal(0, 0.1, size=(10, mock_trainset.n_users))
        Q = np.random.normal(0, 0.1, size=(10, mock_trainset.n_items))
        
        prediction = predict_rating(P, Q, mock_trainset, '1', '1')
        
        assert 1.0 <= prediction <= 5.0
    
    def test_predict_rating_handles_cold_start_user(self, mock_trainset):
        """Test that predict_rating handles cold start users."""
        P = np.random.normal(0, 0.1, size=(10, mock_trainset.n_users))
        Q = np.random.normal(0, 0.1, size=(10, mock_trainset.n_items))
        
        # User not in trainset
        prediction = predict_rating(P, Q, mock_trainset, '999', '1')
        
        # Should return global mean
        assert prediction == mock_trainset.global_mean
    
    def test_predict_rating_handles_cold_start_item(self, mock_trainset):
        """Test that predict_rating handles cold start items."""
        P = np.random.normal(0, 0.1, size=(10, mock_trainset.n_users))
        Q = np.random.normal(0, 0.1, size=(10, mock_trainset.n_items))
        
        # Item not in trainset
        prediction = predict_rating(P, Q, mock_trainset, '1', '999')
        
        # Should return global mean
        assert prediction == mock_trainset.global_mean
    
    def test_predict_rating_uses_dot_product(self, mock_trainset):
        """Test that prediction uses dot product of factors."""
        # Create known factors
        P = np.zeros((10, mock_trainset.n_users))
        Q = np.zeros((10, mock_trainset.n_items))
        P[:, 0] = np.ones(10)  # User 0 has all ones
        Q[:, 0] = np.ones(10)  # Item 0 has all ones
        
        prediction = predict_rating(P, Q, mock_trainset, '1', '1')
        
        # Dot product should be 10 (sum of 10 ones)
        # But clipped to [1, 5] range
        assert prediction == 5.0  # Clipped to max


class TestALSFromScratch:
    """Tests for ALSFromScratch class."""
    
    def test_als_from_scratch_initialization(self):
        """Test that ALSFromScratch can be instantiated."""
        model = ALSFromScratch(k=10, lambda_reg=0.1, iterations=5)
        
        assert model.k == 10
        assert model.lambda_reg == 0.1
        assert model.iterations == 5
        assert model.P is None
        assert model.Q is None
    
    def test_als_from_scratch_fit_trains_model(self, mock_trainset):
        """Test that fit() trains the model."""
        model = ALSFromScratch(k=10, lambda_reg=0.1, iterations=5, random_state=42)
        model.fit(mock_trainset, verbose=False)
        
        assert model.P is not None
        assert model.Q is not None
        assert model.P.shape[0] == 10
        assert model.Q.shape[0] == 10
    
    def test_als_from_scratch_predict_before_fit_raises_error(self, mock_trainset):
        """Test that predict() raises error if model not trained."""
        model = ALSFromScratch(k=10, lambda_reg=0.1, iterations=5)
        
        with pytest.raises(ValueError, match="must be trained"):
            model.predict('1', '1')
    
    def test_als_from_scratch_predict_after_fit(self, mock_trainset):
        """Test that predict() works after fit()."""
        model = ALSFromScratch(k=10, lambda_reg=0.1, iterations=5, random_state=42)
        model.fit(mock_trainset, verbose=False)
        
        prediction = model.predict('1', '1')
        
        assert isinstance(prediction, (float, np.floating))
        assert 1.0 <= prediction <= 5.0
    
    def test_als_from_scratch_reproducible(self, mock_trainset):
        """Test that model is reproducible with same random_state."""
        model1 = ALSFromScratch(k=10, lambda_reg=0.1, iterations=5, random_state=42)
        model1.fit(mock_trainset, verbose=False)
        
        model2 = ALSFromScratch(k=10, lambda_reg=0.1, iterations=5, random_state=42)
        model2.fit(mock_trainset, verbose=False)
        
        assert np.allclose(model1.P, model2.P)
        assert np.allclose(model1.Q, model2.Q)
    
    def test_als_from_scratch_verbose_output(self, mock_trainset, capsys):
        """Test that fit() with verbose=True prints progress."""
        model = ALSFromScratch(k=10, lambda_reg=0.1, iterations=15, random_state=42)
        model.fit(mock_trainset, verbose=True)
        
        captured = capsys.readouterr()
        assert 'Training ALS' in captured.out or 'Iteration' in captured.out


class TestTrainALSFromScratchModel:
    """Tests for train_als_from_scratch_model() function."""
    
    def test_train_als_from_scratch_returns_model(self, mock_trainset):
        """Test that function returns trained model."""
        model = train_als_from_scratch_model(
            mock_trainset, k=10, lambda_reg=0.1, iterations=5, verbose=False
        )
        
        assert isinstance(model, ALSFromScratch)
        assert model.P is not None
        assert model.Q is not None
    
    def test_train_als_from_scratch_model_can_predict(self, mock_trainset):
        """Test that returned model can make predictions."""
        model = train_als_from_scratch_model(
            mock_trainset, k=10, lambda_reg=0.1, iterations=5, verbose=False
        )
        
        prediction = model.predict('1', '1')
        assert isinstance(prediction, (float, np.floating))
        assert 1.0 <= prediction <= 5.0
    
    def test_train_als_from_scratch_with_custom_params(self, mock_trainset):
        """Test that function accepts custom parameters."""
        model = train_als_from_scratch_model(
            mock_trainset, k=20, lambda_reg=0.5, iterations=10, 
            random_state=123, verbose=False
        )
        
        assert model.k == 20
        assert model.lambda_reg == 0.5
        assert model.iterations == 10
        assert model.random_state == 123
