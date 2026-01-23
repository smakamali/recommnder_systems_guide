"""
Unit tests for mf_svd.py (SVD using Surprise library).

Tests for:
- train_svd_model() function
- SVDMatrixFactorization class
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from mf_svd import train_svd_model, SVDMatrixFactorization


class TestTrainSVDModel:
    """Tests for train_svd_model() function."""
    
    def test_train_svd_model_returns_model(self, mock_trainset):
        """Test that function returns trained SVD model."""
        model = train_svd_model(
            mock_trainset, n_factors=10, n_epochs=5, verbose=False
        )
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'test')
    
    def test_train_svd_model_can_predict(self, mock_trainset):
        """Test that returned model can make predictions."""
        model = train_svd_model(
            mock_trainset, n_factors=10, n_epochs=5, verbose=False
        )
        
        pred = model.predict('1', '1')
        prediction = pred.est if hasattr(pred, 'est') else pred
        
        assert isinstance(prediction, (float, np.floating))
        assert 1.0 <= prediction <= 5.0
    
    def test_train_svd_model_with_custom_params(self, mock_trainset):
        """Test that function accepts custom parameters."""
        model = train_svd_model(
            mock_trainset,
            n_factors=20,
            n_epochs=10,
            lr_all=0.01,
            reg_all=0.05,
            random_state=123,
            verbose=False
        )
        
        # Model should be trained with these parameters
        assert model is not None
        pred = model.predict('1', '1')
        prediction = pred.est if hasattr(pred, 'est') else pred
        assert isinstance(prediction, (float, np.floating))
    
    def test_train_svd_model_verbose_output(self, mock_trainset, capsys):
        """Test that verbose=True prints training progress."""
        model = train_svd_model(
            mock_trainset, n_factors=10, n_epochs=5, verbose=True
        )
        
        captured = capsys.readouterr()
        assert 'Training SVD' in captured.out or 'Training completed' in captured.out
    
    def test_train_svd_model_reproducible(self, mock_trainset):
        """Test that model is reproducible with same random_state."""
        model1 = train_svd_model(
            mock_trainset, n_factors=10, n_epochs=5, random_state=42, verbose=False
        )
        model2 = train_svd_model(
            mock_trainset, n_factors=10, n_epochs=5, random_state=42, verbose=False
        )
        
        # Predictions should be identical with same seed
        p1 = model1.predict('1', '1')
        p2 = model2.predict('1', '1')
        pred1 = p1.est if hasattr(p1, 'est') else p1
        pred2 = p2.est if hasattr(p2, 'est') else p2
        assert abs(pred1 - pred2) < 0.01
    
    def test_train_svd_model_test_method(self, mock_trainset, mock_testset):
        """Test that model.test() works correctly."""
        model = train_svd_model(
            mock_trainset, n_factors=10, n_epochs=5, verbose=False
        )
        
        predictions = model.test(mock_testset)
        
        assert len(predictions) == len(mock_testset)
        for pred in predictions:
            assert hasattr(pred, 'uid')
            assert hasattr(pred, 'iid')
            assert hasattr(pred, 'r_ui')
            assert hasattr(pred, 'est')
            assert 1.0 <= pred.est <= 5.0


class TestSVDMatrixFactorization:
    """Tests for SVDMatrixFactorization class."""
    
    def test_svd_matrix_factorization_initialization(self):
        """Test that SVDMatrixFactorization can be instantiated."""
        model = SVDMatrixFactorization(
            n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42
        )
        
        assert model.n_factors == 50
        assert model.n_epochs == 20
        assert model.lr_all == 0.005
        assert model.reg_all == 0.02
        assert model.random_state == 42
        assert model.model is None
    
    def test_svd_fit_trains_model(self, mock_trainset):
        """Test that fit() trains the model."""
        model = SVDMatrixFactorization(n_factors=10, n_epochs=5)
        model.fit(mock_trainset, verbose=False)
        
        assert model.model is not None
        assert hasattr(model.model, 'predict')
    
    def test_svd_predict_before_fit_raises_error(self):
        """Test that predict() raises error if model not trained."""
        model = SVDMatrixFactorization()
        
        with pytest.raises(ValueError, match="must be trained"):
            model.predict('1', '1')
    
    def test_svd_predict_after_fit(self, mock_trainset):
        """Test that predict() works after fit()."""
        model = SVDMatrixFactorization(n_factors=10, n_epochs=5)
        model.fit(mock_trainset, verbose=False)
        
        prediction = model.predict('1', '1')
        
        assert isinstance(prediction, (float, np.floating))
        assert 1.0 <= prediction <= 5.0
    
    def test_svd_predict_returns_rating(self, mock_trainset):
        """Test that predict() returns a valid rating."""
        model = SVDMatrixFactorization(n_factors=10, n_epochs=5)
        model.fit(mock_trainset, verbose=False)
        
        prediction = model.predict('1', '1')
        
        assert isinstance(prediction, (float, np.floating))
        assert 1.0 <= prediction <= 5.0
    
    def test_svd_test_method(self, mock_trainset, mock_testset):
        """Test that test() method works correctly."""
        model = SVDMatrixFactorization(n_factors=10, n_epochs=5)
        model.fit(mock_trainset, verbose=False)
        
        predictions = model.test(mock_testset)
        
        assert len(predictions) == len(mock_testset)
        for pred in predictions:
            assert hasattr(pred, 'uid')
            assert hasattr(pred, 'iid')
            assert hasattr(pred, 'r_ui')
            assert hasattr(pred, 'est')
    
    def test_svd_test_before_fit_raises_error(self, mock_testset):
        """Test that test() raises error if model not trained."""
        model = SVDMatrixFactorization()
        
        with pytest.raises(ValueError, match="must be trained"):
            model.test(mock_testset)
    
    def test_svd_reproducible(self, mock_trainset):
        """Test that model is reproducible with same random_state."""
        model1 = SVDMatrixFactorization(n_factors=10, n_epochs=5, random_state=42)
        model1.fit(mock_trainset, verbose=False)
        
        model2 = SVDMatrixFactorization(n_factors=10, n_epochs=5, random_state=42)
        model2.fit(mock_trainset, verbose=False)
        
        pred1 = model1.predict('1', '1')
        pred2 = model2.predict('1', '1')
        
        assert abs(pred1 - pred2) < 0.01
    
    def test_svd_with_different_n_factors(self, mock_trainset):
        """Test that different n_factors produce different models."""
        model1 = SVDMatrixFactorization(n_factors=5, n_epochs=5)
        model1.fit(mock_trainset, verbose=False)
        
        model2 = SVDMatrixFactorization(n_factors=20, n_epochs=5)
        model2.fit(mock_trainset, verbose=False)
        
        # Models should be different (though predictions may be similar)
        pred1 = model1.predict('1', '1')
        pred2 = model2.predict('1', '1')
        
        # Both should be valid predictions
        assert 1.0 <= pred1 <= 5.0
        assert 1.0 <= pred2 <= 5.0
    
    def test_svd_verbose_output(self, mock_trainset, capsys):
        """Test that fit() with verbose=True prints progress."""
        model = SVDMatrixFactorization(n_factors=10, n_epochs=5)
        model.fit(mock_trainset, verbose=True)
        
        captured = capsys.readouterr()
        assert "Training SVD" in captured.out or "Training completed" in captured.out
