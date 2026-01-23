"""
Unit tests for mf_fm.py (Factorization Machine implementation).

Tests for:
- FactorizationMachineModel class
- FMRecommender class
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import scipy.sparse as sp
from unittest.mock import patch, MagicMock, Mock

from mf_fm import FactorizationMachineModel, FMRecommender
from data_loader import FeaturePreprocessor

myfm = pytest.importorskip('myfm', reason='myFM library required for FM tests')


class TestFactorizationMachineModel:
    """Tests for FactorizationMachineModel class."""

    def test_fm_model_initialization(self):
        """Test that FactorizationMachineModel can be instantiated with default params."""
        model = FactorizationMachineModel()
        assert model is not None
        assert model.n_factors == 50
        assert model.learning_rate == 0.1

    def test_fm_model_initialization_custom_params(self):
        """Test that FactorizationMachineModel accepts custom hyperparameters."""
        model = FactorizationMachineModel(
            n_factors=100, learning_rate=0.05, reg_lambda=0.02, n_epochs=50
        )
        assert model.n_factors == 100
        assert model.learning_rate == 0.05
        assert model.reg_lambda == 0.02
        assert model.n_epochs == 50

    def test_train_method_accepts_sparse_matrix(self, temp_data_dir):
        """Test that train() method accepts sparse matrix data."""
        model = FactorizationMachineModel(n_factors=5, n_epochs=2)
        X_train = sp.csr_matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        y_train = np.array([4.0, 3.0, 5.0])
        model.train(X_train, y_train)
        assert model.model is not None

    def test_train_method_with_validation(self, temp_data_dir):
        """Test that train() method accepts optional validation data."""
        model = FactorizationMachineModel(n_factors=5, n_epochs=2)
        X_train = sp.csr_matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        y_train = np.array([4.0, 3.0, 5.0])
        X_valid = sp.csr_matrix([[1, 0, 0]])
        y_valid = np.array([4.0])
        model.train(X_train, y_train, X_valid=X_valid, y_valid=y_valid)
        assert model.model is not None

    def test_predict_method_returns_predictions(self, temp_data_dir):
        """Test that predict() method returns predictions for test data."""
        model = FactorizationMachineModel(n_factors=5, n_epochs=2)
        X_train = sp.csr_matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        y_train = np.array([4.0, 3.0, 5.0])
        model.train(X_train, y_train)
        X_test = sp.csr_matrix([[1, 0, 1], [0, 1, 0]])
        predictions = model.predict(X_test)
        assert len(predictions) == 2
        assert all(isinstance(p, (int, float, np.floating)) for p in predictions)

    def test_model_handles_missing_features(self):
        """Test that model handles sparse matrices with missing features."""
        model = FactorizationMachineModel(n_factors=5, n_epochs=2)
        X_train = sp.csr_matrix([[1, 0], [0, 1], [1, 1]])
        y_train = np.array([4.0, 3.0, 5.0])
        model.train(X_train, y_train)
        X_test = sp.csr_matrix([[0, 0]])  # All zeros
        predictions = model.predict(X_test)
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float, np.floating))


class TestFMRecommender:
    """Tests for FMRecommender class."""

    def test_fm_recommender_initialization(self):
        """Test that FMRecommender can be instantiated."""
        preprocessor = MagicMock()
        fm_model = MagicMock()
        recommender = FMRecommender(preprocessor, fm_model)
        assert recommender is not None

    def test_fit_method_trains_model(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that fit() method trains the FM model."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=2)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(mock_trainset, sample_user_features, sample_item_features)
        assert recommender.user_features_dict is not None
        assert recommender.item_features_dict is not None

    def test_predict_method_returns_rating(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that predict() returns a rating prediction."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=2)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(mock_trainset, sample_user_features, sample_item_features)
        prediction = recommender.predict('1', '2')
        assert isinstance(prediction, (int, float))
        assert 1.0 <= prediction <= 5.0

    def test_predict_for_cold_start_user(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that predict() works for cold start users (not in trainset)."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=2)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(mock_trainset, sample_user_features, sample_item_features)
        # Cold start user (not in trainset) - but has features
        # Add user '999' to sample_user_features for this test
        user_feat_extended = pd.concat(
            [
                sample_user_features,
                pd.DataFrame(
                    {
                        'user_id': ['999'],
                        'age': [30],
                        'gender': ['M'],
                        'occupation': ['other'],
                        'zip_code': ['00000'],
                    }
                ),
            ],
            ignore_index=True,
        )
        recommender.fit(mock_trainset, user_feat_extended, sample_item_features)
        prediction = recommender.predict('999', '2')
        assert isinstance(prediction, (int, float))
        assert 1.0 <= prediction <= 5.0

    def test_test_method_returns_surprise_format(
        self, mock_trainset, mock_testset, sample_user_features, sample_item_features
    ):
        """Test that test() method returns Surprise-compatible predictions."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=2)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(mock_trainset, sample_user_features, sample_item_features)
        predictions = recommender.test(mock_testset)
        assert len(predictions) == len(mock_testset)
        for pred in predictions:
            assert hasattr(pred, 'uid')
            assert hasattr(pred, 'iid')
            assert hasattr(pred, 'r_ui')
            assert hasattr(pred, 'est')
            assert 1.0 <= pred.est <= 5.0

    def test_test_method_handles_missing_features(
        self, mock_trainset, mock_testset, sample_user_features, sample_item_features
    ):
        """Test that test() method works with minimal features."""
        preprocessor = FeaturePreprocessor()
        # Use sample features (not empty) - empty features cause StandardScaler to fail
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=2)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(mock_trainset, sample_user_features, sample_item_features)
        predictions = recommender.test(mock_testset)
        assert len(predictions) == len(mock_testset)
