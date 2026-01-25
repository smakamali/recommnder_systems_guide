"""
Integration tests for the complete Factorization Machines pipeline.

These tests verify that all components work together correctly.
Note: These tests may require actual data files and trained models.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from common.data_loader import (
    load_movielens_100k,
    get_train_test_split,
    load_user_features,
    load_item_features,
    FeaturePreprocessor,
    get_cold_start_split,
)
from mf_fm import FactorizationMachineModel, FMRecommender, train_fm_model
from common.evaluation import evaluate_model, evaluate_with_cold_start_breakdown
from recommend import generate_recommendations_with_features

myfm = pytest.importorskip('myfm', reason='myFM library required for integration tests')


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete FM pipeline."""

    def test_end_to_end_pipeline(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Test complete pipeline from data loading to evaluation."""
        data = sample_surprise_dataset
        trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=3)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(trainset, sample_user_features, sample_item_features)
        predictions = recommender.test(testset)
        results = evaluate_model(predictions, k=5, threshold=4.0, verbose=False)
        assert results['rmse'] > 0
        assert results['mae'] > 0

    def test_cold_start_evaluation_pipeline(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Test pipeline with cold start evaluation."""
        data = sample_surprise_dataset
        trainset, testset, cold_users, cold_items = get_cold_start_split(
            data, sample_user_features, sample_item_features, random_state=42
        )
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=3)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(trainset, sample_user_features, sample_item_features)
        predictions = recommender.test(testset)
        cold_results = evaluate_with_cold_start_breakdown(
            predictions, cold_users, cold_items, verbose=False
        )
        assert 'warm_warm' in cold_results
        assert 'cold_user' in cold_results
        assert 'cold_item' in cold_results
        assert 'cold_cold' in cold_results

    def test_feature_based_recommendations_pipeline(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Test feature-based recommendations for cold start users."""
        data = sample_surprise_dataset
        trainset, testset, cold_users, cold_items = get_cold_start_split(
            data, sample_user_features, sample_item_features, random_state=42
        )
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=3)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(trainset, sample_user_features, sample_item_features)
        if len(cold_users) > 0:
            cold_user_id = list(cold_users)[0]
            recommendations = generate_recommendations_with_features(
                recommender, cold_user_id, sample_user_features, sample_item_features, trainset, n=10
            )
            assert len(recommendations) > 0
            assert all(isinstance(r[1], (int, float)) for r in recommendations)
            assert all(1.0 <= r[1] <= 5.0 for r in recommendations)


@pytest.mark.integration
@pytest.mark.slow
class TestModelTraining:
    """Tests for model training (marked as slow)."""

    def test_fm_training_converges(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Test that FM model training completes without errors."""
        data = sample_surprise_dataset
        trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=5)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(trainset, sample_user_features, sample_item_features)
        # Training should complete
        assert recommender.user_features_dict is not None

    def test_fm_model_save_load_consistency(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Test that model produces consistent predictions (no save/load in current API)."""
        data = sample_surprise_dataset
        trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        fm_model = FactorizationMachineModel(n_factors=5, n_epochs=3, random_seed=42)
        recommender = FMRecommender(preprocessor, fm_model)
        recommender.fit(trainset, sample_user_features, sample_item_features)
        # Note: Current implementation doesn't have save/load, so we just test consistency
        pred1 = recommender.predict('1', '1')
        pred2 = recommender.predict('1', '1')
        assert abs(pred1 - pred2) < 0.01


@pytest.mark.integration
@pytest.mark.requires_data
class TestWithRealData:
    """Tests that require actual MovieLens 100k data files."""

    def test_load_real_user_features(self, temp_data_dir):
        """Test loading real user features from MovieLens 100k."""
        try:
            user_features = load_user_features()
            assert len(user_features) > 0
            assert 'user_id' in user_features.columns
        except (FileNotFoundError, RuntimeError) as e:
            pytest.skip(f"MovieLens data not available: {e}")

    def test_load_real_item_features(self, temp_data_dir):
        """Test loading real item features from MovieLens 100k."""
        try:
            item_features = load_item_features()
            assert len(item_features) > 0
            assert 'item_id' in item_features.columns
        except (FileNotFoundError, RuntimeError) as e:
            pytest.skip(f"MovieLens data not available: {e}")


@pytest.mark.integration
@pytest.mark.requires_myfm
class TestWithMyFM:
    """Tests that require myFM library."""

    def test_myfm_installation(self):
        """Test that myFM is properly installed."""
        try:
            import myfm
            assert True
        except ImportError:
            pytest.skip("myFM not installed")

    def test_myfm_basic_functionality(self):
        """Test basic myFM functionality."""
        import myfm
        import scipy.sparse as sp
        X = sp.csr_matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        y = np.array([4.0, 3.0, 5.0])
        model = myfm.MyFMRegressor(rank=5, random_seed=42)
        model.fit(X, y, n_iter=2)
        predictions = model.predict(X)
        assert len(predictions) == 3
        assert all(isinstance(p, (int, float, np.floating)) for p in predictions)
