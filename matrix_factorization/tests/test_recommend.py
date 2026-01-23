"""
Unit tests for recommend.py extensions (feature-based recommendations).

Tests for:
- generate_recommendations_with_features()
- Updated handle_cold_start_user() with FM support
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, Mock, patch

from recommend import (
    generate_recommendations_with_features,
    handle_cold_start_user,
)


class TestGenerateRecommendationsWithFeatures:
    """Tests for generate_recommendations_with_features() function."""

    def test_generate_recommendations_returns_list(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that function returns a list of recommendations."""
        fm_model = MagicMock()
        fm_model.predict = MagicMock(return_value=4.0)
        recommendations = generate_recommendations_with_features(
            fm_model, '1', sample_user_features, sample_item_features, mock_trainset, n=10
        )
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10

    def test_recommendations_are_sorted_by_rating(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that recommendations are sorted by predicted rating (descending)."""
        fm_model = MagicMock()

        def mock_predict(user_id, item_id):
            ratings = {'1': 4.5, '2': 3.8, '3': 4.2, '4': 3.5, '5': 4.0}
            return ratings.get(item_id, 3.0)

        fm_model.predict = MagicMock(side_effect=mock_predict)
        recommendations = generate_recommendations_with_features(
            fm_model, '1', sample_user_features, sample_item_features, mock_trainset, n=5
        )
        ratings = [r[1] for r in recommendations]
        assert ratings == sorted(ratings, reverse=True)

    def test_recommendations_exclude_rated_items(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that exclude_rated=True filters out already-rated items."""
        fm_model = MagicMock()
        fm_model.predict = MagicMock(return_value=4.0)
        recommendations = generate_recommendations_with_features(
            fm_model,
            '1',
            sample_user_features,
            sample_item_features,
            mock_trainset,
            n=10,
            exclude_rated=True,
        )
        recommended_item_ids = [r[0] for r in recommendations]
        # User 1 (inner_uid=0) has rated items 0,1 (raw: '1', '2')
        assert '1' not in recommended_item_ids
        assert '2' not in recommended_item_ids

    def test_recommendations_include_rated_items_when_requested(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that exclude_rated=False includes all items."""
        fm_model = MagicMock()
        fm_model.predict = MagicMock(return_value=4.0)
        recommendations = generate_recommendations_with_features(
            fm_model,
            '1',
            sample_user_features,
            sample_item_features,
            mock_trainset,
            n=10,
            exclude_rated=False,
        )
        recommended_item_ids = [r[0] for r in recommendations]
        assert len(recommended_item_ids) > 0

    def test_recommendations_for_cold_start_user(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that function works for cold start users (not in trainset)."""
        fm_model = MagicMock()
        fm_model.predict = MagicMock(return_value=3.8)
        # Add user '999' to features for this test
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
        recommendations = generate_recommendations_with_features(
            fm_model, '999', user_feat_extended, sample_item_features, mock_trainset, n=10
        )
        assert len(recommendations) > 0
        assert all(isinstance(r[1], (int, float)) for r in recommendations)

    def test_recommendations_respects_n_parameter(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that function respects n parameter (number of recommendations)."""
        fm_model = MagicMock()
        fm_model.predict = MagicMock(return_value=4.0)
        for n in [5, 10, 20]:
            recommendations = generate_recommendations_with_features(
                fm_model, '1', sample_user_features, sample_item_features, mock_trainset, n=n
            )
            assert len(recommendations) <= n

    def test_recommendations_handles_missing_features(self, mock_trainset):
        """Test that function handles missing user/item features gracefully."""
        fm_model = MagicMock()
        fm_model.predict = MagicMock(return_value=3.5)
        # User not in features
        empty_user_features = pd.DataFrame(columns=['user_id', 'age', 'gender', 'occupation'])
        recommendations = generate_recommendations_with_features(
            fm_model, '999', empty_user_features, pd.DataFrame(), mock_trainset, n=10
        )
        assert isinstance(recommendations, list)


class TestHandleColdStartUserExtended:
    """Tests for updated handle_cold_start_user() with FM support."""

    def test_handle_cold_start_with_fm_model(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that function uses FM model when provided for cold start users."""
        fm_model = MagicMock()
        fm_model.predict = MagicMock(return_value=4.0)
        # Add user '999' to features
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
        recommendations = handle_cold_start_user(
            model=None,
            trainset=mock_trainset,
            user_id='999',
            n=10,
            verbose=False,
            fm_model=fm_model,
            user_features=user_feat_extended,
            item_features=sample_item_features,
        )
        assert len(recommendations) > 0
        assert fm_model.predict.called

    def test_handle_cold_start_falls_back_to_popularity(self, mock_trainset):
        """Test that function falls back to popularity when FM model not available."""
        recommendations = handle_cold_start_user(
            model=None,
            trainset=mock_trainset,
            user_id='999',
            n=10,
            verbose=False,
            fm_model=None,
            user_features=None,
            item_features=None,
        )
        assert len(recommendations) > 0
        ratings = [r[1] for r in recommendations]
        assert ratings == sorted(ratings, reverse=True)

    def test_handle_cold_start_priority_fm_first(
        self, mock_trainset, sample_user_features, sample_item_features
    ):
        """Test that FM model takes priority over popularity fallback."""
        fm_model = MagicMock()
        fm_model.predict = MagicMock(return_value=4.2)
        # Add user '999' to features
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
        # Patch generate_recommendations_with_features to verify it's called
        with patch('recommend.generate_recommendations_with_features') as mock_gen:
            mock_gen.return_value = [('1', 4.2), ('2', 4.0), ('3', 3.8)]
            recommendations = handle_cold_start_user(
                model=None,
                trainset=mock_trainset,
                user_id='999',
                n=10,
                verbose=False,
                fm_model=fm_model,
                user_features=user_feat_extended,
                item_features=sample_item_features,
            )
            # Should call generate_recommendations_with_features (FM path) not popularity
            assert mock_gen.called
            assert len(recommendations) > 0

    def test_handle_cold_start_warm_user_uses_normal_model(self, mock_trainset):
        """Test that warm users (in trainset) use normal recommendation path."""
        regular_model = MagicMock()
        regular_model.predict = MagicMock(return_value=4.0)
        recommendations = handle_cold_start_user(
            model=regular_model,
            trainset=mock_trainset,
            user_id='1',
            n=10,
            verbose=False,
            fm_model=None,
        )
        assert regular_model.predict.called

    def test_handle_cold_start_verbose_output(self, mock_trainset, capsys):
        """Test that verbose=True prints appropriate messages."""
        recommendations = handle_cold_start_user(
            model=None,
            trainset=mock_trainset,
            user_id='999',
            n=10,
            verbose=True,
            fm_model=None,
        )
        captured = capsys.readouterr()
        assert 'cold start' in captured.out.lower()

    def test_handle_cold_start_with_partial_features(
        self, mock_trainset, sample_user_features
    ):
        """Test that function handles partial feature availability."""
        fm_model = MagicMock()
        fm_model.predict = MagicMock(return_value=3.8)
        # User features available, but no item features
        recommendations = handle_cold_start_user(
            model=None,
            trainset=mock_trainset,
            user_id='999',
            n=10,
            verbose=False,
            fm_model=fm_model,
            user_features=sample_user_features,
            item_features=None,
        )
        # Should fall back to popularity when FM fails due to missing features
        assert isinstance(recommendations, list)
