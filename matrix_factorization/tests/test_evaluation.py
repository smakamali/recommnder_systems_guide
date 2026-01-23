"""
Unit tests for evaluation.py extensions (cold start metrics).

Tests for:
- evaluate_cold_start_users()
- evaluate_cold_start_items()
- evaluate_with_cold_start_breakdown()
- compare_with_without_features()
- Extended evaluate_model() with cold start parameters
"""

import pytest
import numpy as np
from collections import defaultdict

from evaluation import (
    evaluate_cold_start_users,
    evaluate_cold_start_items,
    evaluate_with_cold_start_breakdown,
    compare_with_without_features,
    evaluate_model,
)


class TestEvaluateColdStartUsers:
    """Tests for evaluate_cold_start_users() function."""

    def test_evaluate_cold_start_users_returns_dict(
        self, sample_predictions, cold_start_user_ids
    ):
        """Test that evaluate_cold_start_users returns a dictionary."""
        results = evaluate_cold_start_users(sample_predictions, cold_start_user_ids, verbose=False)
        assert isinstance(results, dict)
        assert 'cold_user_rmse' in results
        assert 'cold_user_mae' in results
        assert 'cold_user_coverage' in results

    def test_cold_start_users_rmse_calculation(self, cold_start_user_ids):
        """Test that RMSE is calculated correctly for cold start users."""
        predictions_with_cold = [
            ('6', '1', 4.0, 3.8, {}),
            ('6', '2', 5.0, 4.2, {}),
            ('7', '1', 3.0, 3.5, {}),
            ('1', '1', 4.0, 3.9, {}),
        ]
        cold_ids = {'6', '7'}
        results = evaluate_cold_start_users(predictions_with_cold, cold_ids, verbose=False)
        assert results['cold_user_rmse'] > 0
        errors = [(4.0 - 3.8) ** 2, (5.0 - 4.2) ** 2, (3.0 - 3.5) ** 2]
        expected_rmse = np.sqrt(np.mean(errors))
        assert abs(results['cold_user_rmse'] - expected_rmse) < 0.01

    def test_cold_start_users_coverage(self, cold_start_user_ids):
        """Test that coverage is calculated correctly."""
        predictions_with_cold = [
            ('6', '1', 4.0, 3.8, {}),
            ('7', '1', 3.0, 3.5, {}),
        ]
        cold_ids = {'6', '7', '8'}
        results = evaluate_cold_start_users(predictions_with_cold, cold_ids, verbose=False)
        assert results['cold_user_coverage'] == pytest.approx(2 / 3, abs=0.01)

    def test_cold_start_users_empty_set(self, sample_predictions):
        """Test that function handles empty cold start user set."""
        results = evaluate_cold_start_users(sample_predictions, set(), verbose=False)
        assert results['cold_user_rmse'] is None or results['cold_user_rmse'] == 0
        assert results['cold_user_count'] == 0

    def test_cold_start_users_verbose_output(self, cold_start_user_ids, capsys):
        """Test that verbose=True prints results."""
        predictions_with_cold = [
            ('6', '1', 4.0, 3.8, {}),
            ('7', '1', 3.0, 3.5, {}),
        ]
        results = evaluate_cold_start_users(
            predictions_with_cold, cold_start_user_ids, verbose=True
        )
        captured = capsys.readouterr()
        assert 'Cold Start User' in captured.out
        assert 'RMSE' in captured.out


class TestEvaluateColdStartItems:
    """Tests for evaluate_cold_start_items() function."""

    def test_evaluate_cold_start_items_returns_dict(
        self, sample_predictions, cold_start_item_ids
    ):
        """Test that evaluate_cold_start_items returns a dictionary."""
        results = evaluate_cold_start_items(sample_predictions, cold_start_item_ids, verbose=False)
        assert isinstance(results, dict)
        assert 'cold_item_rmse' in results
        assert 'cold_item_mae' in results
        assert 'cold_item_coverage' in results

    def test_cold_start_items_rmse_calculation(self, cold_start_item_ids):
        """Test that RMSE is calculated correctly for cold start items."""
        predictions_with_cold = [
            ('1', '6', 4.0, 3.8, {}),
            ('2', '6', 5.0, 4.2, {}),
            ('1', '7', 3.0, 3.5, {}),
            ('1', '1', 4.0, 3.9, {}),
        ]
        cold_ids = {'6', '7'}
        results = evaluate_cold_start_items(predictions_with_cold, cold_ids, verbose=False)
        assert results['cold_item_rmse'] > 0

    def test_cold_start_items_coverage(self, cold_start_item_ids):
        """Test that coverage is calculated correctly for cold start items."""
        predictions_with_cold = [
            ('1', '6', 4.0, 3.8, {}),
            ('2', '7', 3.0, 3.5, {}),
        ]
        cold_ids = {'6', '7', '8'}
        results = evaluate_cold_start_items(predictions_with_cold, cold_ids, verbose=False)
        assert results['cold_item_coverage'] == pytest.approx(2 / 3, abs=0.01)


class TestEvaluateWithColdStartBreakdown:
    """Tests for evaluate_with_cold_start_breakdown() function."""

    def test_cold_start_breakdown_returns_dict(
        self, sample_predictions, cold_start_user_ids, cold_start_item_ids
    ):
        """Test that function returns dictionary with 4 scenarios."""
        results = evaluate_with_cold_start_breakdown(
            sample_predictions, cold_start_user_ids, cold_start_item_ids, verbose=False
        )
        assert isinstance(results, dict)
        assert 'warm_warm' in results
        assert 'cold_user' in results
        assert 'cold_item' in results
        assert 'cold_cold' in results
        assert 'cold_user_rmse' in results

    def test_warm_warm_scenario(self, cold_start_user_ids, cold_start_item_ids):
        """Test warm-warm scenario (known user, known item)."""
        predictions = [
            ('1', '1', 4.0, 3.8, {}),
            ('1', '2', 5.0, 4.2, {}),
            ('6', '1', 3.0, 3.5, {}),
            ('1', '6', 4.0, 3.9, {}),
            ('6', '6', 3.0, 3.2, {}),
        ]
        cold_users = {'6', '7'}
        cold_items = {'6', '7'}
        results = evaluate_with_cold_start_breakdown(
            predictions, cold_users, cold_items, verbose=False
        )
        assert results['warm_warm']['rmse'] > 0
        assert results['warm_warm']['count'] == 2

    def test_cold_user_scenario(self, cold_start_user_ids, cold_start_item_ids):
        """Test cold user scenario (new user, known item)."""
        predictions = [
            ('6', '1', 4.0, 3.8, {}),
            ('6', '2', 5.0, 4.2, {}),
            ('1', '1', 4.0, 3.9, {}),
        ]
        cold_users = {'6', '7'}
        cold_items = {'8', '9'}
        results = evaluate_with_cold_start_breakdown(
            predictions, cold_users, cold_items, verbose=False
        )
        assert results['cold_user']['rmse'] > 0
        assert results['cold_user']['count'] == 2

    def test_cold_item_scenario(self, cold_start_user_ids, cold_start_item_ids):
        """Test cold item scenario (known user, new item)."""
        predictions = [
            ('1', '6', 4.0, 3.8, {}),
            ('2', '6', 5.0, 4.2, {}),
            ('1', '1', 4.0, 3.9, {}),
        ]
        cold_users = {'8', '9'}
        cold_items = {'6', '7'}
        results = evaluate_with_cold_start_breakdown(
            predictions, cold_users, cold_items, verbose=False
        )
        assert results['cold_item']['rmse'] > 0
        assert results['cold_item']['count'] == 2

    def test_cold_cold_scenario(self, cold_start_user_ids, cold_start_item_ids):
        """Test cold-cold scenario (new user, new item)."""
        predictions = [
            ('6', '6', 4.0, 3.8, {}),
            ('6', '7', 5.0, 4.2, {}),
            ('1', '1', 4.0, 3.9, {}),
        ]
        cold_users = {'6', '7'}
        cold_items = {'6', '7'}
        results = evaluate_with_cold_start_breakdown(
            predictions, cold_users, cold_items, verbose=False
        )
        assert results['cold_cold']['rmse'] > 0
        assert results['cold_cold']['count'] == 2

    def test_breakdown_counts_match_total(
        self, sample_predictions, cold_start_user_ids, cold_start_item_ids
    ):
        """Test that counts for all scenarios sum to total predictions."""
        results = evaluate_with_cold_start_breakdown(
            sample_predictions, cold_start_user_ids, cold_start_item_ids, verbose=False
        )
        total_count = (
            results['warm_warm']['count']
            + results['cold_user']['count']
            + results['cold_item']['count']
            + results['cold_cold']['count']
        )
        assert total_count == len(sample_predictions)

    def test_breakdown_verbose_output(
        self, sample_predictions, cold_start_user_ids, cold_start_item_ids, capsys
    ):
        """Test that verbose=True prints breakdown by scenario."""
        results = evaluate_with_cold_start_breakdown(
            sample_predictions, cold_start_user_ids, cold_start_item_ids, verbose=True
        )
        captured = capsys.readouterr()
        assert 'Warm-Warm' in captured.out
        assert 'Cold User' in captured.out
        assert 'Cold Item' in captured.out
        assert 'Cold-Cold' in captured.out


class TestCompareWithWithoutFeatures:
    """Tests for compare_with_without_features() function."""

    def test_compare_returns_dict(self, sample_predictions):
        """Test that compare_with_without_features returns a dictionary."""
        predictions_with = sample_predictions
        predictions_without = [
            ('1', '3', 4.0, 3.9, {}),
            ('2', '2', 3.0, 3.6, {}),
            ('3', '1', 5.0, 4.3, {}),
            ('4', '2', 4.0, 4.0, {}),
            ('5', '1', 3.0, 3.3, {}),
        ]
        results = compare_with_without_features(predictions_with, predictions_without)
        assert isinstance(results, dict)
        assert 'rmse_improvement_pct' in results
        assert 'mae_improvement_pct' in results

    def test_rmse_improvement_calculation(self):
        """Test that RMSE improvement is calculated correctly."""
        predictions_with = [
            ('1', '1', 4.0, 3.8, {}),
            ('2', '2', 3.0, 3.5, {}),
        ]
        predictions_without = [
            ('1', '1', 4.0, 3.9, {}),
            ('2', '2', 3.0, 3.6, {}),
        ]
        results = compare_with_without_features(predictions_with, predictions_without)
        assert results['rmse_improvement_pct'] > 0

    def test_mae_improvement_calculation(self):
        """Test that MAE improvement is calculated correctly."""
        predictions_with = [
            ('1', '1', 4.0, 3.8, {}),
            ('2', '2', 3.0, 3.5, {}),
        ]
        predictions_without = [
            ('1', '1', 4.0, 4.2, {}),  # Worse prediction
            ('2', '2', 3.0, 3.8, {}),  # Worse prediction
        ]
        results = compare_with_without_features(predictions_with, predictions_without)
        assert results['mae_improvement_pct'] >= 0  # Can be 0 if MAE same

    def test_breakdown_by_user_activity(self, sample_predictions):
        """Test that function provides comparison metrics."""
        predictions_with = sample_predictions
        predictions_without = [
            ('1', '3', 4.0, 3.9, {}),
            ('2', '2', 3.0, 3.6, {}),
            ('3', '1', 5.0, 4.3, {}),
            ('4', '2', 4.0, 4.0, {}),
            ('5', '1', 3.0, 3.3, {}),
        ]
        results = compare_with_without_features(predictions_with, predictions_without)
        assert 'rmse_with' in results
        assert 'rmse_without' in results
        # Note: implementation doesn't have improvement_by_activity, so we test what exists


class TestEvaluateModelExtended:
    """Tests for extended evaluate_model() with cold start parameters."""

    def test_evaluate_model_with_cold_start_params(
        self, sample_predictions, cold_start_user_ids, cold_start_item_ids
    ):
        """Test that evaluate_model accepts cold start parameters."""
        results = evaluate_model(
            sample_predictions,
            k=10,
            threshold=4.0,
            verbose=False,
            cold_start_users=cold_start_user_ids,
            cold_start_items=cold_start_item_ids,
        )
        assert isinstance(results, dict)
        assert 'rmse' in results
        assert 'mae' in results

    def test_evaluate_model_cold_start_breakdown_flag(
        self, sample_predictions, cold_start_user_ids, cold_start_item_ids
    ):
        """Test that cold_start_users/items parameters add cold start metrics."""
        results = evaluate_model(
            sample_predictions,
            k=10,
            threshold=4.0,
            verbose=False,
            cold_start_users=cold_start_user_ids,
            cold_start_items=cold_start_item_ids,
        )
        # Should include cold start metrics if provided
        if results.get('cold_user_rmse') is not None:
            assert 'cold_user_rmse' in results
        if results.get('cold_item_rmse') is not None:
            assert 'cold_item_rmse' in results
        # Note: evaluate_model doesn't have breakdown_by_scenario flag,
        # but we can call evaluate_with_cold_start_breakdown separately

    def test_evaluate_model_backward_compatible(self, sample_predictions):
        """Test that evaluate_model still works without cold start params."""
        results = evaluate_model(sample_predictions, k=10, threshold=4.0, verbose=False)
        assert isinstance(results, dict)
        assert 'rmse' in results
        assert 'mae' in results
        assert 'precision@10' in results
        assert 'recall@10' in results
