"""
Evaluation Metrics for Recommender Systems.

This module implements evaluation metrics as described in the guide's
Section 1.3 - Evaluation Metrics (README.md lines 104-160).

Includes rating prediction metrics (RMSE, MAE) and ranking metrics
(Precision@K, Recall@K, NDCG@K, MRR, Hit Rate@K).

Reference: Guide Section 1.3 - Evaluation Metrics
"""

import numpy as np
from collections import defaultdict


def calculate_rmse(predictions):
    """
    Calculate Root Mean Square Error (RMSE).
    
    Formula (guide line 113):
    RMSE = sqrt((1/N) * sum((r_ui - r_hat_ui)^2))
    
    Lower values indicate better performance.
    
    Args:
        predictions: List of prediction tuples (uid, iid, true_r, est_r, ...)
                    or list of tuples (uid, iid, true_r, est_r)
                    
    Returns:
        float: RMSE value
    """
    errors = []
    
    for pred in predictions:
        if len(pred) >= 4:
            # Surprise format: (uid, iid, true_r, est_r, details)
            true_r = pred[2]
            est_r = pred[3]
        elif len(pred) == 3:
            # Format: (uid, iid, true_r) with separate prediction
            true_r = pred[2]
            est_r = pred[3] if hasattr(pred, '__getitem__') else None
        else:
            continue
            
        errors.append((true_r - est_r) ** 2)
    
    rmse = np.sqrt(np.mean(errors))
    return rmse


def calculate_mae(predictions):
    """
    Calculate Mean Absolute Error (MAE).
    
    Formula (guide line 119):
    MAE = (1/N) * sum(|r_ui - r_hat_ui|)
    
    Lower values indicate better performance. More robust to outliers than RMSE.
    
    Args:
        predictions: List of prediction tuples (uid, iid, true_r, est_r, ...)
                    or list of tuples (uid, iid, true_r, est_r)
                    
    Returns:
        float: MAE value
    """
    errors = []
    
    for pred in predictions:
        if len(pred) >= 4:
            true_r = pred[2]
            est_r = pred[3]
        else:
            continue
            
        errors.append(abs(true_r - est_r))
    
    mae = np.mean(errors)
    return mae


def precision_at_k(predictions, k=10, threshold=4.0):
    """
    Calculate Precision@K.
    
    Formula (guide line 129):
    Precision@K = |Relevant Items ∩ Top-K Recommendations| / K
    
    Fraction of recommended items that are relevant.
    
    Args:
        predictions: List of prediction tuples (uid, iid, true_r, est_r, ...)
        k (int): Number of top recommendations to consider (default: 10)
        threshold (float): Rating threshold for relevance (default: 4.0)
                          Items with true_r >= threshold are considered relevant
                          
    Returns:
        float: Precision@K value
    """
    # Group predictions by user
    user_predictions = defaultdict(list)
    
    for pred in predictions:
        if len(pred) >= 4:
            uid = pred[0]
            true_r = pred[2]
            est_r = pred[3]
            user_predictions[uid].append((true_r, est_r))
    
    precisions = []
    
    for uid, preds in user_predictions.items():
        # Sort by estimated rating (descending)
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        
        # Top-K recommendations
        top_k = preds_sorted[:k]
        
        # Count relevant items in top-K
        relevant_count = sum(1 for true_r, _ in top_k if true_r >= threshold)
        
        if len(top_k) > 0:
            precision = relevant_count / len(top_k)
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def recall_at_k(predictions, k=10, threshold=4.0):
    """
    Calculate Recall@K.
    
    Formula (guide line 135):
    Recall@K = |Relevant Items ∩ Top-K Recommendations| / |Relevant Items|
    
    Fraction of relevant items that are recommended.
    
    Args:
        predictions: List of prediction tuples (uid, iid, true_r, est_r, ...)
        k (int): Number of top recommendations to consider (default: 10)
        threshold (float): Rating threshold for relevance (default: 4.0)
                          
    Returns:
        float: Recall@K value
    """
    # Group predictions by user
    user_predictions = defaultdict(list)
    
    for pred in predictions:
        if len(pred) >= 4:
            uid = pred[0]
            true_r = pred[2]
            est_r = pred[3]
            user_predictions[uid].append((true_r, est_r))
    
    recalls = []
    
    for uid, preds in user_predictions.items():
        # Total relevant items for this user
        total_relevant = sum(1 for true_r, _ in preds if true_r >= threshold)
        
        if total_relevant == 0:
            continue
        
        # Sort by estimated rating (descending)
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        
        # Top-K recommendations
        top_k = preds_sorted[:k]
        
        # Count relevant items in top-K
        relevant_in_top_k = sum(1 for true_r, _ in top_k if true_r >= threshold)
        
        recall = relevant_in_top_k / total_relevant
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(predictions, k=10, threshold=4.0):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    Formula (guide lines 141, 145):
    DCG@K = sum(rel_i / log2(i+1))
    NDCG@K = DCG@K / IDCG@K
    
    Considers ranking position - higher positions contribute more.
    
    Args:
        predictions: List of prediction tuples (uid, iid, true_r, est_r, ...)
        k (int): Number of top recommendations to consider (default: 10)
        threshold (float): Rating threshold for relevance (default: 4.0)
                          
    Returns:
        float: NDCG@K value
    """
    # Group predictions by user
    user_predictions = defaultdict(list)
    
    for pred in predictions:
        if len(pred) >= 4:
            uid = pred[0]
            true_r = pred[2]
            est_r = pred[3]
            user_predictions[uid].append((true_r, est_r))
    
    ndcgs = []
    
    for uid, preds in user_predictions.items():
        # Sort by estimated rating (descending)
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        
        # Top-K recommendations
        top_k = preds_sorted[:k]
        
        # Calculate DCG@K
        dcg = 0.0
        for i, (true_r, _) in enumerate(top_k, start=1):
            rel = 1.0 if true_r >= threshold else 0.0
            dcg += rel / np.log2(i + 1)
        
        # Calculate IDCG@K (ideal DCG - sort by true rating)
        preds_sorted_by_true = sorted(preds, key=lambda x: x[0], reverse=True)
        ideal_top_k = preds_sorted_by_true[:k]
        
        idcg = 0.0
        for i, (true_r, _) in enumerate(ideal_top_k, start=1):
            rel = 1.0 if true_r >= threshold else 0.0
            idcg += rel / np.log2(i + 1)
        
        # Calculate NDCG
        if idcg > 0:
            ndcg = dcg / idcg
            ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def hit_rate_at_k(predictions, k=10, threshold=4.0):
    """
    Calculate Hit Rate@K.
    
    Formula (guide line 159):
    Hit Rate@K = |{u : |Relevant_u ∩ Top-K_u| > 0}| / |U|
    
    Percentage of users with at least one relevant item in top-K.
    
    Args:
        predictions: List of prediction tuples (uid, iid, true_r, est_r, ...)
        k (int): Number of top recommendations to consider (default: 10)
        threshold (float): Rating threshold for relevance (default: 4.0)
                          
    Returns:
        float: Hit Rate@K value
    """
    # Group predictions by user
    user_predictions = defaultdict(list)
    
    for pred in predictions:
        if len(pred) >= 4:
            uid = pred[0]
            true_r = pred[2]
            est_r = pred[3]
            user_predictions[uid].append((true_r, est_r))
    
    hits = 0
    total_users = len(user_predictions)
    
    for uid, preds in user_predictions.items():
        # Sort by estimated rating (descending)
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        
        # Top-K recommendations
        top_k = preds_sorted[:k]
        
        # Check if at least one relevant item in top-K
        has_relevant = any(true_r >= threshold for true_r, _ in top_k)
        
        if has_relevant:
            hits += 1
    
    return hits / total_users if total_users > 0 else 0.0


def evaluate_cold_start_users(predictions, cold_start_user_ids, verbose=True):
    """
    Evaluate model performance on cold start users.
    
    Args:
        predictions: List of prediction tuples (uid, iid, true_r, est_r, ...)
        cold_start_user_ids: Set of user IDs not in training
        verbose: Print detailed results (default: True)
        
    Returns:
        dict with cold_start_rmse, cold_start_mae, cold_start_coverage
    """
    cold_predictions = []
    
    for pred in predictions:
        if len(pred) >= 4:
            uid = pred[0]
            if uid in cold_start_user_ids:
                cold_predictions.append(pred)
    
    if len(cold_predictions) == 0:
        if verbose:
            print("No cold start user predictions found.")
        return {
            'cold_user_rmse': None,
            'cold_user_mae': None,
            'cold_user_coverage': 0.0,
            'cold_user_count': 0
        }
    
    rmse = calculate_rmse(cold_predictions)
    mae = calculate_mae(cold_predictions)
    
    # Coverage: percentage of cold start users that have at least one prediction
    unique_cold_users_with_predictions = set(pred[0] for pred in cold_predictions if len(pred) >= 4)
    coverage = len(unique_cold_users_with_predictions) / len(cold_start_user_ids) if cold_start_user_ids else 0.0
    
    if verbose:
        print("\n" + "=" * 60)
        print("Cold Start User Evaluation")
        print("=" * 60)
        print(f"Cold start users in test set: {len(cold_start_user_ids)}")
        print(f"Cold start users with predictions: {len(unique_cold_users_with_predictions)}")
        print(f"Total predictions: {len(cold_predictions)}")
        print(f"Coverage: {coverage:.2%}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print("=" * 60)
    
    return {
        'cold_user_rmse': rmse,
        'cold_user_mae': mae,
        'cold_user_coverage': coverage,
        'cold_user_count': len(cold_predictions)
    }


def evaluate_cold_start_items(predictions, cold_start_item_ids, verbose=True):
    """
    Evaluate model performance on cold start items.
    
    Args:
        predictions: List of prediction tuples
        cold_start_item_ids: Set of item IDs with few ratings in training
        verbose: Print detailed results (default: True)
        
    Returns:
        dict with metrics specific to new items
    """
    cold_predictions = []
    
    for pred in predictions:
        if len(pred) >= 4:
            iid = pred[1]
            if iid in cold_start_item_ids:
                cold_predictions.append(pred)
    
    if len(cold_predictions) == 0:
        if verbose:
            print("No cold start item predictions found.")
        return {
            'cold_item_rmse': None,
            'cold_item_mae': None,
            'cold_item_coverage': 0.0,
            'cold_item_count': 0
        }
    
    rmse = calculate_rmse(cold_predictions)
    mae = calculate_mae(cold_predictions)
    
    # Coverage: percentage of cold start items that have at least one prediction
    unique_cold_items_with_predictions = set(pred[1] for pred in cold_predictions if len(pred) >= 4)
    coverage = len(unique_cold_items_with_predictions) / len(cold_start_item_ids) if cold_start_item_ids else 0.0
    
    if verbose:
        print("\n" + "=" * 60)
        print("Cold Start Item Evaluation")
        print("=" * 60)
        print(f"Cold start items in test set: {len(cold_start_item_ids)}")
        print(f"Cold start items with predictions: {len(unique_cold_items_with_predictions)}")
        print(f"Total predictions: {len(cold_predictions)}")
        print(f"Coverage: {coverage:.2%}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print("=" * 60)
    
    return {
        'cold_item_rmse': rmse,
        'cold_item_mae': mae,
        'cold_item_coverage': coverage,
        'cold_item_count': len(cold_predictions)
    }


def evaluate_with_cold_start_breakdown(predictions, cold_start_users, 
                                      cold_start_items, verbose=True):
    """
    Comprehensive evaluation with 4 scenarios:
    1. Warm-warm (known user, known item)
    2. Cold user (new user, known item)
    3. Cold item (known user, new item)
    4. Cold-cold (new user, new item)
    
    Args:
        predictions: List of prediction tuples
        cold_start_users: Set of cold start user IDs
        cold_start_items: Set of cold start item IDs
        verbose: Print detailed results (default: True)
        
    Returns:
        dict with metrics for each scenario
    """
    warm_warm = []
    cold_user = []
    cold_item = []
    cold_cold = []
    
    for pred in predictions:
        if len(pred) >= 4:
            uid = pred[0]
            iid = pred[1]
            
            is_cold_user = uid in cold_start_users
            is_cold_item = iid in cold_start_items
            
            if is_cold_user and is_cold_item:
                cold_cold.append(pred)
            elif is_cold_user:
                cold_user.append(pred)
            elif is_cold_item:
                cold_item.append(pred)
            else:
                warm_warm.append(pred)
    
    results = {}
    
    # Warm-warm
    if warm_warm:
        results['warm_warm'] = {
            'rmse': calculate_rmse(warm_warm),
            'mae': calculate_mae(warm_warm),
            'count': len(warm_warm)
        }
    else:
        results['warm_warm'] = {'rmse': None, 'mae': None, 'count': 0}
    
    # Cold user
    if cold_user:
        results['cold_user'] = {
            'rmse': calculate_rmse(cold_user),
            'mae': calculate_mae(cold_user),
            'count': len(cold_user)
        }
    else:
        results['cold_user'] = {'rmse': None, 'mae': None, 'count': 0}
    
    # Cold item
    if cold_item:
        results['cold_item'] = {
            'rmse': calculate_rmse(cold_item),
            'mae': calculate_mae(cold_item),
            'count': len(cold_item)
        }
    else:
        results['cold_item'] = {'rmse': None, 'mae': None, 'count': 0}
    
    # Cold-cold
    if cold_cold:
        results['cold_cold'] = {
            'rmse': calculate_rmse(cold_cold),
            'mae': calculate_mae(cold_cold),
            'count': len(cold_cold)
        }
    else:
        results['cold_cold'] = {'rmse': None, 'mae': None, 'count': 0}
    
    # Overall cold start metrics
    results['cold_user_rmse'] = results['cold_user']['rmse']
    results['cold_item_rmse'] = results['cold_item']['rmse']
    results['cold_cold_rmse'] = results['cold_cold']['rmse']
    
    if verbose:
        print("\n" + "=" * 60)
        print("Cold Start Breakdown Evaluation")
        print("=" * 60)
        print(f"\nWarm-Warm (known user, known item):")
        print(f"  Count: {results['warm_warm']['count']}")
        if results['warm_warm']['rmse'] is not None:
            print(f"  RMSE: {results['warm_warm']['rmse']:.4f}")
            print(f"  MAE:  {results['warm_warm']['mae']:.4f}")
        
        print(f"\nCold User (new user, known item):")
        print(f"  Count: {results['cold_user']['count']}")
        if results['cold_user']['rmse'] is not None:
            print(f"  RMSE: {results['cold_user']['rmse']:.4f}")
            print(f"  MAE:  {results['cold_user']['mae']:.4f}")
        
        print(f"\nCold Item (known user, new item):")
        print(f"  Count: {results['cold_item']['count']}")
        if results['cold_item']['rmse'] is not None:
            print(f"  RMSE: {results['cold_item']['rmse']:.4f}")
            print(f"  MAE:  {results['cold_item']['mae']:.4f}")
        
        print(f"\nCold-Cold (new user, new item):")
        print(f"  Count: {results['cold_cold']['count']}")
        if results['cold_cold']['rmse'] is not None:
            print(f"  RMSE: {results['cold_cold']['rmse']:.4f}")
            print(f"  MAE:  {results['cold_cold']['mae']:.4f}")
        print("=" * 60)
    
    return results


def compare_with_without_features(predictions_with_features, 
                                  predictions_without_features):
    """
    Quantify improvement from using features.
    
    Shows RMSE/MAE improvement and breakdown by user activity level.
    
    Args:
        predictions_with_features: Predictions from model with features
        predictions_without_features: Predictions from model without features
        
    Returns:
        dict with comparison metrics
    """
    rmse_with = calculate_rmse(predictions_with_features)
    rmse_without = calculate_rmse(predictions_without_features)
    
    mae_with = calculate_mae(predictions_with_features)
    mae_without = calculate_mae(predictions_without_features)
    
    rmse_improvement = ((rmse_without - rmse_with) / rmse_without) * 100
    mae_improvement = ((mae_without - mae_with) / mae_without) * 100
    
    print("\n" + "=" * 60)
    print("Feature Impact Analysis")
    print("=" * 60)
    print(f"\nWithout Features:")
    print(f"  RMSE: {rmse_without:.4f}")
    print(f"  MAE:  {mae_without:.4f}")
    
    print(f"\nWith Features:")
    print(f"  RMSE: {rmse_with:.4f}")
    print(f"  MAE:  {mae_with:.4f}")
    
    print(f"\nImprovement:")
    print(f"  RMSE: {rmse_improvement:.2f}%")
    print(f"  MAE:  {mae_improvement:.2f}%")
    print("=" * 60)
    
    return {
        'rmse_with': rmse_with,
        'rmse_without': rmse_without,
        'mae_with': mae_with,
        'mae_without': mae_without,
        'rmse_improvement_pct': rmse_improvement,
        'mae_improvement_pct': mae_improvement
    }


def evaluate_model(predictions, k=10, threshold=4.0, verbose=True,
                   cold_start_users=None, cold_start_items=None):
    """
    Comprehensive evaluation of a model on a test set.
    
    Calculates all metrics: RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate@K.
    Optionally includes cold start evaluation.
    
    Args:
        predictions: List of prediction tuples
        k (int): K value for ranking metrics (default: 10)
        threshold (float): Relevance threshold (default: 4.0)
        verbose (bool): Print results (default: True)
        cold_start_users: Optional set of cold start user IDs (default: None)
        cold_start_items: Optional set of cold start item IDs (default: None)
        
    Returns:
        dict: Dictionary with all metric values
    """
    results = {
        'rmse': calculate_rmse(predictions),
        'mae': calculate_mae(predictions),
        f'precision@{k}': precision_at_k(predictions, k=k, threshold=threshold),
        f'recall@{k}': recall_at_k(predictions, k=k, threshold=threshold),
        f'ndcg@{k}': ndcg_at_k(predictions, k=k, threshold=threshold),
        f'hit_rate@{k}': hit_rate_at_k(predictions, k=k, threshold=threshold),
    }
    
    # Add cold start metrics if provided
    if cold_start_users is not None:
        cold_user_results = evaluate_cold_start_users(
            predictions, cold_start_users, verbose=False
        )
        results.update(cold_user_results)
    
    if cold_start_items is not None:
        cold_item_results = evaluate_cold_start_items(
            predictions, cold_start_items, verbose=False
        )
        results.update(cold_item_results)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"\nRating Prediction Metrics:")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAE:  {results['mae']:.4f}")
        
        print(f"\nRanking Metrics (K={k}, threshold={threshold}):")
        print(f"  Precision@{k}: {results[f'precision@{k}']:.4f}")
        print(f"  Recall@{k}:    {results[f'recall@{k}']:.4f}")
        print(f"  NDCG@{k}:      {results[f'ndcg@{k}']:.4f}")
        print(f"  Hit Rate@{k}:  {results[f'hit_rate@{k}']:.4f}")
        
        if cold_start_users is not None and results.get('cold_user_rmse') is not None:
            print(f"\nCold Start User Metrics:")
            print(f"  RMSE: {results['cold_user_rmse']:.4f}")
            print(f"  MAE:  {results['cold_user_mae']:.4f}")
            print(f"  Coverage: {results['cold_user_coverage']:.2%}")
        
        if cold_start_items is not None and results.get('cold_item_rmse') is not None:
            print(f"\nCold Start Item Metrics:")
            print(f"  RMSE: {results['cold_item_rmse']:.4f}")
            print(f"  MAE:  {results['cold_item_mae']:.4f}")
            print(f"  Coverage: {results['cold_item_coverage']:.2%}")
        
        print("=" * 60)
    
    return results


def compare_svd_als(svd_model, als_model, testset):
    """
    Compare predictions from SVD and ALS models.
    
    Args:
        svd_model: Trained SVD model
        als_model: Trained ALS model
        testset: Test set for evaluation
        
    Returns:
        dict: Dictionary with comparison results
    """
    print("\n" + "=" * 60)
    print("Model Comparison: SVD vs ALS")
    print("=" * 60)
    
    # Get predictions from both models
    svd_predictions = svd_model.test(testset)
    als_predictions = []
    
    for pred in svd_predictions:
        uid = pred.uid
        iid = pred.iid
        true_r = pred.r_ui
        als_pred_r = als_model.predict(uid, iid)
        als_predictions.append((uid, iid, true_r, als_pred_r))
    
    # Calculate metrics
    svd_rmse = calculate_rmse(svd_predictions)
    als_rmse = calculate_rmse(als_predictions)
    
    svd_mae = calculate_mae(svd_predictions)
    als_mae = calculate_mae(als_predictions)
    
    print(f"\nRMSE:")
    print(f"  SVD: {svd_rmse:.4f}")
    print(f"  ALS: {als_rmse:.4f}")
    print(f"  Difference: {abs(svd_rmse - als_rmse):.4f}")
    
    print(f"\nMAE:")
    print(f"  SVD: {svd_mae:.4f}")
    print(f"  ALS: {als_mae:.4f}")
    print(f"  Difference: {abs(svd_mae - als_mae):.4f}")
    
    print("=" * 60)
    
    return {
        'svd': {'rmse': svd_rmse, 'mae': svd_mae},
        'als': {'rmse': als_rmse, 'mae': als_mae}
    }


if __name__ == "__main__":
    # Example usage
    from common.data_loader import load_movielens_100k, get_train_test_split
    from matrix_factorization.mf_svd import train_svd_model
    
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data)
    
    # Train model
    print("\nTraining SVD model...")
    model = train_svd_model(trainset, n_factors=50, n_epochs=20, verbose=False)
    
    # Make predictions
    predictions = model.test(testset)
    
    # Evaluate
    results = evaluate_model(predictions, k=10, threshold=4.0)
