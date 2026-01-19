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


def evaluate_model(predictions, k=10, threshold=4.0, verbose=True):
    """
    Comprehensive evaluation of a model on a test set.
    
    Calculates all metrics: RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate@K.
    
    Args:
        predictions: List of prediction tuples
        k (int): K value for ranking metrics (default: 10)
        threshold (float): Relevance threshold (default: 4.0)
        verbose (bool): Print results (default: True)
        
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
        print("=" * 60)
    
    return results


def compare_sgd_als(sgd_model, als_model, testset):
    """
    Compare predictions from SGD and ALS models.
    
    Args:
        sgd_model: Trained SGD model
        als_model: Trained ALS model
        testset: Test set for evaluation
        
    Returns:
        dict: Dictionary with comparison results
    """
    print("\n" + "=" * 60)
    print("Model Comparison: SGD vs ALS")
    print("=" * 60)
    
    # Get predictions from both models
    sgd_predictions = sgd_model.test(testset)
    als_predictions = []
    
    for pred in sgd_predictions:
        uid = pred.uid
        iid = pred.iid
        true_r = pred.r_ui
        als_pred_r = als_model.predict(uid, iid)
        als_predictions.append((uid, iid, true_r, als_pred_r))
    
    # Calculate metrics
    sgd_rmse = calculate_rmse(sgd_predictions)
    als_rmse = calculate_rmse(als_predictions)
    
    sgd_mae = calculate_mae(sgd_predictions)
    als_mae = calculate_mae(als_predictions)
    
    print(f"\nRMSE:")
    print(f"  SGD: {sgd_rmse:.4f}")
    print(f"  ALS: {als_rmse:.4f}")
    print(f"  Difference: {abs(sgd_rmse - als_rmse):.4f}")
    
    print(f"\nMAE:")
    print(f"  SGD: {sgd_mae:.4f}")
    print(f"  ALS: {als_mae:.4f}")
    print(f"  Difference: {abs(sgd_mae - als_mae):.4f}")
    
    print("=" * 60)
    
    return {
        'sgd': {'rmse': sgd_rmse, 'mae': sgd_mae},
        'als': {'rmse': als_rmse, 'mae': als_mae}
    }


if __name__ == "__main__":
    # Example usage
    from data_loader import load_movielens_100k, get_train_test_split
    from mf_sgd import train_sgd_model
    
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data)
    
    # Train model
    print("\nTraining SGD model...")
    model = train_sgd_model(trainset, n_factors=50, n_epochs=20, verbose=False)
    
    # Make predictions
    predictions = model.test(testset)
    
    # Evaluate
    results = evaluate_model(predictions, k=10, threshold=4.0)

