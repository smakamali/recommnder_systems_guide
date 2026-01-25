"""
Recommendation Generation.

This module generates top-N recommendations for users using trained
matrix factorization models. Handles cold start scenarios and filters
already-rated items.
"""

import numpy as np
from collections import defaultdict


def generate_top_n_recommendations(model, trainset, user_id, n=10, 
                                   exclude_rated=True, verbose=False):
    """
    Generate top-N recommendations for a user.
    
    Args:
        model: Trained model with predict() method
        trainset: Surprise Trainset object
        user_id: Raw user id (as in dataset)
        n (int): Number of recommendations to generate (default: 10)
        exclude_rated (bool): Exclude items the user has already rated (default: True)
        verbose (bool): Print details (default: False)
        
    Returns:
        list: List of tuples (item_id, predicted_rating) sorted by rating (descending)
    """
    if exclude_rated:
        # Get items the user has already rated
        try:
            inner_uid = trainset.to_inner_uid(user_id)
            rated_items = set()
            for inner_iid, rating in trainset.ur[inner_uid]:
                raw_iid = trainset.to_raw_iid(inner_iid)
                rated_items.add(raw_iid)
        except ValueError:
            # User not in training set (cold start)
            rated_items = set()
    else:
        rated_items = set()
    
    # Get all items in the dataset
    all_items = set()
    for inner_iid in range(trainset.n_items):
        raw_iid = trainset.to_raw_iid(inner_iid)
        all_items.add(raw_iid)
    
    # Get candidate items (all items minus rated items)
    candidate_items = all_items - rated_items
    
    # Predict ratings for all candidate items
    predictions = []
    for item_id in candidate_items:
        try:
            pred_rating = model.predict(user_id, item_id)
            # Handle different prediction formats
            if hasattr(pred_rating, 'est'):
                pred_rating = pred_rating.est
            elif isinstance(pred_rating, tuple):
                pred_rating = pred_rating[0]
            
            predictions.append((item_id, pred_rating))
        except Exception as e:
            # Skip items that can't be predicted (e.g., cold start issues)
            if verbose:
                print(f"  Warning: Could not predict for item {item_id}: {e}")
            continue
    
    # Sort by predicted rating (descending) and take top-N
    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)
    top_n = predictions_sorted[:n]
    
    return top_n


def generate_recommendations_for_all_users(model, trainset, n=10, 
                                           exclude_rated=True, sample_size=None):
    """
    Generate top-N recommendations for all users (or a sample).
    
    Args:
        model: Trained model with predict() method
        trainset: Surprise Trainset object
        n (int): Number of recommendations per user (default: 10)
        exclude_rated (bool): Exclude rated items (default: True)
        sample_size (int): If specified, only generate for first N users (default: None)
        
    Returns:
        dict: Dictionary mapping user_id -> list of (item_id, rating) tuples
    """
    recommendations = {}
    
    # Get all user IDs
    all_users = []
    for inner_uid in range(trainset.n_users):
        raw_uid = trainset.to_raw_uid(inner_uid)
        all_users.append(raw_uid)
    
    # Optionally sample users
    if sample_size is not None:
        all_users = all_users[:sample_size]
    
    print(f"Generating recommendations for {len(all_users)} users...")
    
    for i, user_id in enumerate(all_users):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_users)} users...")
        
        recs = generate_top_n_recommendations(
            model, trainset, user_id, n=n, 
            exclude_rated=exclude_rated, verbose=False
        )
        recommendations[user_id] = recs
    
    return recommendations


def print_recommendations(user_id, recommendations, item_names=None, 
                          max_display=10):
    """
    Print recommendations for a user in a nice format.
    
    Args:
        user_id: User ID
        recommendations: List of (item_id, rating) tuples
        item_names: Optional dict mapping item_id -> item name (default: None)
        max_display: Maximum number of recommendations to display (default: 10)
    """
    print(f"\nTop {min(len(recommendations), max_display)} Recommendations for User {user_id}:")
    print("-" * 60)
    
    for i, (item_id, rating) in enumerate(recommendations[:max_display], start=1):
        if item_names and item_id in item_names:
            item_name = item_names[item_id]
            print(f"{i:2d}. {item_id:5s} - {item_name[:45]:45s} (Predicted: {rating:.2f})")
        else:
            print(f"{i:2d}. Item {item_id:5s} (Predicted Rating: {rating:.2f})")


def get_item_names_movielens(trainset):
    """
    Load MovieLens item names (movie titles).
    
    This is a helper function that attempts to load movie names if available.
    Note: Surprise's built-in MovieLens dataset may not include movie names.
    
    Args:
        trainset: Surprise Trainset object
        
    Returns:
        dict: Dictionary mapping item_id -> movie name (if available)
    """
    # Try to load from u.item file (MovieLens 100K format)
    item_names = {}
    
    try:
        import os
        # Surprise stores data in a cache directory
        # We'd need the actual data path, which Surprise manages internally
        # For now, return empty dict
        pass
    except Exception:
        pass
    
    return item_names


def generate_recommendations_with_features(fm_model, user_id, user_features, 
                                           item_features, trainset, n=10, 
                                           exclude_rated=True, verbose=False):
    """
    Generate recommendations using FM with feature support.
    
    Works for both warm and cold start users (as long as features available).
    
    Args:
        fm_model: Trained FMRecommender model
        user_id: User ID (string)
        user_features: DataFrame with user features (must contain user_id)
        item_features: DataFrame with item features
        trainset: Surprise Trainset object
        n (int): Number of recommendations (default: 10)
        exclude_rated (bool): Exclude items user has already rated (default: True)
        verbose (bool): Print details (default: False)
        
    Returns:
        list: List of (item_id, predicted_rating) tuples
    """
    # Get user's feature row
    user_row = user_features[user_features['user_id'] == str(user_id)]
    if len(user_row) == 0:
        if verbose:
            print(f"  Warning: User {user_id} not found in user_features")
        return []
    
    # Get items to predict for
    if exclude_rated:
        try:
            inner_uid = trainset.to_inner_uid(user_id)
            rated_items = set()
            for inner_iid, rating in trainset.ur[inner_uid]:
                raw_iid = trainset.to_raw_iid(inner_iid)
                rated_items.add(raw_iid)
        except ValueError:
            # User not in training set (cold start) - no rated items
            rated_items = set()
    else:
        rated_items = set()
    
    # Get all items
    all_items = set()
    for inner_iid in range(trainset.n_items):
        raw_iid = trainset.to_raw_iid(inner_iid)
        all_items.add(raw_iid)
    
    # Candidate items
    candidate_items = all_items - rated_items
    
    # Predict ratings for all candidate items
    predictions = []
    for item_id in candidate_items:
        try:
            pred_rating = fm_model.predict(str(user_id), str(item_id))
            predictions.append((item_id, pred_rating))
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not predict for item {item_id}: {e}")
            continue
    
    # Sort by predicted rating (descending) and take top-N
    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)
    top_n = predictions_sorted[:n]
    
    return top_n


def handle_cold_start_user(model, trainset, user_id, n=10, verbose=False,
                           fm_model=None, user_features=None, item_features=None):
    """
    Enhanced cold start handling.
    
    Priority:
    1. If FM model + features available: Use feature-based prediction
    2. Else: Fall back to popularity-based recommendation
    
    Args:
        model: Trained model (may not work for cold start users)
        trainset: Surprise Trainset object
        user_id: Raw user id (not in training set)
        n (int): Number of recommendations (default: 10)
        verbose (bool): Print details (default: False)
        fm_model: Optional FMRecommender model (default: None)
        user_features: Optional DataFrame with user features (default: None)
        item_features: Optional DataFrame with item features (default: None)
        
    Returns:
        list: List of (item_id, predicted_rating) tuples
    """
    # Check if user is in training set
    try:
        trainset.to_inner_uid(user_id)
        # User exists - use normal recommendation
        return generate_top_n_recommendations(model, trainset, user_id, n=n)
    except ValueError:
        # Cold start - user not in training set
        if verbose:
            print(f"  Cold start detected for user {user_id}")
        
        # Try FM model with features first
        if fm_model is not None and user_features is not None and item_features is not None:
            if verbose:
                print(f"  Using FM model with features for cold start user")
            try:
                return generate_recommendations_with_features(
                    fm_model, user_id, user_features, item_features, 
                    trainset, n=n, exclude_rated=True, verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"  FM prediction failed: {e}, falling back to popularity")
        
        # Fallback: Recommend most popular items (by average rating)
        item_ratings = defaultdict(list)
        
        for uid, iid, rating in trainset.all_ratings():
            item_ratings[iid].append(rating)
        
        # Calculate average ratings
        item_avg_ratings = {}
        for iid, ratings in item_ratings.items():
            raw_iid = trainset.to_raw_iid(iid)
            item_avg_ratings[raw_iid] = np.mean(ratings)
        
        # Sort by average rating (descending) and take top-N
        popular_items = sorted(
            item_avg_ratings.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n]
        
        return popular_items


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import numpy as np
    from common.data_loader import load_movielens_100k, get_train_test_split
    from mf_svd import train_svd_model
    
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data)
    
    # Train model
    print("\nTraining SVD model...")
    model = train_svd_model(trainset, n_factors=50, n_epochs=20, verbose=False)
    
    # Generate recommendations for a sample user
    sample_user = testset[0][0]
    print(f"\nGenerating recommendations for user {sample_user}...")
    
    recommendations = generate_top_n_recommendations(
        model, trainset, sample_user, n=10, exclude_rated=True
    )
    
    print_recommendations(sample_user, recommendations, max_display=10)
    
    # Test cold start handling
    print("\n\nTesting cold start handling...")
    cold_start_user = "999999"  # User ID that doesn't exist
    cold_start_recs = handle_cold_start_user(
        model, trainset, cold_start_user, n=5, verbose=True
    )
    print_recommendations(cold_start_user, cold_start_recs, max_display=5)

