"""
Main Pipeline for Matrix Factorization Experiments.

This script orchestrates the complete pipeline:
1. Load MovieLens 100K data
2. Split into train/test
3. Train ALS model
4. Train SVD model
5. Evaluate both models (RMSE, MAE, Precision@K, Recall@K)
6. Generate sample recommendations
7. Print comparison results
8. Save results to results/

Reference: Guide Section 2.3 - Matrix Factorization
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports when running as script
# This allows imports from common/ to work when running this file directly
parent_path = str(Path(__file__).parent.parent)
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

# Import modules
from common.data_loader import (load_movielens_100k, get_train_test_split, get_dataset_stats,
                         load_user_features, load_item_features, get_cold_start_split)
from mf_als import train_als_model
from mf_svd import train_svd_model
from mf_als_from_scratch import train_als_from_scratch_model
from mf_fm import train_fm_model
from common.evaluation import evaluate_model, evaluate_with_cold_start_breakdown
from recommend import generate_top_n_recommendations, print_recommendations


def save_results(results_dict, filename="results.txt"):
    """
    Save evaluation results to a file.
    
    Args:
        results_dict: Dictionary containing results
        filename (str): Output filename (default: "results.txt")
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Matrix Factorization Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in results_dict.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to {filepath}")


def main():
    """
    Main pipeline for matrix factorization experiments.
    """
    print("=" * 60)
    print("Matrix Factorization Pipeline")
    print("MovieLens 100K Dataset")
    print("=" * 60)
    
    # Step 1: Load MovieLens 100K data
    print("\n[Step 1] Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    get_dataset_stats(data)
    
    # Step 2: Load user and item features
    print("\n[Step 2a] Loading user and item features...")
    user_features = load_user_features()
    item_features = load_item_features()
    print(f"  User features: {user_features.shape}")
    print(f"  Item features: {item_features.shape}")
    
    # Step 2b: Split into train/test (80/20)
    print("\n[Step 2b] Splitting into train/test (80/20)...")
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    print(f"  Training set: {trainset.n_ratings:,} ratings")
    print(f"  Test set: {len(testset):,} ratings")
    print(f"  Users: {trainset.n_users:,}, Items: {trainset.n_items:,}")
    
    # Step 2c: Create cold start split
    print("\n[Step 2c] Creating cold start test sets...")
    trainset_cs, testset_cs, test_cold_users, test_cold_items = get_cold_start_split(
        data, user_features, item_features, cold_start_ratio=0.1, 
        test_size=0.2, random_state=42,
        cold_user_threshold=20,  # Users with <5 ratings are considered cold start
        cold_item_threshold=10  # Items with <10 ratings are considered cold start
    )
    print(f"  Warm test: {len(testset_cs):,} ratings")
    print(f"  Cold start users: {len(test_cold_users)} users")
    print(f"  Cold start items: {len(test_cold_items)} items")
    
    # Step 3: Train ALS model
    print("\n" + "=" * 60)
    print("[Step 3] Training ALS Matrix Factorization...")
    print("=" * 60)
    als_model = train_als_model(
        trainset, 
        n_factors=50, 
        reg=0.1, 
        n_iter=50, 
        random_state=42
    )
    
    # Step 4: Train ALS from Scratch
    print("\n" + "=" * 60)
    print("[Step 4] Training ALS Matrix Factorization from Scratch...")
    print("=" * 60)
    als_scratch_model = train_als_from_scratch_model(
        trainset,
        k=50,
        lambda_reg=0.1,
        iterations=50,
        random_state=42,
        verbose=True
    )
    
    # Step 5: Train SVD model
    print("\n" + "=" * 60)
    print("[Step 5] Training SVD Matrix Factorization...")
    print("=" * 60)
    svd_model = train_svd_model(
        trainset,
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )
    
    # Step 6: Train Factorization Machine with Features
    print("\n" + "=" * 60)
    print("[Step 6] Training Factorization Machine with Features...")
    print("=" * 60)
    try:
        fm_model = train_fm_model(
            trainset,
            user_features,
            item_features,
            n_factors=50,
            learning_rate=0.1,
            reg_lambda=0.01,
            n_epochs=30,
            verbose=True
        )
        fm_available = True
    except (ImportError, Exception) as e:
        print(f"  Warning: myFM not available. Skipping FM training.")
        print(f"  Error: {str(e)}")
        print(f"  Install with: pip install myfm")
        print(f"  Note: myFM has cross-platform support (Windows, Linux, macOS)")
        fm_model = None
        fm_available = False
    
    # Step 7: Evaluate all models
    print("\n" + "=" * 60)
    print("[Step 6] Evaluating All Models...")
    print("=" * 60)
    
    print("\nEvaluating ALS (implicit library) model...")
    als_predictions = []
    for uid, iid, true_r in testset:
        pred_r = als_model.predict(uid, iid)
        als_predictions.append((uid, iid, true_r, pred_r))
    
    als_results = evaluate_model(als_predictions, k=10, threshold=4.0, verbose=True)
    
    print("\nEvaluating ALS from Scratch model...")
    als_scratch_predictions = []
    for uid, iid, true_r in testset:
        pred_r = als_scratch_model.predict(uid, iid)
        als_scratch_predictions.append((uid, iid, true_r, pred_r))
    
    als_scratch_results = evaluate_model(als_scratch_predictions, k=10, threshold=4.0, verbose=True)
    
    print("\nEvaluating SVD model...")
    svd_predictions = svd_model.test(testset)
    svd_results = evaluate_model(svd_predictions, k=10, threshold=4.0, verbose=True)
    
    # Evaluate FM model if available
    fm_results = None
    fm_cold_results = None
    if fm_available:
        print("\nEvaluating Factorization Machine model...")
        fm_predictions = fm_model.test(testset)
        fm_results = evaluate_model(
            fm_predictions, k=10, threshold=4.0, verbose=True,
            cold_start_users=test_cold_users,
            cold_start_items=test_cold_items
        )
        
        # Cold start evaluation for FM
        print("\nEvaluating FM on cold start scenarios...")
        fm_cold_results = evaluate_with_cold_start_breakdown(
            fm_predictions, test_cold_users, test_cold_items, verbose=True
        )
    
    # Compare all models
    print("\n" + "=" * 60)
    if fm_available:
        print("Model Comparison: All Four Models")
    else:
        print("Model Comparison: All Three Models")
    print("=" * 60)
    
    print(f"\nRMSE:")
    print(f"  ALS (implicit):     {als_results['rmse']:.4f}")
    print(f"  ALS (from scratch): {als_scratch_results['rmse']:.4f}")
    print(f"  SVD:                {svd_results['rmse']:.4f}")
    if fm_available:
        print(f"  FM (with features): {fm_results['rmse']:.4f}")
    
    print(f"\nMAE:")
    print(f"  ALS (implicit):     {als_results['mae']:.4f}")
    print(f"  ALS (from scratch): {als_scratch_results['mae']:.4f}")
    print(f"  SVD:                {svd_results['mae']:.4f}")
    if fm_available:
        print(f"  FM (with features): {fm_results['mae']:.4f}")
    
    print(f"\nPrecision@10:")
    print(f"  ALS (implicit):     {als_results['precision@10']:.4f}")
    print(f"  ALS (from scratch): {als_scratch_results['precision@10']:.4f}")
    print(f"  SVD:                {svd_results['precision@10']:.4f}")
    if fm_available:
        print(f"  FM (with features): {fm_results['precision@10']:.4f}")
    
    print(f"\nRecall@10:")
    print(f"  ALS (implicit):     {als_results['recall@10']:.4f}")
    print(f"  ALS (from scratch): {als_scratch_results['recall@10']:.4f}")
    print(f"  SVD:                {svd_results['recall@10']:.4f}")
    if fm_available:
        print(f"  FM (with features): {fm_results['recall@10']:.4f}")
    
    print(f"\nNDCG@10:")
    print(f"  ALS (implicit):     {als_results['ndcg@10']:.4f}")
    print(f"  ALS (from scratch): {als_scratch_results['ndcg@10']:.4f}")
    print(f"  SVD:                {svd_results['ndcg@10']:.4f}")
    if fm_available:
        print(f"  FM (with features): {fm_results['ndcg@10']:.4f}")
    
    print(f"\nHit Rate@10:")
    print(f"  ALS (implicit):     {als_results['hit_rate@10']:.4f}")
    # print(f"  ALS (from scratch): {als_scratch_results['hit_rate@10']:.4f}")
    print(f"  SVD:                {svd_results['hit_rate@10']:.4f}")
    if fm_available:
        print(f"  FM (with features): {fm_results['hit_rate@10']:.4f}")
    
    # Cold start comparison
    if fm_available and fm_cold_results:
        print(f"\nCold Start RMSE (new users):")
        print(f"  ALS (implicit):     N/A (cannot predict)")
        print(f"  ALS (from scratch): N/A (cannot predict)")
        print(f"  SVD:                N/A (cannot predict)")
        if fm_cold_results.get('cold_user_rmse') is not None:
            print(f"  FM (with features): {fm_cold_results['cold_user_rmse']:.4f}")
        else:
            print(f"  FM (with features): N/A (no cold start users in test set)")
        
        print(f"\nCold Start RMSE (new items):")
        print(f"  ALS (implicit):     N/A (cannot predict)")
        # print(f"  ALS (from scratch): N/A (cannot predict)")
        print(f"  SVD:                N/A (cannot predict)")
        if fm_cold_results.get('cold_item_rmse') is not None:
            print(f"  FM (with features): {fm_cold_results['cold_item_rmse']:.4f}")
        else:
            print(f"  FM (with features): N/A (no cold start items in test set)")
    
    print("=" * 60)
    
    # Step 8: Generate sample recommendations
    print("\n" + "=" * 60)
    print("[Step 8] Generating Sample Recommendations...")
    print("=" * 60)
    
    # Use SVD model for recommendations (can switch to ALS)
    sample_user = testset[0][0]
    print(f"\nGenerating top-10 recommendations for user {sample_user} (SVD model)...")
    
    recommendations = generate_top_n_recommendations(
        svd_model, trainset, sample_user, n=10, exclude_rated=True
    )
    
    print_recommendations(sample_user, recommendations, max_display=10)
    
    # Also show ALS recommendations for comparison
    print(f"\nGenerating top-10 recommendations for user {sample_user} (ALS implicit library)...")
    als_recommendations = generate_top_n_recommendations(
        als_model, trainset, sample_user, n=10, exclude_rated=True
    )
    print_recommendations(sample_user, als_recommendations, max_display=10)
    
    # Show FM recommendations if available
    if fm_available:
        print(f"\nGenerating top-10 recommendations for user {sample_user} (Factorization Machine with features)...")
        fm_recommendations = generate_top_n_recommendations(
            fm_model, trainset, sample_user, n=10, exclude_rated=True
        )
        print_recommendations(sample_user, fm_recommendations, max_display=10)
    
    print(f"\nGenerating top-10 recommendations for user {sample_user} (ALS from scratch)...")
    als_scratch_recommendations = generate_top_n_recommendations(
        als_scratch_model, trainset, sample_user, n=10, exclude_rated=True
    )
    print_recommendations(sample_user, als_scratch_recommendations, max_display=10)
    
    # Step 9: Save results
    print("\n" + "=" * 60)
    print("[Step 9] Saving Results...")
    print("=" * 60)
    
    results_summary = {
        'Dataset': 'MovieLens 100K',
        'Train/Test Split': '80/20 with 10% cold start',
        'Models': {
            'ALS (implicit library)': {
                'n_factors': 50,
                'reg': 0.1,
                'n_iter': 50,
                **als_results
            },
            'ALS (from scratch)': {
                'k': 50,
                'lambda_reg': 0.1,
                'iterations': 50,
                **als_scratch_results
            },
            'SVD': {
                'n_factors': 50,
                'n_epochs': 20,
                'lr_all': 0.005,
                'reg_all': 0.02,
                **svd_results
            }
        }
    }
    
    # Add FM results if available
    if fm_available and fm_results:
        results_summary['Models']['Factorization Machine'] = {
            'n_factors': 50,
            'learning_rate': 0.1,
            'reg_lambda': 0.01,
            'n_epochs': 30,
            **fm_results
        }
        if fm_cold_results:
            results_summary['Models']['Factorization Machine']['cold_start'] = fm_cold_results
    
    save_results(results_summary, "matrix_factorization_results.txt")
    
    print("\n" + "=" * 60)
    print("Pipeline Completed Successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check results/matrix_factorization_results.txt for detailed metrics")
    print("  2. Experiment with different hyperparameters")
    print("  3. Compare ALS from scratch vs SVD performance")
    print("=" * 60)


if __name__ == "__main__":
    main()

