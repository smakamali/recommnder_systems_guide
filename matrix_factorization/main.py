"""
Main Pipeline for Matrix Factorization Experiments.

This script orchestrates the complete pipeline:
1. Load MovieLens 100K data
2. Split into train/test
3. Train ALS model
4. Train SGD model
5. Evaluate both models (RMSE, MAE, Precision@K, Recall@K)
6. Generate sample recommendations
7. Print comparison results
8. Save results to results/

Reference: Guide Section 2.3 - Matrix Factorization
"""

import os
import sys
from datetime import datetime

# Import modules
from data_loader import load_movielens_100k, get_train_test_split, get_dataset_stats
from mf_als import train_als_model
from mf_sgd import train_sgd_model
from evaluation import evaluate_model, compare_sgd_als
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
    
    # Step 2: Split into train/test (80/20)
    print("\n[Step 2] Splitting into train/test (80/20)...")
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    print(f"  Training set: {trainset.n_ratings:,} ratings")
    print(f"  Test set: {len(testset):,} ratings")
    print(f"  Users: {trainset.n_users:,}, Items: {trainset.n_items:,}")
    
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
    
    # Step 4: Train SGD model
    print("\n" + "=" * 60)
    print("[Step 4] Training SGD Matrix Factorization...")
    print("=" * 60)
    sgd_model = train_sgd_model(
        trainset,
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )
    
    # Step 5: Evaluate both models
    print("\n" + "=" * 60)
    print("[Step 5] Evaluating Models...")
    print("=" * 60)
    
    print("\nEvaluating ALS model...")
    als_predictions = []
    for uid, iid, true_r in testset:
        pred_r = als_model.predict(uid, iid)
        als_predictions.append((uid, iid, true_r, pred_r))
    
    als_results = evaluate_model(als_predictions, k=10, threshold=4.0, verbose=True)
    
    print("\nEvaluating SGD model...")
    sgd_predictions = sgd_model.test(testset)
    sgd_results = evaluate_model(sgd_predictions, k=10, threshold=4.0, verbose=True)
    
    # Compare models using evaluation function
    comparison_results = compare_sgd_als(sgd_model, als_model, testset)
    
    # Step 6: Generate sample recommendations
    print("\n" + "=" * 60)
    print("[Step 6] Generating Sample Recommendations...")
    print("=" * 60)
    
    # Use SGD model for recommendations (can switch to ALS)
    sample_user = testset[0][0]
    print(f"\nGenerating top-10 recommendations for user {sample_user} (SGD model)...")
    
    recommendations = generate_top_n_recommendations(
        sgd_model, trainset, sample_user, n=10, exclude_rated=True
    )
    
    print_recommendations(sample_user, recommendations, max_display=10)
    
    # Also show ALS recommendations for comparison
    print(f"\nGenerating top-10 recommendations for user {sample_user} (ALS model)...")
    als_recommendations = generate_top_n_recommendations(
        als_model, trainset, sample_user, n=10, exclude_rated=True
    )
    print_recommendations(sample_user, als_recommendations, max_display=10)
    
    # Step 7: Save results
    print("\n" + "=" * 60)
    print("[Step 7] Saving Results...")
    print("=" * 60)
    
    results_summary = {
        'Dataset': 'MovieLens 100K',
        'Train/Test Split': '80/20',
        'Models': {
            'ALS': {
                'n_factors': 50,
                'reg': 0.1,
                'n_iter': 50,
                **als_results
            },
            'SGD': {
                'n_factors': 50,
                'n_epochs': 20,
                'lr_all': 0.005,
                'reg_all': 0.02,
                **sgd_results
            }
        }
    }
    
    save_results(results_summary, "matrix_factorization_results.txt")
    
    print("\n" + "=" * 60)
    print("Pipeline Completed Successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check results/results.txt for detailed metrics")
    print("  2. Experiment with different hyperparameters")
    print("  3. Try the educational ALS from scratch: python mf_als_from_scratch.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

