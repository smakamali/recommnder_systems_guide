"""
Comparison Script: Factorization Machines vs GraphSAGE.

Side-by-side evaluation of FM and GraphSAGE on MovieLens 100K.
Compares performance on warm users, cold start users, and overall metrics.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data_loader import (load_movielens_100k, get_train_test_split,
                                load_user_features, load_item_features,
                                get_cold_start_split)
from common.evaluation import (evaluate_model, evaluate_with_cold_start_breakdown)
from gnn.graphsage_recommender import train_graphsage_recommender

# Import FM from matrix_factorization
try:
    from matrix_factorization.mf_fm import train_fm_model
    FM_AVAILABLE = True
except ImportError:
    print("Warning: Factorization Machines not available.")
    print("  Install myFM with: pip install myfm")
    FM_AVAILABLE = False


def compare_models(trainset, testset, user_features, item_features,
                   cold_start_users=None, cold_start_items=None,
                   graphsage_epochs=20, fm_epochs=30, verbose=True,
                   val_ratio=0.1, early_stopping_patience=5, 
                   early_stopping_min_delta=1e-4):
    """
    Compare FM and GraphSAGE models on the same test set.
    
    Args:
        trainset: Surprise Trainset
        testset: List of (uid, iid, rating) tuples
        user_features: DataFrame with user features
        item_features: DataFrame with item features
        cold_start_users: Optional set of cold start user IDs
        cold_start_items: Optional set of cold start item IDs
        graphsage_epochs: Number of epochs for GraphSAGE (default: 20)
        fm_epochs: Number of epochs for FM (default: 30)
        verbose: Print progress (default: True)
        val_ratio: Ratio of training data to use for validation (default: 0.1)
        early_stopping_patience: Number of epochs to wait without improvement before stopping (default: 5)
        early_stopping_min_delta: Minimum change in validation loss to qualify as improvement (default: 1e-4)
        
    Returns:
        dict: Comparison results with metrics for both models
    """
    results = {}
    
    # Train GraphSAGE
    if verbose:
        print("\n" + "=" * 60)
        print("Training GraphSAGE Model")
        print("=" * 60)
    
    graphsage_model = train_graphsage_recommender(
        trainset, user_features, item_features,
        hidden_dim=64,
        num_layers=3,
        num_epochs=graphsage_epochs,
        batch_size=512,
        learning_rate=0.001,
        loss_type='mse',  # NEW: Use MSE loss by default for better rating prediction
        val_ratio=val_ratio,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        verbose=verbose
    )
    
    # Evaluate GraphSAGE
    if verbose:
        print("\nEvaluating GraphSAGE...")
    
    graphsage_predictions = graphsage_model.test(testset)
    graphsage_results = evaluate_model(
        graphsage_predictions,
        k=10,
        threshold=4.0,
        verbose=verbose,
        cold_start_users=cold_start_users,
        cold_start_items=cold_start_items
    )
    
    results['graphsage'] = {
        'model': graphsage_model,
        'predictions': graphsage_predictions,
        'results': graphsage_results
    }
    
    # Train FM (if available)
    if FM_AVAILABLE:
        if verbose:
            print("\n" + "=" * 60)
            print("Training Factorization Machine Model")
            print("=" * 60)
        
        try:
            fm_model = train_fm_model(
                trainset, user_features, item_features,
                n_factors=50,
                learning_rate=0.1,
                reg_lambda=0.01,
                n_epochs=fm_epochs,
                verbose=verbose
            )
            
            # Evaluate FM
            if verbose:
                print("\nEvaluating Factorization Machine...")
            
            fm_predictions = fm_model.test(testset)
            fm_results = evaluate_model(
                fm_predictions,
                k=10,
                threshold=4.0,
                verbose=verbose,
                cold_start_users=cold_start_users,
                cold_start_items=cold_start_items
            )
            
            results['fm'] = {
                'model': fm_model,
                'predictions': fm_predictions,
                'results': fm_results
            }
            
        except Exception as e:
            if verbose:
                print(f"  Error training FM: {str(e)}")
            results['fm'] = None
    else:
        results['fm'] = None
    
    # Print comparison
    if verbose:
        print("\n" + "=" * 60)
        print("Model Comparison: GraphSAGE vs Factorization Machine")
        print("=" * 60)
        
        print(f"\nRating Prediction Metrics:")
        print(f"  GraphSAGE:")
        print(f"    RMSE: {graphsage_results['rmse']:.4f}")
        print(f"    MAE:  {graphsage_results['mae']:.4f}")
        
        if results['fm'] is not None:
            print(f"  Factorization Machine:")
            print(f"    RMSE: {fm_results['rmse']:.4f}")
            print(f"    MAE:  {fm_results['mae']:.4f}")
            print(f"\n  Improvement (GraphSAGE vs FM):")
            rmse_improvement = ((fm_results['rmse'] - graphsage_results['rmse']) / fm_results['rmse']) * 100
            mae_improvement = ((fm_results['mae'] - graphsage_results['mae']) / fm_results['mae']) * 100
            print(f"    RMSE: {rmse_improvement:+.2f}%")
            print(f"    MAE:  {mae_improvement:+.2f}%")
        
        print(f"\nRanking Metrics (K=10):")
        print(f"  GraphSAGE:")
        print(f"    Precision@10: {graphsage_results['precision@10']:.4f}")
        print(f"    Recall@10:    {graphsage_results['recall@10']:.4f}")
        print(f"    NDCG@10:      {graphsage_results['ndcg@10']:.4f}")
        print(f"    Hit Rate@10:  {graphsage_results['hit_rate@10']:.4f}")
        
        if results['fm'] is not None:
            print(f"  Factorization Machine:")
            print(f"    Precision@10: {fm_results['precision@10']:.4f}")
            print(f"    Recall@10:    {fm_results['recall@10']:.4f}")
            print(f"    NDCG@10:      {fm_results['ndcg@10']:.4f}")
            print(f"    Hit Rate@10:  {fm_results['hit_rate@10']:.4f}")
        
        # Cold start comparison
        if cold_start_users is not None:
            print(f"\nCold Start User Metrics:")
            if graphsage_results.get('cold_user_rmse') is not None:
                print(f"  GraphSAGE:")
                print(f"    Rating Prediction:")
                print(f"      RMSE: {graphsage_results['cold_user_rmse']:.4f}")
                print(f"      MAE:  {graphsage_results['cold_user_mae']:.4f}")
                print(f"      Coverage: {graphsage_results['cold_user_coverage']:.2%}")
                print(f"    Ranking (K=10):")
                print(f"      Precision@10: {graphsage_results['cold_user_precision@10']:.4f}")
                print(f"      Recall@10:    {graphsage_results['cold_user_recall@10']:.4f}")
                print(f"      NDCG@10:      {graphsage_results['cold_user_ndcg@10']:.4f}")
                print(f"      Hit Rate@10:  {graphsage_results['cold_user_hit_rate@10']:.4f}")
            
            if results['fm'] is not None and fm_results.get('cold_user_rmse') is not None:
                print(f"  Factorization Machine:")
                print(f"    Rating Prediction:")
                print(f"      RMSE: {fm_results['cold_user_rmse']:.4f}")
                print(f"      MAE:  {fm_results['cold_user_mae']:.4f}")
                print(f"      Coverage: {fm_results['cold_user_coverage']:.2%}")
                print(f"    Ranking (K=10):")
                print(f"      Precision@10: {fm_results['cold_user_precision@10']:.4f}")
                print(f"      Recall@10:    {fm_results['cold_user_recall@10']:.4f}")
                print(f"      NDCG@10:      {fm_results['cold_user_ndcg@10']:.4f}")
                print(f"      Hit Rate@10:  {fm_results['cold_user_hit_rate@10']:.4f}")
        
        if cold_start_items is not None:
            print(f"\nCold Start Item Metrics:")
            if graphsage_results.get('cold_item_rmse') is not None:
                print(f"  GraphSAGE:")
                print(f"    Rating Prediction:")
                print(f"      RMSE: {graphsage_results['cold_item_rmse']:.4f}")
                print(f"      MAE:  {graphsage_results['cold_item_mae']:.4f}")
                print(f"      Coverage: {graphsage_results['cold_item_coverage']:.2%}")
                print(f"    Ranking (K=10):")
                print(f"      Precision@10: {graphsage_results['cold_item_precision@10']:.4f}")
                print(f"      Recall@10:    {graphsage_results['cold_item_recall@10']:.4f}")
                print(f"      NDCG@10:      {graphsage_results['cold_item_ndcg@10']:.4f}")
                print(f"      Hit Rate@10:  {graphsage_results['cold_item_hit_rate@10']:.4f}")
            
            if results['fm'] is not None and fm_results.get('cold_item_rmse') is not None:
                print(f"  Factorization Machine:")
                print(f"    Rating Prediction:")
                print(f"      RMSE: {fm_results['cold_item_rmse']:.4f}")
                print(f"      MAE:  {fm_results['cold_item_mae']:.4f}")
                print(f"      Coverage: {fm_results['cold_item_coverage']:.2%}")
                print(f"    Ranking (K=10):")
                print(f"      Precision@10: {fm_results['cold_item_precision@10']:.4f}")
                print(f"      Recall@10:    {fm_results['cold_item_recall@10']:.4f}")
                print(f"      NDCG@10:      {fm_results['cold_item_ndcg@10']:.4f}")
                print(f"      Hit Rate@10:  {fm_results['cold_item_hit_rate@10']:.4f}")
        
        print("=" * 60)
    
    return results


def save_results(results: dict, filepath: str):
    """
    Save comparison results to a JSON file.
    
    Args:
        results: Results dictionary from compare_models
        filepath: Path to save JSON file
    """
    # Prepare results for JSON serialization
    serializable_results = {
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    # Save GraphSAGE results
    if results.get('graphsage') is not None:
        graphsage_data = results['graphsage']
        serializable_results['models']['graphsage'] = {
            'results': graphsage_data['results'] if graphsage_data.get('results') else None
        }
    
    # Save FM results
    if results.get('fm') is not None:
        fm_data = results['fm']
        serializable_results['models']['fm'] = {
            'results': fm_data['results'] if fm_data.get('results') else None
        }
    else:
        serializable_results['models']['fm'] = None
    
    # Save to file
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main comparison pipeline."""
    print("=" * 60)
    print("GraphSAGE vs Factorization Machine Comparison")
    print("MovieLens 100K Dataset")
    print("=" * 60)
    
    # Load data
    print("\n[Step 1] Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"  Training set: {trainset.n_ratings:,} ratings")
    print(f"  Test set: {len(testset):,} ratings")
    print(f"  Users: {trainset.n_users:,}, Items: {trainset.n_items:,}")
    
    # Load features
    print("\n[Step 2] Loading user and item features...")
    user_features = load_user_features()
    item_features = load_item_features()
    print(f"  User features: {user_features.shape}")
    print(f"  Item features: {item_features.shape}")
    
    # Create cold start split
    print("\n[Step 3] Creating cold start test sets...")
    trainset_cs, testset_cs, cold_users, cold_items = get_cold_start_split(
        data, user_features, item_features,
        test_size=0.2,
        random_state=42,
        cold_user_threshold=5,  # Users with <5 ratings are cold start
        cold_item_threshold=10  # Items with <10 ratings are cold start
    )
    print(f"  Cold start users: {len(cold_users)}")
    print(f"  Cold start items: {len(cold_items)}")
    
    # Compare models
    print("\n[Step 4] Training and comparing models...")
    results = compare_models(
        trainset,
        testset,
        user_features,
        item_features,
        cold_start_users=cold_users,
        cold_start_items=cold_items,
        graphsage_epochs=150,  # Increased for early stopping
        fm_epochs=30,
        val_ratio=0.1,
        early_stopping_patience=15,
        early_stopping_min_delta=1e-4,
        verbose=True
    )
    
    # Cold start breakdown
    if cold_users or cold_items:
        print("\n[Step 5] Cold Start Breakdown Analysis...")
        
        if results['graphsage']:
            print("\n--- GraphSAGE Cold Start Breakdown ---")
            graphsage_cold = evaluate_with_cold_start_breakdown(
                results['graphsage']['predictions'],
                cold_users,
                cold_items,
                k=10,
                threshold=4.0,
                verbose=True
            )
        
        if results['fm'] is not None:
            print("\n--- Factorization Machine Cold Start Breakdown ---")
            fm_cold = evaluate_with_cold_start_breakdown(
                results['fm']['predictions'],
                cold_users,
                cold_items,
                k=10,
                threshold=4.0,
                verbose=True
            )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gnn/tuning/comparison_fm_graphsage_{timestamp}.json"
    save_results(results, results_file)
    
    print("\n" + "=" * 60)
    print("Comparison Completed!")
    print("=" * 60)
    print("\nKey Findings:")
    print("  - GraphSAGE leverages graph structure for collaborative signals")
    print("  - Both models use user/item features for cold start")
    print("  - GraphSAGE's inductive learning helps with new users/items")
    print("=" * 60)


if __name__ == "__main__":
    main()
