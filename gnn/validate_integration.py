"""
Validation script to test GraphSAGE integration with evaluation module.

Tests that:
1. GraphSAGE can be trained
2. Predictions are in Surprise format
3. Evaluation module works with GraphSAGE predictions
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data_loader import (load_movielens_100k, get_train_test_split,
                                load_user_features, load_item_features)
from common.evaluation import (evaluate_model, calculate_rmse, calculate_mae,
                              precision_at_k, recall_at_k, ndcg_at_k)
from gnn.graphsage_recommender import train_graphsage_recommender


def test_graphsage_evaluation():
    """Test that GraphSAGE predictions work with evaluation module."""
    print("=" * 60)
    print("GraphSAGE Integration Validation")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    print(f"  Training: {trainset.n_ratings:,} ratings")
    print(f"  Test: {len(testset):,} ratings")
    
    # Load features
    print("\n[2] Loading features...")
    user_features = load_user_features()
    item_features = load_item_features()
    print(f"  User features: {user_features.shape}")
    print(f"  Item features: {item_features.shape}")
    
    # Train GraphSAGE (short training for validation)
    print("\n[3] Training GraphSAGE (2 epochs for quick validation)...")
    try:
        model = train_graphsage_recommender(
            trainset, user_features, item_features,
            hidden_dim=64,
            num_layers=2,
            num_epochs=2,  # Short training for validation
            batch_size=256,
            verbose=True
        )
        print("  ✓ Model trained successfully")
    except Exception as e:
        print(f"  [ERROR] Training failed: {str(e)}")
        return False
    
    # Test prediction format
    print("\n[4] Testing prediction format...")
    try:
        # Test single prediction
        sample_user, sample_item, true_rating = testset[0]
        pred_rating = model.predict(sample_user, sample_item)
        print(f"  Sample prediction: User {sample_user}, Item {sample_item}")
        print(f"    True: {true_rating:.2f}, Predicted: {pred_rating:.2f}")
        print("  ✓ Single prediction works")
        
        # Test batch predictions (test() method)
        test_subset = testset[:100]  # Small subset for validation
        predictions = model.test(test_subset)
        print(f"  Generated {len(predictions)} predictions")
        
        # Check format
        if len(predictions) > 0:
            pred = predictions[0]
            if hasattr(pred, 'uid') and hasattr(pred, 'iid') and hasattr(pred, 'r_ui') and hasattr(pred, 'est'):
                print("  [OK] Predictions are in Surprise format")
            else:
                print("  ✗ Predictions not in correct format")
                return False
        else:
            print("  [ERROR] No predictions generated")
            return False
    except Exception as e:
        print(f"  ✗ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test evaluation functions
    print("\n[5] Testing evaluation functions...")
    try:
        # Test individual metrics
        rmse = calculate_rmse(predictions)
        mae = calculate_mae(predictions)
        precision = precision_at_k(predictions, k=10)
        recall = recall_at_k(predictions, k=10)
        ndcg = ndcg_at_k(predictions, k=10)
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  Precision@10: {precision:.4f}")
        print(f"  Recall@10:    {recall:.4f}")
        print(f"  NDCG@10:      {ndcg:.4f}")
        print("  ✓ All evaluation metrics work")
        
        # Test comprehensive evaluation
        results = evaluate_model(predictions, k=10, threshold=4.0, verbose=False)
        if 'rmse' in results and 'mae' in results:
            print("  [OK] Comprehensive evaluation works")
        else:
            print("  ✗ Comprehensive evaluation missing metrics")
            return False
    except Exception as e:
        print(f"  ✗ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All validation tests passed!")
    print("=" * 60)
    print("\nGraphSAGE is fully integrated with the evaluation module.")
    print("You can now use it for comparison with Factorization Machines.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_graphsage_evaluation()
    sys.exit(0 if success else 1)
