"""
Script to find the cold_user_threshold that yields approximately 500 cold start users.
Run this after activating the conda environment.
"""

import sys
from pathlib import Path

# Add parent directory to path to import common module
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from common.data_loader import (
    load_movielens_100k, 
    load_user_features, 
    load_item_features, 
    find_cold_start_threshold
)

if __name__ == "__main__":
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    
    print("Loading user and item features...")
    user_features = load_user_features()
    item_features = load_item_features()
    
    print("\nFinding threshold for ~500 cold start users...")
    threshold = find_cold_start_threshold(
        data, 
        user_features, 
        item_features, 
        target_count=50,
        verbose=True
    )
    
    print(f"\n✓ Recommended cold_user_threshold: {threshold}")
