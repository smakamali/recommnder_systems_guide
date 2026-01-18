"""
Data loading and preprocessing for MovieLens 100K dataset.

This module handles downloading and loading the MovieLens 100K dataset,
including train/test splitting and data preprocessing.
"""

import os
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


def load_movielens_100k(data_dir='data'):
    """
    Load MovieLens 100K dataset using Surprise library.
    
    The dataset is automatically downloaded if not present locally.
    
    Args:
        data_dir (str): Directory to store the dataset (default: 'data')
        
    Returns:
        Dataset: Surprise Dataset object containing the MovieLens 100K data
    """
    # Surprise has built-in support for MovieLens datasets
    # This will download automatically if needed
    data = Dataset.load_builtin('ml-100k')
    return data


def get_train_test_split(data, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    
    Args:
        data: Surprise Dataset object
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (trainset, testset) where testset is a list of tuples (uid, iid, rating)
    """
    trainset, testset = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state
    )
    return trainset, testset


def load_dataframe_from_builtin():
    """
    Load MovieLens 100K as pandas DataFrame for easier manipulation.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
    """
    # Get the path to the built-in dataset
    data = Dataset.load_builtin('ml-100k')
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data.raw_ratings, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    return df


def get_dataset_stats(data):
    """
    Print statistics about the dataset.
    
    Args:
        data: Surprise Dataset object
    """
    df = load_dataframe_from_builtin()
    
    print("=" * 60)
    print("MovieLens 100K Dataset Statistics")
    print("=" * 60)
    print(f"Total ratings: {len(df):,}")
    print(f"Number of users: {df['user_id'].nunique():,}")
    print(f"Number of items: {df['item_id'].nunique():,}")
    print(f"Rating scale: {df['rating'].min()} - {df['rating'].max()}")
    print(f"Sparsity: {(1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique())) * 100:.2f}%")
    print(f"Average ratings per user: {len(df) / df['user_id'].nunique():.2f}")
    print(f"Average ratings per item: {len(df) / df['item_id'].nunique():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    
    # Display statistics
    get_dataset_stats(data)
    
    # Split into train/test
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {trainset.n_ratings:,} ratings")
    print(f"Test set size: {len(testset):,} ratings")
    print(f"Number of users in training: {trainset.n_users}")
    print(f"Number of items in training: {trainset.n_items}")

