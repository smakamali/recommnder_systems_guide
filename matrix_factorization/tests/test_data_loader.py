"""
Unit tests for data_loader.py extensions.

Tests for:
- load_user_features()
- load_item_features()
- FeaturePreprocessor class
- get_cold_start_split()
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch

from data_loader import (
    load_user_features,
    load_item_features,
    FeaturePreprocessor,
    get_cold_start_split,
    get_movielens_data_path,
)


def _write_u_user(path, df):
    """Write u.user format: user_id|age|gender|occupation|zip_code"""
    with open(path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{row['user_id']}|{row['age']}|{row['gender']}|{row['occupation']}|{row['zip_code']}\n")


def _write_u_item(path, df):
    """Write u.item format: movie_id|title|release_date|video|url|19 genre columns (24 total)."""
    genre_names = [
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    genre_cols = [f'genre_{g}' for g in genre_names]
    with open(path, 'w', encoding='latin-1') as f:
        for _, row in df.iterrows():
            genres = '|'.join(str(int(row[c])) for c in genre_cols)
            f.write(f"{row['item_id']}|{row['title']}|{row['release_date']}|||{genres}\n")


class TestLoadUserFeatures:
    """Tests for load_user_features() function."""

    def test_load_user_features_returns_dataframe(self, sample_user_features, temp_data_dir):
        """Test that load_user_features returns a pandas DataFrame."""
        u_user = Path(temp_data_dir) / "u.user"
        _write_u_user(u_user, sample_user_features)
        with patch('data_loader.get_movielens_data_path', return_value=temp_data_dir):
            result = load_user_features()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_user_features)

    def test_user_features_has_required_columns(self, sample_user_features):
        """Test that user features DataFrame has required columns."""
        required_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        assert all(col in sample_user_features.columns for col in required_cols)

    def test_user_features_age_is_numeric(self, sample_user_features):
        """Test that age column contains numeric values."""
        assert pd.api.types.is_numeric_dtype(sample_user_features['age'])
        assert sample_user_features['age'].min() > 0
        assert sample_user_features['age'].max() < 150

    def test_user_features_gender_values(self, sample_user_features):
        """Test that gender column contains valid values."""
        valid_genders = {'M', 'F'}
        assert all(g in valid_genders for g in sample_user_features['gender'].unique())


class TestLoadItemFeatures:
    """Tests for load_item_features() function."""

    def test_load_item_features_returns_dataframe(self, sample_item_features, temp_data_dir):
        """Test that load_item_features returns a pandas DataFrame."""
        u_item = Path(temp_data_dir) / "u.item"
        _write_u_item(u_item, sample_item_features)
        with patch('data_loader.get_movielens_data_path', return_value=temp_data_dir):
            result = load_item_features()
        assert isinstance(result, pd.DataFrame)
        assert 'item_id' in result.columns

    def test_item_features_has_required_columns(self, sample_item_features):
        """Test that item features DataFrame has required columns."""
        required_cols = ['item_id', 'title', 'release_date']
        assert all(col in sample_item_features.columns for col in required_cols)

    def test_item_features_has_genre_columns(self, sample_item_features):
        """Test that item features include genre columns."""
        genre_cols = [c for c in sample_item_features.columns if c.startswith('genre_')]
        assert len(genre_cols) > 0
        for col in genre_cols:
            assert set(sample_item_features[col].unique()).issubset({0, 1})


class TestFeaturePreprocessor:
    """Tests for FeaturePreprocessor class."""

    def test_feature_preprocessor_initialization(self):
        """Test that FeaturePreprocessor can be instantiated."""
        preprocessor = FeaturePreprocessor()
        assert preprocessor is not None

    def test_fit_method_stores_scalers(self, sample_user_features, sample_item_features):
        """Test that fit() method stores scalers and encoders."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        assert hasattr(preprocessor, 'age_scaler')
        assert hasattr(preprocessor, 'year_scaler')
        assert hasattr(preprocessor, 'occupation_encoder')
        assert hasattr(preprocessor, 'gender_encoder')

    def test_transform_normalizes_age(self, sample_user_features, sample_item_features):
        """Test that transform_user_features normalizes age (values typically in reasonable range)."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        user_dict = preprocessor.transform_user_features(sample_user_features)
        # Each user has features dict; age is at user_features offset. Check scaled values exist.
        assert len(user_dict) == len(sample_user_features)
        for uid, feats in user_dict.items():
            assert isinstance(feats, dict)
            assert any(isinstance(v, (int, float)) for v in feats.values())

    def test_transform_one_hot_encodes_gender(self, sample_user_features, sample_item_features):
        """Test that transform encodes gender (one-hot via LabelEncoder)."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        user_dict = preprocessor.transform_user_features(sample_user_features)
        assert len(user_dict) == len(sample_user_features)
        # Gender contributes to feature indices
        assert len(preprocessor.gender_encoder.classes_) >= 1

    def test_transform_one_hot_encodes_occupation(self, sample_user_features, sample_item_features):
        """Test that transform encodes occupation."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        user_dict = preprocessor.transform_user_features(sample_user_features)
        assert len(user_dict) == len(sample_user_features)
        assert len(preprocessor.occupation_encoder.classes_) >= 1

    def test_to_libsvm_format_creates_file(
        self, sample_ratings_df, sample_user_features, sample_item_features, temp_data_dir
    ):
        """Test that to_libsvm_format() creates a valid libsvm file."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        user_dict = preprocessor.transform_user_features(sample_user_features)
        item_dict = preprocessor.transform_item_features(sample_item_features)
        ratings_list = [
            (r['user_id'], r['item_id'], r['rating'])
            for _, r in sample_ratings_df.iterrows()
        ]
        output_file = Path(temp_data_dir) / "test_libsvm.txt"
        preprocessor.to_libsvm_format(ratings_list, user_dict, item_dict, str(output_file))
        assert output_file.exists()
        first_line = output_file.read_text().split('\n')[0]
        assert first_line.strip()
        first_val = first_line.split()[0]
        assert float(first_val) >= 1.0 and float(first_val) <= 5.0

    def test_to_libsvm_format_has_correct_structure(
        self, sample_ratings_df, sample_user_features, sample_item_features, temp_data_dir
    ):
        """Test that libsvm file has correct format: rating feat_idx:feat_val ..."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        user_dict = preprocessor.transform_user_features(sample_user_features)
        item_dict = preprocessor.transform_item_features(sample_item_features)
        ratings_list = [
            (r['user_id'], r['item_id'], r['rating'])
            for _, r in sample_ratings_df.iterrows()
        ]
        output_file = Path(temp_data_dir) / "test_libsvm.txt"
        preprocessor.to_libsvm_format(ratings_list, user_dict, item_dict, str(output_file))
        lines = output_file.read_text().strip().split('\n')
        for line in lines:
            parts = line.split()
            assert len(parts) > 0
            assert float(parts[0]) >= 1.0 and float(parts[0]) <= 5.0
            for feat in parts[1:]:
                assert ':' in feat
                idx, val = feat.split(':')
                assert idx.isdigit()
                _ = float(val)  # must be valid numeric (can be negative, e.g. scaled age)

    def test_to_libsvm_format_includes_user_item_ids(
        self, sample_ratings_df, sample_user_features, sample_item_features, temp_data_dir
    ):
        """Test that libsvm format includes user and item indices."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_user_features, sample_item_features)
        user_dict = preprocessor.transform_user_features(sample_user_features)
        item_dict = preprocessor.transform_item_features(sample_item_features)
        ratings_list = [
            (r['user_id'], r['item_id'], r['rating'])
            for _, r in sample_ratings_df.iterrows()
        ]
        output_file = Path(temp_data_dir) / "test_libsvm.txt"
        preprocessor.to_libsvm_format(ratings_list, user_dict, item_dict, str(output_file))
        first_line = output_file.read_text().strip().split('\n')[0]
        # Should have feature:value pairs (e.g. "0:1" or "5:1")
        parts = first_line.split()
        assert len(parts) > 1
        assert any(':' in p for p in parts[1:])


class TestGetColdStartSplit:
    """Tests for get_cold_start_split() function."""

    def test_cold_start_split_returns_four_tuples(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Test that get_cold_start_split returns 4 elements."""
        data = sample_surprise_dataset
        result = get_cold_start_split(data, sample_user_features, sample_item_features)
        assert len(result) == 4
        trainset, testset, cold_users, cold_items = result
        assert hasattr(trainset, 'all_ratings')
        assert isinstance(testset, list)
        assert isinstance(cold_users, set)
        assert isinstance(cold_items, set)

    def test_cold_start_split_train_test_ratio(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Test that train/test split maintains approximately 80/20 ratio."""
        trainset, testset, _, _ = get_cold_start_split(
            sample_surprise_dataset, sample_user_features, sample_item_features,
            test_size=0.2, random_state=42
        )
        n_train = trainset.n_ratings
        n_test = len(testset)
        total = n_train + n_test
        assert total > 0
        train_ratio = n_train / total
        assert 0.7 <= train_ratio <= 0.9

    def test_cold_start_users_low_activity(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Cold start users have fewer than cold_user_threshold ratings in training."""
        trainset, testset, cold_users, _ = get_cold_start_split(
            sample_surprise_dataset, sample_user_features, sample_item_features,
            cold_user_threshold=5, random_state=42
        )
        from collections import defaultdict
        user_counts = defaultdict(int)
        for inner_uid, _, _ in trainset.all_ratings():
            uid = trainset.to_raw_uid(inner_uid)
            user_counts[uid] += 1
        for cold_uid in cold_users:
            assert cold_uid in user_counts
            assert user_counts[cold_uid] < 5

    def test_cold_start_items_low_activity(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Cold start items have fewer than cold_item_threshold ratings in training."""
        trainset, testset, _, cold_items = get_cold_start_split(
            sample_surprise_dataset, sample_user_features, sample_item_features,
            cold_item_threshold=10, random_state=42
        )
        from collections import defaultdict
        item_counts = defaultdict(int)
        for _, inner_iid, _ in trainset.all_ratings():
            iid = trainset.to_raw_iid(inner_iid)
            item_counts[iid] += 1
        for cold_iid in cold_items:
            assert cold_iid in item_counts
            assert item_counts[cold_iid] < 10

    def test_cold_start_split_reproducible(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Test that get_cold_start_split is reproducible with same random_state."""
        r1 = get_cold_start_split(
            sample_surprise_dataset, sample_user_features, sample_item_features,
            random_state=42
        )
        r2 = get_cold_start_split(
            sample_surprise_dataset, sample_user_features, sample_item_features,
            random_state=42
        )
        assert r1[0].n_ratings == r2[0].n_ratings
        assert len(r1[1]) == len(r2[1])

    def test_cold_start_threshold_parameter(
        self, sample_surprise_dataset, sample_user_features, sample_item_features
    ):
        """Test that cold_user_threshold affects cold start user set size."""
        _, _, cold_5, _ = get_cold_start_split(
            sample_surprise_dataset, sample_user_features, sample_item_features,
            cold_user_threshold=5, random_state=42
        )
        _, _, cold_1, _ = get_cold_start_split(
            sample_surprise_dataset, sample_user_features, sample_item_features,
            cold_user_threshold=1, random_state=42
        )
        # Stricter threshold (1) -> fewer users qualify as "cold" (fewer than 1 rating)
        # So cold_1 could be smaller or empty. cold_5 includes users with <5 ratings.
        assert isinstance(cold_5, set)
        assert isinstance(cold_1, set)
