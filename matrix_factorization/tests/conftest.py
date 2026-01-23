"""
Pytest configuration and shared fixtures for unit tests.

This module provides common test fixtures and utilities used across
all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from surprise import Dataset, Trainset, Reader
from collections import defaultdict


@pytest.fixture
def sample_user_features():
    """Create sample user features DataFrame matching MovieLens 100k format."""
    return pd.DataFrame({
        'user_id': ['1', '2', '3', '4', '5'],
        'age': [24, 53, 23, 33, 42],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'occupation': ['technician', 'other', 'writer', 'other', 'executive'],
        'zip_code': ['85711', '94043', '32067', '15213', '98101']
    })


@pytest.fixture
def sample_item_features():
    """Create sample item features DataFrame matching MovieLens 100k format.
    Uses genre_* columns and release_year for FeaturePreprocessor compatibility."""
    genre_names = [
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    df = pd.DataFrame({
        'item_id': ['1', '2', '3', '4', '5'],
        'title': ['Toy Story', 'GoldenEye', 'Four Rooms', 'Get Shorty', 'Copycat'],
        'release_date': ['01-Jan-1995', '01-Jan-1995', '01-Jan-1995', '01-Jan-1995', '01-Jan-1995'],
        'video_release_date': ['', '', '', '', ''],
        'IMDb_URL': ['http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)'] * 5,
        'release_year': [1995, 1995, 1995, 1995, 1995],
    })
    for g in genre_names:
        col = f'genre_{g}'
        if g == 'unknown':
            df[col] = [0, 0, 0, 0, 0]
        elif g == 'Action':
            df[col] = [0, 1, 0, 1, 0]
        elif g == 'Adventure':
            df[col] = [1, 1, 0, 0, 0]
        elif g == 'Animation':
            df[col] = [1, 0, 0, 0, 0]
        elif g == 'Children':
            df[col] = [1, 0, 0, 0, 0]
        elif g == 'Comedy':
            df[col] = [1, 0, 1, 1, 0]
        elif g == 'Crime':
            df[col] = [0, 0, 0, 0, 1]
        elif g == 'Mystery':
            df[col] = [0, 0, 0, 0, 1]
        elif g == 'Sci-Fi':
            df[col] = [0, 1, 0, 0, 0]
        elif g == 'Thriller':
            df[col] = [0, 1, 0, 1, 1]
        elif g == 'Drama':
            df[col] = [0, 0, 0, 0, 1]
        else:
            df[col] = [0, 0, 0, 0, 0]
    return df


@pytest.fixture
def sample_ratings_df():
    """Create sample ratings DataFrame."""
    return pd.DataFrame({
        'user_id': ['1', '1', '2', '2', '3', '3', '4', '4', '5', '5'],
        'item_id': ['1', '2', '1', '3', '2', '4', '1', '5', '3', '4'],
        'rating': [5.0, 4.0, 3.0, 5.0, 4.0, 2.0, 5.0, 4.0, 3.0, 4.0],
        'timestamp': [881250949, 881250949, 881250949, 881250949, 881250949,
                      881250949, 881250949, 881250949, 881250949, 881250949]
    })


@pytest.fixture
def mock_trainset():
    """Create a mock Surprise Trainset object for testing."""
    class MockTrainset:
        def __init__(self):
            self.n_users = 5
            self.n_items = 5
            self.n_ratings = 10
            self.rating_scale = (1.0, 5.0)
            self.global_mean = 3.8
            
            # Create user-item rating mappings (10 ratings across 5 users, 5 items)
            # ur[inner_uid] = [(inner_iid, rating), ...], ir[inner_iid] = [(inner_uid, rating), ...]
            self.ur = defaultdict(list)
            self.ir = defaultdict(list)
            self.ur[0] = [(0, 5.0), (1, 4.0)]
            self.ur[1] = [(0, 3.0), (2, 5.0)]
            self.ur[2] = [(1, 4.0), (3, 2.0)]
            self.ur[3] = [(0, 5.0), (4, 4.0)]
            self.ur[4] = [(2, 3.0), (3, 4.0)]
            self.ir[0] = [(0, 5.0), (1, 3.0), (3, 5.0)]
            self.ir[1] = [(0, 4.0), (2, 4.0)]
            self.ir[2] = [(1, 5.0), (4, 3.0)]
            self.ir[3] = [(2, 2.0), (4, 4.0)]
            self.ir[4] = [(3, 4.0)]
            
            # User mappings
            self._raw2inner_id_users = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
            self._inner2raw_id_users = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}
            self._raw2inner_id_items = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
            self._inner2raw_id_items = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}
        
        def to_inner_uid(self, ruid):
            if ruid not in self._raw2inner_id_users:
                raise ValueError(f"User {ruid} not in trainset")
            return self._raw2inner_id_users[ruid]
        
        def to_raw_uid(self, iuid):
            return self._inner2raw_id_users[iuid]
        
        def to_inner_iid(self, riid):
            if riid not in self._raw2inner_id_items:
                raise ValueError(f"Item {riid} not in trainset")
            return self._raw2inner_id_items[riid]
        
        def to_raw_iid(self, iiid):
            return self._inner2raw_id_items[iiid]
        
        def knows_user(self, iuid):
            """Return True if user (inner id) is in trainset. Surprise SVD uses this."""
            return 0 <= iuid < self.n_users
        
        def knows_item(self, iiid):
            """Return True if item (inner id) is in trainset. Surprise SVD uses this."""
            return 0 <= iiid < self.n_items
        
        def all_ratings(self):
            """Return all ratings as (inner_uid, inner_iid, rating) tuples."""
            ratings = []
            for inner_uid, item_ratings in self.ur.items():
                for inner_iid, rating in item_ratings:
                    ratings.append((inner_uid, inner_iid, rating))
            return ratings
    
    return MockTrainset()


@pytest.fixture
def mock_testset():
    """Create a mock testset (list of tuples)."""
    return [
        ('1', '3', 4.0),
        ('2', '2', 3.0),
        ('3', '1', 5.0),
        ('4', '2', 4.0),
        ('5', '1', 3.0)
    ]


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Windows may lock files temporarily, so try cleanup with retry
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except (PermissionError, OSError):
        # On Windows, files may be locked briefly - ignore cleanup errors
        pass


@pytest.fixture
def sample_predictions():
    """Create sample predictions in Surprise format."""
    return [
        ('1', '3', 4.0, 3.8, {}),
        ('2', '2', 3.0, 3.5, {}),
        ('3', '1', 5.0, 4.2, {}),
        ('4', '2', 4.0, 3.9, {}),
        ('5', '1', 3.0, 3.2, {}),
        ('1', '4', 5.0, 4.5, {}),
        ('2', '5', 4.0, 3.7, {}),
        ('3', '3', 2.0, 2.8, {}),
    ]


@pytest.fixture
def cold_start_user_ids():
    """Set of user IDs that are cold start (not in training)."""
    return {'6', '7', '8'}


@pytest.fixture
def cold_start_item_ids():
    """Set of item IDs that are cold start (not in training)."""
    return {'6', '7', '8'}


@pytest.fixture
def sample_surprise_dataset(sample_ratings_df):
    """Create Surprise Dataset from sample_ratings_df for get_cold_start_split etc."""
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(
        sample_ratings_df[['user_id', 'item_id', 'rating']],
        reader
    )


@pytest.fixture
def sample_libsvm_line():
    """Sample libsvm format line for testing."""
    return "4.0 1:1 2:1 3:0.5 4:1 5:0 6:1 7:0.8"


@pytest.fixture
def sample_libsvm_file(temp_data_dir):
    """Create a sample libsvm format file."""
    filepath = Path(temp_data_dir) / "sample_libsvm.txt"
    content = """4.0 1:1 2:1 3:0.5 4:1
3.0 1:2 2:1 3:0.3 4:0
5.0 1:1 2:2 3:0.7 4:1
4.0 1:3 2:1 3:0.4 4:1
"""
    filepath.write_text(content)
    return str(filepath)
