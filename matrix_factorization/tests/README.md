# Unit Tests for Factorization Machines Implementation

This directory contains comprehensive unit tests for the Factorization Machines with feature support implementation as outlined in the implementation plan.

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Pytest fixtures and shared utilities
├── test_data_loader.py            # Tests for data_loader.py extensions
├── test_mf_fm.py                  # Tests for mf_fm.py (FM implementation)
├── test_mf_als_from_scratch.py     # Tests for mf_als_from_scratch.py
├── test_mf_als.py                 # Tests for mf_als.py (implicit library)
├── test_mf_svd.py                 # Tests for mf_svd.py
├── test_evaluation.py             # Tests for evaluation.py extensions
├── test_recommend.py              # Tests for recommend.py extensions
├── test_integration.py            # Integration tests
├── pytest.ini                     # Pytest configuration
├── requirements-test.txt          # Test dependencies
└── README.md                      # This file
```

## Test Coverage

### 1. Data Loader Tests (`test_data_loader.py`)

Tests for extended `data_loader.py` functionality:

- **`load_user_features()`**: User feature loading from MovieLens 100k
  - Returns DataFrame with required columns
  - Age is numeric and valid
  - Gender values are valid (M/F)
  
- **`load_item_features()`**: Item feature loading
  - Returns DataFrame with required columns
  - Genre columns are binary (0/1)
  - Release date parsing

- **`FeaturePreprocessor` class**: Feature preprocessing
  - Initialization
  - `fit()` stores scalers and encoders
  - `transform()` normalizes age to [0, 1]
  - `transform()` one-hot encodes gender and occupation
  - `to_libsvm_format()` creates valid libsvm files
  - Libsvm format structure validation
  - Includes user/item ID features

- **`get_cold_start_split()`**: Cold start data splitting
  - Returns 4 tuples (train_warm, test_warm, cold_users, cold_items)
  - Maintains 80/20 train/test ratio
  - Cold start users not in training set
  - Cold start items not in training set
  - Reproducibility with random_state
  - Cold start ratio parameter

### 2. ALS From Scratch Tests (`test_mf_als_from_scratch.py`)

Tests for `mf_als_from_scratch.py` implementation:

- **`als_matrix_factorization()` function**:
  - Returns P and Q matrices
  - Correct matrix shapes (k × n_users, k × n_items)
  - Random initialization
  - Reproducibility with same seed
  - Loss decreases over iterations
  - Different k values produce different sized matrices
  - Different regularization values affect results
  - Verbose output

- **`calculate_loss()` function**:
  - Returns float value
  - Loss increases with higher regularization
  - Includes prediction error component
  - Includes regularization component

- **`predict_rating()` function**:
  - Returns float in valid range [1, 5]
  - Handles cold start users (returns global mean)
  - Handles cold start items (returns global mean)
  - Uses dot product of factors

- **`ALSFromScratch` class**:
  - Initialization with parameters
  - `fit()` trains the model
  - `predict()` raises error if not trained
  - `predict()` works after training
  - Reproducibility
  - Verbose output

- **`train_als_from_scratch_model()` function**:
  - Returns trained model
  - Model can make predictions
  - Accepts custom parameters

### 3. ALS (Implicit Library) Tests (`test_mf_als.py`)

Tests for `mf_als.py` implementation:

- **`ALSMatrixFactorization` class**:
  - Initialization with parameters
  - `fit()` trains the model
  - `fit()` creates user/item mappings
  - `predict()` raises error if not trained
  - `predict()` returns float in range [1, 5]
  - Handles cold start users/items
  - `get_user_factors()` and `get_item_factors()`
  - Reproducibility with same seed
  - Different n_factors produce different dimensions

- **`train_als_model()` function**:
  - Returns trained model
  - Model can make predictions
  - Accepts custom parameters
  - Verbose output

### 4. SVD Tests (`test_mf_svd.py`)

Tests for `mf_svd.py` implementation:

- **`train_svd_model()` function**:
  - Returns trained SVD model
  - Model can make predictions
  - Accepts custom parameters
  - Verbose output
  - Reproducibility with same seed
  - `test()` method works correctly

- **`SVDMatrixFactorization` class**:
  - Initialization with parameters
  - `fit()` trains the model
  - `predict()` raises error if not trained
  - `predict()` works after training
  - `test()` method works correctly
  - Reproducibility
  - Different n_factors produce different models
  - Verbose output

### 5. Factorization Machine Tests (`test_mf_fm.py`)

Tests for `mf_fm.py` implementation:

- **`FactorizationMachineModel` class**:
  - Initialization with default and custom parameters
  - `train()` accepts libsvm format files
  - `train()` with validation set
  - `predict()` returns predictions
  - `predict_single()` for user-item pairs with features
  - `save_model()` and `load_model()` persistence
  - Handles missing features gracefully

- **`FMRecommender` class**:
  - Initialization
  - `fit()` trains the model
  - `predict()` returns rating predictions
  - `predict()` works for cold start users
  - `test()` returns Surprise-compatible format
  - Handles missing features

### 6. Evaluation Tests (`test_evaluation.py`)

Tests for extended `evaluation.py` functionality:

- **`evaluate_cold_start_users()`**:
  - Returns dictionary with RMSE, MAE, coverage
  - Correct RMSE calculation for cold users
  - Coverage calculation (percentage with predictions)
  - Handles empty cold start set
  - Verbose output

- **`evaluate_cold_start_items()`**:
  - Returns dictionary with metrics
  - Correct RMSE calculation for cold items
  - Coverage calculation

- **`evaluate_with_cold_start_breakdown()`**:
  - Returns dictionary with 4 scenarios
  - Warm-warm scenario (known user, known item)
  - Cold user scenario (new user, known item)
  - Cold item scenario (known user, new item)
  - Cold-cold scenario (new user, new item)
  - Counts match total predictions
  - Verbose output

- **`compare_with_without_features()`**:
  - Returns dictionary with improvements
  - RMSE improvement calculation
  - MAE improvement calculation
  - Breakdown by user activity level

- **Extended `evaluate_model()`**:
  - Accepts cold start parameters
  - `breakdown_by_scenario` flag
  - Backward compatibility (works without new params)

### 7. Recommendation Tests (`test_recommend.py`)

Tests for extended `recommend.py` functionality:

- **`generate_recommendations_with_features()`**:
  - Returns list of recommendations
  - Sorted by rating (descending)
  - Excludes rated items when requested
  - Includes rated items when requested
  - Works for cold start users
  - Respects n parameter
  - Handles missing features

- **Updated `handle_cold_start_user()`**:
  - Uses FM model when provided
  - Falls back to popularity when FM unavailable
  - FM takes priority over popularity
  - Warm users use normal model
  - Verbose output
  - Handles partial features

## Running Tests

### Install Dependencies

```bash
# Install pytest and test dependencies
pip install pytest pytest-cov pytest-mock
```

### Run All Tests

```bash
# From matrix_factorization directory
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test Files

```bash
# Test data loader only
pytest tests/test_data_loader.py

# Test ALS from scratch
pytest tests/test_mf_als_from_scratch.py

# Test ALS (implicit library)
pytest tests/test_mf_als.py

# Test SVD
pytest tests/test_mf_svd.py

# Test FM model only
pytest tests/test_mf_fm.py

# Test evaluation only
pytest tests/test_evaluation.py

# Test recommendations only
pytest tests/test_recommend.py

# Test integration
pytest tests/test_integration.py
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_data_loader.py::TestLoadUserFeatures

# Run specific test function
pytest tests/test_evaluation.py::TestEvaluateColdStartUsers::test_cold_start_users_rmse_calculation
```

### Run Tests with Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `sample_user_features`: Sample user features DataFrame
- `sample_item_features`: Sample item features DataFrame
- `sample_ratings_df`: Sample ratings DataFrame
- `mock_trainset`: Mock Surprise Trainset object
- `mock_testset`: Mock testset (list of tuples)
- `temp_data_dir`: Temporary directory for test files
- `sample_predictions`: Sample predictions in Surprise format
- `cold_start_user_ids`: Set of cold start user IDs
- `cold_start_item_ids`: Set of cold start item IDs
- `sample_libsvm_line`: Sample libsvm format line
- `sample_libsvm_file`: Sample libsvm format file

## Writing New Tests

When adding new functionality, follow these guidelines:

1. **Test Structure**: Use classes to group related tests
   ```python
   class TestNewFunction:
       def test_basic_functionality(self):
           # Test basic case
           pass
       
       def test_edge_cases(self):
           # Test edge cases
           pass
   ```

2. **Use Fixtures**: Reuse fixtures from `conftest.py` when possible

3. **Mock External Dependencies**: Mock file I/O, external libraries (xLearn), etc.

4. **Test Both Success and Failure Cases**: Include tests for error handling

5. **Use Descriptive Names**: Test function names should describe what they test

6. **Assert Clearly**: Use specific assertions with helpful error messages

## Notes

- Many tests are currently commented out with "When implemented" notes, as they test functionality that will be implemented according to the plan
- Tests use mocks for external dependencies (xLearn, file I/O) to ensure tests run quickly and don't require actual data files
- Some tests may need adjustment once the actual implementation is complete

## Continuous Integration

These tests should be integrated into CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pip install pytest pytest-cov
    pytest tests/ --cov=. --cov-report=xml
```
