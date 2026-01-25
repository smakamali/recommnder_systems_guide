# Hybrid Recommendation Systems

This directory contains documentation and implementation guidance for hybrid recommendation approaches that combine **Content-Based Filtering** with **Collaborative Filtering** (including Matrix Factorization) for the MovieLens 100K dataset.

## Overview

Hybrid recommendation systems combine multiple recommendation approaches to leverage their complementary strengths. For MovieLens 100K, combining content-based filtering (using movie genres and metadata) with collaborative filtering (matrix factorization) addresses the limitations of each approach individually.

### Why Hybrid for MovieLens?

**Content-Based Filtering Strengths:**
- ✅ Solves **item cold start** (new movies can be recommended immediately)
- ✅ **Explainable** recommendations ("Because you liked Sci-Fi movies...")
- ✅ Works for **niche users** with unique tastes
- ✅ Not affected by rating sparsity

**Collaborative Filtering Strengths:**
- ✅ Better **accuracy** through collaborative signals
- ✅ Discovers **cross-genre patterns** (e.g., Sci-Fi fans also like certain Thrillers)
- ✅ Handles **user preferences** beyond just genres
- ✅ Proven performance on MovieLens

**Hybrid Approach Benefits:**
- 🎯 **Best of both worlds**: Accuracy + Explainability + Cold Start handling
- 🎯 **Improved diversity**: Content-based can break filter bubbles
- 🎯 **Robust performance**: Works across different scenarios

For detailed theory, see: [`../README.md`](../README.md) Section 2.4 - Hybrid Methods (lines 420-453)

## MovieLens 100K Features

The MovieLens 100K dataset provides the following features for content-based filtering:

### Movie Metadata
- **19 Genre Categories**: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western, Unknown
- **Movie Titles**: Include release year (e.g., "Toy Story (1995)")
- **Binary Genre Vectors**: Each movie represented as 19-dimensional binary vector

### User Metadata
- **Demographics**: Age, gender, occupation, zip code
- **Ratings**: Explicit 1-5 star ratings

### Dataset Statistics
- 100,000 ratings from 943 users on 1,682 movies
- Rating scale: 1-5 (integer ratings)
- Sparsity: ~93.7% (very sparse)

## Hybridization Strategies

### 1. Weighted Hybrid

Combines scores from both methods with a weighting parameter:

```math
\text{score}(u,i) = \alpha \cdot \text{score}_{\mathrm{CF}}(u,i) + (1-\alpha) \cdot \text{score}_{\mathrm{CB}}(u,i)
```

**Implementation:**
```python
def weighted_hybrid_recommend(user_id, item_id, alpha=0.7):
    """
    Weighted combination of CF and content-based scores.
    
    Args:
        user_id: User identifier
        item_id: Item identifier
        alpha: Weight for CF (0-1), (1-alpha) for content-based
    
    Returns:
        Combined recommendation score
    """
    cf_score = collaborative_filtering_score(user_id, item_id)
    cb_score = content_based_score(user_id, item_id)
    
    return alpha * cf_score + (1 - alpha) * cb_score
```

**When to Use:**
- General-purpose recommendations
- Stable performance across scenarios
- Need to balance accuracy and explainability

**Tuning α:**
- `α = 0.7-0.8`: Favor CF (better accuracy, more collaborative signals)
- `α = 0.5`: Balanced approach
- `α = 0.2-0.3`: Favor content-based (better explainability, handles cold start)

### 2. Switching Hybrid

Uses different methods based on context:

```python
def switching_hybrid_recommend(user_id, item_id):
    """
    Switch between methods based on context.
    """
    # New items: use content-based
    if is_new_item(item_id):
        return content_based_score(user_id, item_id)
    
    # New users: use content-based or popular items
    elif is_new_user(user_id):
        return content_based_score(user_id, item_id)
    
    # Otherwise: use collaborative filtering
    else:
        return collaborative_filtering_score(user_id, item_id)
```

**When to Use:**
- Handling cold start problems
- Different strategies for different scenarios
- Need clear decision rules

**Decision Rules:**
- **New items** (< N ratings): Content-based
- **New users** (< M interactions): Content-based or popular items
- **Established items/users**: Collaborative filtering

### 3. Cascade Hybrid

Refines recommendations progressively:

```python
def cascade_hybrid_recommend(user_id, n=10):
    """
    Two-stage recommendation:
    1. CF generates candidate set
    2. Content-based re-ranks for diversity
    """
    # Stage 1: CF generates top candidates
    candidates = collaborative_filtering_top_n(user_id, n=50)
    
    # Stage 2: Content-based re-ranking with diversity
    final_recs = content_based_rerank(
        user_id, 
        candidates, 
        n=n,
        diversity_weight=0.3
    )
    
    return final_recs
```

**When to Use:**
- Large catalogs (efficient candidate generation)
- Need diversity in recommendations
- Two-stage architecture (candidate generation + ranking)

**Benefits:**
- CF quickly narrows to relevant candidates
- Content-based adds diversity and explainability
- Efficient for large-scale systems

### 4. Feature Combination

Combines features from different sources into a single model:

```python
def feature_combination_model(user_id, item_id):
    """
    Combine CF embeddings with content features.
    """
    # CF embeddings (from matrix factorization)
    user_embedding = cf_model.get_user_embedding(user_id)
    item_embedding = cf_model.get_item_embedding(item_id)
    
    # Content features
    genre_features = get_genre_features(item_id)
    user_genre_profile = get_user_genre_profile(user_id)
    
    # Concatenate features
    combined_features = np.concatenate([
        user_embedding,
        item_embedding,
        genre_features,
        user_genre_profile
    ])
    
    # Neural network or linear model
    score = neural_network(combined_features)
    return score
```

**When to Use:**
- Rich feature sets available
- Deep learning infrastructure
- Want end-to-end learning

**Architecture:**
- Input: User CF embedding + Item CF embedding + Genre features + User genre profile
- Hidden layers: Learn interactions between features
- Output: Recommendation score

### 5. Meta-Level Hybrid

Uses output of one method as input to another:

```python
def meta_level_hybrid(user_id):
    """
    Content-based builds user profile,
    CF uses these profiles for recommendations.
    """
    # Step 1: Build content-based user profile
    user_genre_profile = build_content_profile(user_id)
    
    # Step 2: Find similar users based on content profile
    similar_users = find_similar_users_by_profile(user_genre_profile)
    
    # Step 3: Use CF with content-informed neighborhoods
    recommendations = collaborative_filtering_with_neighborhood(
        user_id, 
        similar_users
    )
    
    return recommendations
```

**When to Use:**
- Transfer learning scenarios
- Cross-domain recommendations
- Need content-informed similarity

## Implementation Guide

### Step 1: Content-Based Component

#### Feature Extraction

```python
def extract_movie_features(movie_id):
    """
    Extract features for a movie.
    
    Returns:
        numpy array: Feature vector (genres + year features)
    """
    # Load movie metadata
    movie_data = load_movie_metadata(movie_id)
    
    # Genre binary vector (19 dimensions)
    genre_vector = movie_data['genres']  # [0, 1, 0, 1, ...]
    
    # Extract year from title
    year = extract_year(movie_data['title'])
    decade = (year // 10) * 10  # 1995 -> 1990
    
    # One-hot encode decade (optional)
    decade_vector = one_hot_encode_decade(decade)
    
    # Combine features
    features = np.concatenate([genre_vector, decade_vector])
    
    return features
```

#### User Profile Construction

```python
def build_user_profile(user_id, rating_threshold=4.0):
    """
    Build user profile from rated movies.
    
    Args:
        user_id: User identifier
        rating_threshold: Minimum rating to consider as "liked"
    
    Returns:
        numpy array: User profile vector
    """
    # Get user's ratings
    user_ratings = get_user_ratings(user_id)
    
    # Filter to liked movies (above threshold)
    liked_movies = [
        (movie_id, rating) 
        for movie_id, rating in user_ratings 
        if rating >= rating_threshold
    ]
    
    if len(liked_movies) == 0:
        # Fallback: use all ratings
        liked_movies = user_ratings
    
    # Weight features by ratings
    profile = np.zeros(NUM_FEATURES)
    total_weight = 0
    
    for movie_id, rating in liked_movies:
        movie_features = extract_movie_features(movie_id)
        profile += rating * movie_features
        total_weight += rating
    
    # Normalize
    if total_weight > 0:
        profile = profile / total_weight
    
    return profile
```

#### Similarity Computation

```python
def content_based_score(user_id, item_id):
    """
    Compute content-based recommendation score.
    
    Returns:
        float: Similarity score between user profile and item features
    """
    user_profile = build_user_profile(user_id)
    item_features = extract_movie_features(item_id)
    
    # Cosine similarity
    score = cosine_similarity(user_profile, item_features)
    
    return score
```

### Step 2: Collaborative Filtering Component

Use existing matrix factorization implementation from `../matrix_factorization/`:

```python
from matrix_factorization.mf_als import train_als_model
from matrix_factorization.data_loader import load_movielens_100k, get_train_test_split

# Load data and train CF model
data = load_movielens_100k()
trainset, testset = get_train_test_split(data)
cf_model = train_als_model(trainset, n_factors=50, reg=0.1, n_iter=50)

def collaborative_filtering_score(user_id, item_id):
    """Get CF prediction score."""
    prediction = cf_model.predict(user_id, item_id)
    return prediction.est
```

### Step 3: Hybrid Combination

Choose and implement one of the hybridization strategies above.

## Evaluation

### Metrics

Use the same evaluation metrics as matrix factorization:

**Rating Prediction:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

**Ranking:**
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K

See: [`../matrix_factorization/evaluation.py`](../matrix_factorization/evaluation.py)

### Comparing Approaches

Compare hybrid approach against:
1. **Pure Content-Based**: Baseline for explainability
2. **Pure Collaborative Filtering**: Baseline for accuracy
3. **Hybrid (Weighted)**: Vary α to find optimal balance
4. **Hybrid (Switching)**: Context-aware recommendations

### Expected Results

**Typical Performance (MovieLens 100K):**

| Method | RMSE | MAE | Precision@10 | Recall@10 | Explainability |
|--------|------|-----|--------------|-----------|----------------|
| Pure CF (MF) | ~0.92-0.96 | ~0.72-0.75 | ~0.35-0.40 | ~0.10-0.15 | Low |
| Pure Content-Based | ~1.10-1.20 | ~0.85-0.95 | ~0.25-0.30 | ~0.08-0.12 | High |
| Hybrid (α=0.7) | ~0.93-0.97 | ~0.73-0.76 | ~0.36-0.41 | ~0.11-0.16 | Medium |
| Hybrid (α=0.5) | ~0.95-0.99 | ~0.75-0.78 | ~0.33-0.38 | ~0.10-0.15 | Medium-High |

*Note: Results vary with hyperparameters and random seed*

**Trade-offs:**
- Higher α (more CF): Better accuracy, less explainability
- Lower α (more content-based): Better explainability, handles cold start, slightly lower accuracy

## Project Structure

```
hybrid_recommendation/
├── README.md                   # This file
├── content_based.py           # Content-based filtering implementation
├── hybrid_models.py           # Hybrid combination strategies
├── feature_extraction.py      # Movie feature extraction
├── user_profiles.py           # User profile construction
├── main.py                    # Complete pipeline script
├── evaluation.py              # Evaluation metrics (can reuse from matrix_factorization)
└── results/                   # Output directory for results
    └── .gitkeep
```

## Usage Example

### Quick Start: Weighted Hybrid

```python
from content_based import ContentBasedRecommender
from matrix_factorization.mf_als import train_als_model
from hybrid_models import WeightedHybrid

# Load data
data = load_movielens_100k()
trainset, testset = get_train_test_split(data)

# Train CF model
cf_model = train_als_model(trainset, n_factors=50, reg=0.1, n_iter=50)

# Initialize content-based
cb_model = ContentBasedRecommender(trainset)

# Create hybrid model
hybrid = WeightedHybrid(cf_model, cb_model, alpha=0.7)

# Generate recommendations
recommendations = hybrid.recommend(user_id="123", n=10)

# Evaluate
results = evaluate_model(hybrid, testset)
```

### Switching Hybrid for Cold Start

```python
from hybrid_models import SwitchingHybrid

hybrid = SwitchingHybrid(
    cf_model=cf_model,
    cb_model=cb_model,
    new_item_threshold=5,  # Items with < 5 ratings are "new"
    new_user_threshold=10   # Users with < 10 ratings are "new"
)

recommendations = hybrid.recommend(user_id="123", n=10)
```

## Hyperparameter Tuning

### Weighted Hybrid (α)

**Grid Search:**
```python
alphas = [0.3, 0.5, 0.7, 0.8, 0.9]
best_alpha = None
best_rmse = float('inf')

for alpha in alphas:
    hybrid = WeightedHybrid(cf_model, cb_model, alpha=alpha)
    rmse = evaluate(hybrid, testset)['rmse']
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print(f"Best alpha: {best_alpha}, RMSE: {best_rmse}")
```

### Content-Based Parameters

- **Rating Threshold**: What counts as "liked" (3.0, 4.0, or 4.5)
- **Feature Weighting**: Equal vs. rating-weighted
- **Similarity Metric**: Cosine, Jaccard, or Euclidean

### CF Parameters

See: [`../matrix_factorization/README.md`](../matrix_factorization/README.md) - Hyperparameters section

## Advantages of Hybrid Approach

### 1. Cold Start Handling
- **New Items**: Content-based can recommend immediately using genres
- **New Users**: Can use content-based or demographic-based recommendations

### 2. Explainability
- Content-based provides clear explanations ("Because you liked Sci-Fi...")
- Hybrid maintains some explainability while improving accuracy

### 3. Diversity
- Content-based can break filter bubbles
- Cascade hybrid explicitly adds diversity through re-ranking

### 4. Robustness
- Works across different scenarios (cold start, established users/items)
- Switching hybrid adapts to context

### 5. Accuracy
- Combines collaborative signals with content features
- Typically outperforms pure content-based
- Can match or exceed pure CF with proper tuning

## Limitations

### 1. Complexity
- More complex than single-method approaches
- Requires tuning multiple components

### 2. Feature Dependency
- Content-based quality depends on available features
- MovieLens has limited features (only genres + year)

### 3. Computational Cost
- Running both methods increases computation
- Can be mitigated with caching and efficient implementations

### 4. Tuning Overhead
- Need to tune α (weighted) or thresholds (switching)
- More hyperparameters to optimize

## Best Practices

### 1. Start Simple
- Begin with weighted hybrid (easiest to implement)
- Tune α on validation set

### 2. Evaluate Both Metrics
- Don't just optimize accuracy
- Consider explainability, diversity, and cold start performance

### 3. Use Appropriate Strategy
- **Weighted**: General-purpose, stable
- **Switching**: When cold start is critical
- **Cascade**: For large catalogs, need diversity
- **Feature Combination**: When you have rich features and DL infrastructure

### 4. Cache Expensive Operations
- User profiles (content-based)
- CF predictions
- Feature vectors

### 5. A/B Test
- Compare hybrid against pure methods
- Measure online metrics (CTR, engagement)

## Next Steps

1. **Implement Content-Based Component**: Extract features, build user profiles
2. **Integrate with CF**: Use existing matrix factorization code
3. **Implement Hybrid Strategies**: Start with weighted, then try others
4. **Evaluate**: Compare against baselines
5. **Tune Hyperparameters**: Find optimal α or thresholds
6. **Extend**: Add more features (directors, actors) if available

## References

- **Guide Section 2.2**: Content-Based Filtering (lines 278-329)
- **Guide Section 2.3**: Matrix Factorization (lines 330-419)
- **Guide Section 2.4**: Hybrid Methods (lines 420-453)
- **Guide Section 5.1**: Cold Start Problem (lines 1252-1316)

Main guide: [`../README.md`](../README.md)

## Troubleshooting

### Poor Content-Based Performance

- **Check feature quality**: Are genres correctly extracted?
- **Adjust rating threshold**: Try different values (3.0, 4.0, 4.5)
- **Normalize features**: Ensure feature vectors are normalized
- **Add more features**: Extract year/decade, use TF-IDF weighting

### Hybrid Not Improving Over CF

- **Tune α**: Try different weights (0.3, 0.5, 0.7, 0.9)
- **Check content-based quality**: Ensure it's working correctly
- **Consider switching strategy**: Maybe weighted isn't optimal
- **Evaluate on cold start**: Hybrid should excel here

### Cold Start Still Poor

- **Use switching hybrid**: Explicitly handle new users/items
- **Add demographic features**: Use user age, gender, occupation
- **Popular items fallback**: Recommend popular items for new users
- **Ask for preferences**: Explicitly ask new users for genre preferences

---

**Note**: This is a learning implementation following the comprehensive guide. For production systems, consider additional optimizations, distributed training, and real-time serving infrastructure.
