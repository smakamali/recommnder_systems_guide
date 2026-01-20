# Extending Matrix Factorization with User and Item Features

## Table of Contents

1. [Introduction](#introduction)
2. [Why Incorporate Features?](#why-incorporate-features)
3. [Core Approaches](#core-approaches)
   - [1. Factorization Machines (FM)](#1-factorization-machines-fm)
   - [2. SVD++ Extensions](#2-svd-extensions)
   - [3. Regression-Based Feature Integration](#3-regression-based-feature-integration)
   - [4. Content-Boosted Collaborative Filtering](#4-content-boosted-collaborative-filtering)
   - [5. Hybrid Matrix Factorization](#5-hybrid-matrix-factorization)
   - [6. Neural Collaborative Filtering with Side Information](#6-neural-collaborative-filtering-with-side-information)
   - [7. LightFM Approach](#7-lightfm-approach)
4. [Comparison of Approaches](#comparison-of-approaches)
5. [Practical Considerations](#practical-considerations)
6. [Concrete Examples](#concrete-examples)
7. [References](#references)

---

## Introduction

Basic matrix factorization (MF) relies solely on user-item interaction data (ratings, clicks, purchases) to learn latent representations. While powerful, this approach has limitations:

- **Cold start problem**: Cannot recommend for new users or items without interaction history
- **Data sparsity**: Most users interact with only a small fraction of items
- **Limited context**: Ignores rich side information about users and items

This document explores **seven approaches** to extend matrix factorization by incorporating **user features** (e.g., age, gender, location, occupation) and **item features** (e.g., genre, release date, director, price) to address these limitations.

### Related Documentation

- **Core MF Theory**: See [../README.md](../README.md) lines 330-419 for matrix factorization fundamentals
- **Basic Implementation**: See [README.md](README.md) for ALS and SGD implementations
- **Hybrid Systems**: See [../hybrid_recommendation/README.md](../hybrid_recommendation/README.md) for content-based integration

---

## Why Incorporate Features?

### Benefits

| Benefit | Description | Impact |
|---------|-------------|--------|
| **Cold Start Mitigation** | New users/items can leverage features for immediate recommendations | Critical for platforms with high user/item churn |
| **Better Predictions** | Richer representations capture more nuanced preferences | Improves RMSE, NDCG by 10-30% in practice |
| **Interpretability** | Feature weights provide explanations (e.g., "recommended because you're in the 25-34 age group") | Builds user trust |
| **Data Efficiency** | Learn from fewer interactions when features are informative | Faster model convergence |
| **Personalization** | Demographic and contextual features enable finer-grained personalization | Handles diverse user populations better |

### Types of Features

**User Features:**
- **Demographic**: Age, gender, location, language, income bracket
- **Behavioral**: Activity level, session frequency, device type
- **Temporal**: Time of day, day of week, season
- **Social**: Friend count, network centrality, community membership

**Item Features:**
- **Content**: Genre, category, tags, keywords, description embeddings
- **Metadata**: Release date, creator, brand, price, duration
- **Statistical**: Popularity, average rating, view count, recency
- **Structural**: Hierarchical categories, knowledge graph relations

---

## Core Approaches

### 1. Factorization Machines (FM)

**Overview**: Factorization Machines generalize matrix factorization to model interactions between arbitrary features, not just user and item IDs.

#### Mathematical Formulation

The prediction function incorporates all features and their pairwise interactions:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

where:
- $\mathbf{x} \in \mathbb{R}^n$ is the feature vector (includes user ID, item ID, user features, item features)
- $w_0$ is the global bias
- $w_i$ is the weight for feature $i$ (linear term)
- $\mathbf{v}_i \in \mathbb{R}^k$ is the latent factor vector for feature $i$
- $\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{\ell=1}^k v_{i,\ell} v_{j,\ell}$ models feature interactions

**Key Insight**: The interaction term $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ allows the model to learn how features interact (e.g., "young males prefer action movies").

#### Feature Vector Construction

For a rating prediction scenario:

```python
# Example: User 42 (age=25, gender=M) rating Movie 100 (genre=Action, year=2020)
x = [
    # User ID (one-hot encoding)
    0, 0, ..., 1, ..., 0,  # Position 42 = 1, rest = 0
    # Item ID (one-hot encoding)
    0, 0, ..., 1, ..., 0,  # Position 100 = 1, rest = 0
    # User features
    25,        # age (normalized)
    1, 0,      # gender (M=1, F=0)
    # Item features
    1, 0, 0,   # genre (Action=1, Drama=0, Comedy=0)
    2020,      # year (normalized)
]
```

#### Training

The model parameters $\{w_0, \mathbf{w}, \mathbf{V}\}$ are learned by minimizing:

$$\mathcal{L} = \sum_{(u,i,r) \in \mathcal{D}} \left( r_{ui} - \hat{y}(\mathbf{x}_{ui}) \right)^2 + \lambda \left( \|\mathbf{w}\|^2 + \|\mathbf{V}\|_F^2 \right)$$

Optimization via SGD or ALS. Complexity: $O(k \cdot n)$ per prediction (linear in features).

#### Properties

| Aspect | Description |
|--------|-------------|
| **Complexity** | Moderate - $O(k \cdot n)$ inference |
| **Cold Start** | Excellent - new users/items with features work immediately |
| **Interpretability** | Moderate - feature weights + interaction terms |
| **Scalability** | Good - sparse feature vectors enable efficient computation |
| **Best For** | Click-through rate prediction, rich feature sets, sparse data |

#### Implementation

```python
# Using libFM or xLearn
from xlearn import FFMModel

# Create FM model
fm = FFMModel(
    task='reg',           # Regression task
    metric='rmse',
    lr=0.1,
    k=10,                 # Latent factor dimension
    reg_lambda=0.01
)

# Train (data in libsvm format with field information)
fm.fit(train_data, eval_data)

# Predict
predictions = fm.predict(test_data)
```

**Libraries**:
- `xLearn`: Fast C++ implementation with Python bindings
- `libFM`: Original implementation by Steffen Rendle
- `fastFM`: Python/Cython implementation

---

### 2. SVD++ Extensions

**Overview**: SVD++ extends basic matrix factorization by incorporating implicit feedback. This framework can be further extended to include explicit user and item features.

#### Original SVD (Baseline)

For comparison, the basic SVD with biases predicts ratings using only latent factors and bias terms:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{q}_i^T \mathbf{p}_u$$

where:
- $\mu$ is the global average rating
- $b_u$, $b_i$ are user and item biases
- $\mathbf{p}_u \in \mathbb{R}^k$ is the user latent factor vector
- $\mathbf{q}_i \in \mathbb{R}^k$ is the item latent factor vector

This baseline model captures user-item interactions through latent factors but doesn't account for implicit feedback signals.

#### Standard SVD++

The original SVD++ incorporates implicit feedback (items user has interacted with):

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{q}_i^T \left( \mathbf{p}_u + |I_u|^{-0.5} \sum_{j \in I_u} \mathbf{y}_j \right)$$

where:
- $\mu$ is the global average rating
- $b_u$, $b_i$ are user and item biases
- $\mathbf{p}_u$, $\mathbf{q}_i$ are latent factors
- $I_u$ is the set of items user $u$ has rated
- $\mathbf{y}_j$ captures implicit feedback from item $j$

#### Feature-Rich Extension

Extend SVD++ to incorporate explicit features:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{q}_i^T \left( \mathbf{p}_u + |I_u|^{-0.5} \sum_{j \in I_u} \mathbf{y}_j + \mathbf{W}_u \mathbf{f}_u + \mathbf{W}_i \mathbf{g}_i \right)$$

where:
- $\mathbf{f}_u \in \mathbb{R}^{d_u}$ is the user feature vector (age, gender, etc.)
- $\mathbf{g}_i \in \mathbb{R}^{d_i}$ is the item feature vector (genre, year, etc.)
- $\mathbf{W}_u \in \mathbb{R}^{k \times d_u}$ projects user features to latent space
- $\mathbf{W}_i \in \mathbb{R}^{k \times d_i}$ projects item features to latent space

**Alternative Formulation (Bias Terms)**:

A simpler variant adds features as bias corrections:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{q}_i^T \mathbf{p}_u + \boldsymbol{\alpha}^T \mathbf{f}_u + \boldsymbol{\beta}^T \mathbf{g}_i$$

where $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$ are learned weight vectors for features.

#### Properties

| Aspect | Description |
|--------|-------------|
| **Complexity** | Moderate - extends SVD++ complexity |
| **Cold Start** | Good - features help, but implicit feedback component requires history |
| **Interpretability** | High - clear separation of biases, latent factors, and feature effects |
| **Best For** | Systems with both explicit ratings and implicit feedback |

#### Implementation Considerations

- Learn $\mathbf{W}_u$, $\mathbf{W}_i$ jointly with other parameters via SGD
- Feature vectors should be normalized (mean 0, std 1 for continuous features)
- Categorical features: one-hot encode or learn embeddings

---

### 3. Regression-Based Feature Integration

**Overview**: The simplest approach - add feature-based terms directly to the standard MF prediction.

#### Mathematical Formulation

$$\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i + \boldsymbol{\alpha}^T \mathbf{f}_u + \boldsymbol{\beta}^T \mathbf{g}_i$$

or with biases:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i + \boldsymbol{\alpha}^T \mathbf{f}_u + \boldsymbol{\beta}^T \mathbf{g}_i$$

where:
- $\mathbf{p}_u^T \mathbf{q}_i$ is the standard MF component (captures collaborative signal)
- $\boldsymbol{\alpha}^T \mathbf{f}_u$ captures user feature effects
- $\boldsymbol{\beta}^T \mathbf{g}_i$ captures item feature effects
- $\boldsymbol{\alpha} \in \mathbb{R}^{d_u}$, $\boldsymbol{\beta} \in \mathbb{R}^{d_i}$ are learned weight vectors

#### Optimization

Minimize the regularized loss:

$$\mathcal{L} = \sum_{(u,i) \in \mathcal{O}} \left( r_{ui} - \hat{r}_{ui} \right)^2 + \lambda \left( \|\mathbf{P}\|_F^2 + \|\mathbf{Q}\|_F^2 + \|\boldsymbol{\alpha}\|^2 + \|\boldsymbol{\beta}\|^2 \right)$$

**SGD Update Rules**:

$$\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta \left( e_{ui} \mathbf{q}_i - \lambda \mathbf{p}_u \right)$$

$$\mathbf{q}_i \leftarrow \mathbf{q}_i + \eta \left( e_{ui} \mathbf{p}_u - \lambda \mathbf{q}_i \right)$$

$$\boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha} + \eta \left( e_{ui} \mathbf{f}_u - \lambda \boldsymbol{\alpha} \right)$$

$$\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} + \eta \left( e_{ui} \mathbf{g}_i - \lambda \boldsymbol{\beta} \right)$$

where $e_{ui} = r_{ui} - \hat{r}_{ui}$ is the prediction error.

#### Properties

| Aspect | Description |
|--------|-------------|
| **Complexity** | Low - minimal overhead over basic MF |
| **Cold Start** | Good - new users/items can use feature terms |
| **Interpretability** | High - clear additive effects |
| **Best For** | Small feature sets, simple linear relationships |

#### Implementation

```python
class LinearFeatureMF:
    def __init__(self, n_factors, n_user_features, n_item_features):
        self.P = np.random.randn(n_users, n_factors) * 0.01
        self.Q = np.random.randn(n_items, n_factors) * 0.01
        self.alpha = np.zeros(n_user_features)  # User feature weights
        self.beta = np.zeros(n_item_features)   # Item feature weights
        self.global_mean = 0
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
    
    def predict(self, user_id, item_id, user_features, item_features):
        mf_term = self.P[user_id] @ self.Q[item_id]
        user_feature_term = self.alpha @ user_features
        item_feature_term = self.beta @ item_features
        
        prediction = (self.global_mean + 
                     self.user_bias[user_id] + 
                     self.item_bias[item_id] +
                     mf_term + 
                     user_feature_term + 
                     item_feature_term)
        return prediction
```

---

### 4. Content-Boosted Collaborative Filtering

**Overview**: A two-stage approach that uses content features to augment the sparse rating matrix before applying collaborative filtering.

#### Algorithm

**Stage 1: Content-Based Prediction**

For each missing entry $(u, i)$ in the rating matrix:
1. Build a content-based predictor using available features
2. Predict pseudo-rating: $\tilde{r}_{ui} = f_{\text{content}}(\mathbf{f}_u, \mathbf{g}_i)$

Common approaches for $f_{\text{content}}$:
- Linear regression on feature similarity
- Nearest neighbor based on feature distance
- Simple neural network

**Stage 2: Matrix Factorization**

Create augmented matrix $\tilde{\mathbf{R}}$:

$$\tilde{r}_{ui} = \begin{cases}
r_{ui} & \text{if rating exists} \\
\tilde{r}_{ui}^{\text{content}} & \text{if rating missing}
\end{cases}$$

Apply standard MF on $\tilde{\mathbf{R}}$:

$$\min_{\mathbf{P}, \mathbf{Q}} \sum_{u,i} w_{ui} \left( \tilde{r}_{ui} - \mathbf{p}_u^T \mathbf{q}_i \right)^2 + \lambda \left( \|\mathbf{P}\|_F^2 + \|\mathbf{Q}\|_F^2 \right)$$

where $w_{ui} = 1$ for observed ratings and $w_{ui} = w_0 < 1$ for pseudo-ratings.

#### Weighting Strategy

Critical to balance observed vs pseudo-ratings:
- $w_0 = 0.1$ to $0.5$: typical range for pseudo-rating weights
- Too high: pseudo-ratings dominate, loses collaborative signal
- Too low: features underutilized

#### Properties

| Aspect | Description |
|--------|-------------|
| **Complexity** | Moderate - two separate training phases |
| **Cold Start** | Good - pseudo-ratings provide initial estimates |
| **Interpretability** | High - separates content and collaborative components |
| **Best For** | Very sparse data, strong content features available |

#### Advantages and Limitations

**Advantages**:
- Alleviates sparsity by filling in unobserved entries
- Separates content and collaborative modeling
- Can use any content-based predictor in stage 1

**Limitations**:
- Two-stage process more complex than end-to-end
- Quality depends heavily on content predictor
- Pseudo-ratings can introduce noise

---

### 5. Hybrid Matrix Factorization

**Overview**: Learn separate latent representations for collaborative and content signals, then combine them.

#### Mathematical Formulation

**User representation** combines collaborative and feature-based components:

$$\mathbf{p}_u^{\text{total}} = \mathbf{p}_u^{\text{CF}} + \mathbf{W}_u \mathbf{f}_u$$

**Item representation**:

$$\mathbf{q}_i^{\text{total}} = \mathbf{q}_i^{\text{CF}} + \mathbf{W}_i \mathbf{g}_i$$

**Prediction**:

$$\hat{r}_{ui} = \left( \mathbf{p}_u^{\text{CF}} + \mathbf{W}_u \mathbf{f}_u \right)^T \left( \mathbf{q}_i^{\text{CF}} + \mathbf{W}_i \mathbf{g}_i \right)$$

Expanding:

$$\hat{r}_{ui} = \underbrace{\mathbf{p}_u^{\text{CF} \, T} \mathbf{q}_i^{\text{CF}}}_{\text{CF term}} + \underbrace{\mathbf{p}_u^{\text{CF} \, T} \mathbf{W}_i \mathbf{g}_i + \mathbf{q}_i^{\text{CF} \, T} \mathbf{W}_u \mathbf{f}_u}_{\text{Cross terms}} + \underbrace{\mathbf{f}_u^T \mathbf{W}_u^T \mathbf{W}_i \mathbf{g}_i}_{\text{Content term}}$$

This formulation captures:
1. **Pure collaborative filtering** (first term)
2. **CF-content interaction** (middle terms)
3. **Pure content-based** (last term)

#### Alternative: Weighted Combination

Simpler variant with explicit weighting:

$$\hat{r}_{ui} = \alpha \left( \mathbf{p}_u^{\text{CF} \, T} \mathbf{q}_i^{\text{CF}} \right) + (1-\alpha) \left( \mathbf{f}_u^T \mathbf{W}^T \mathbf{g}_i \right)$$

where $\alpha \in [0,1]$ balances CF and content. Can be:
- Fixed hyperparameter
- Learned globally
- Learned per-user or per-item

#### Training

Joint optimization of all parameters:

$$\min_{\mathbf{P}^{\text{CF}}, \mathbf{Q}^{\text{CF}}, \mathbf{W}_u, \mathbf{W}_i} \sum_{(u,i) \in \mathcal{O}} \left( r_{ui} - \hat{r}_{ui} \right)^2 + \lambda \left( \|\mathbf{P}^{\text{CF}}\|_F^2 + \|\mathbf{Q}^{\text{CF}}\|_F^2 + \|\mathbf{W}_u\|_F^2 + \|\mathbf{W}_i\|_F^2 \right)$$

#### Properties

| Aspect | Description |
|--------|-------------|
| **Complexity** | Moderate - more parameters than basic MF |
| **Cold Start** | Good - feature projections handle new entities |
| **Interpretability** | Moderate - can examine CF vs content contributions |
| **Best For** | Rich features, need to balance CF and content signals |

---

### 6. Neural Collaborative Filtering with Side Information

**Overview**: Use neural networks to flexibly combine user/item IDs with features, learning non-linear interactions.

#### Architecture

```
User ID ──→ User Embedding ──┐
User Features ──→ Dense Layer ─┴──→ Concatenate ──→ MLP ──→ Prediction
Item ID ──→ Item Embedding ──┐                      ↑
Item Features ──→ Dense Layer ─┴──→ Concatenate ────┘
```

#### Mathematical Formulation

**Embedding Layer**:
- $\mathbf{e}_u = \text{Embed}(\text{user\_id})$
- $\mathbf{e}_i = \text{Embed}(\text{item\_id})$

**Feature Processing**:
- $\mathbf{h}_u^{\text{feat}} = \sigma(\mathbf{W}_u^{(1)} \mathbf{f}_u + \mathbf{b}_u^{(1)})$
- $\mathbf{h}_i^{\text{feat}} = \sigma(\mathbf{W}_i^{(1)} \mathbf{g}_i + \mathbf{b}_i^{(1)})$

**Concatenation**:
- $\mathbf{z}_u = [\mathbf{e}_u \,|\, \mathbf{h}_u^{\text{feat}}]$
- $\mathbf{z}_i = [\mathbf{e}_i \,|\, \mathbf{h}_i^{\text{feat}}]$
- $\mathbf{z} = [\mathbf{z}_u \,|\, \mathbf{z}_i]$

**Multi-Layer Perceptron**:

$$\mathbf{h}^{(1)} = \sigma(\mathbf{W}^{(1)} \mathbf{z} + \mathbf{b}^{(1)})$$

$$\mathbf{h}^{(2)} = \sigma(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$

$$\vdots$$

$$\hat{r}_{ui} = \mathbf{w}^{\text{out} \, T} \mathbf{h}^{(L)} + b^{\text{out}}$$

#### Implementation

```python
import torch
import torch.nn as nn

class NCFWithFeatures(nn.Module):
    def __init__(self, n_users, n_items, n_user_features, n_item_features,
                 embedding_dim=32, hidden_dims=[64, 32, 16]):
        super().__init__()
        
        # Embeddings for IDs
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Feature processors
        self.user_feature_net = nn.Linear(n_user_features, embedding_dim)
        self.item_feature_net = nn.Linear(n_item_features, embedding_dim)
        
        # MLP layers
        input_dim = 2 * 2 * embedding_dim  # (user_emb + user_feat + item_emb + item_feat)
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
    
    def forward(self, user_ids, item_ids, user_features, item_features):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Process features
        user_feat = torch.relu(self.user_feature_net(user_features))
        item_feat = torch.relu(self.item_feature_net(item_features))
        
        # Concatenate all
        x = torch.cat([user_emb, user_feat, item_emb, item_feat], dim=-1)
        
        # MLP
        x = self.mlp(x)
        rating = self.output(x)
        
        return rating.squeeze()
```

#### Training

```python
model = NCFWithFeatures(n_users, n_items, n_user_features, n_item_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(n_epochs):
    for batch in train_loader:
        user_ids, item_ids, user_feats, item_feats, ratings = batch
        
        predictions = model(user_ids, item_ids, user_feats, item_feats)
        loss = criterion(predictions, ratings)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Properties

| Aspect | Description |
|--------|-------------|
| **Complexity** | High - many parameters, requires careful tuning |
| **Cold Start** | Excellent - features enable predictions for new entities |
| **Interpretability** | Low - neural networks are black boxes |
| **Scalability** | Moderate - requires GPU for large-scale training |
| **Best For** | Complex non-linear patterns, rich features, sufficient data |

#### Advantages

- **Flexibility**: Can model arbitrary non-linear interactions
- **Automatic feature learning**: Learns useful representations
- **State-of-the-art performance**: Often achieves best accuracy
- **Extensible**: Easy to add attention, residual connections, etc.

---

### 7. LightFM Approach

**Overview**: LightFM represents users and items purely as collections of features, enabling a unified model for both cold-start and warm-start scenarios.

#### Core Idea

Instead of learning separate embeddings for each user/item ID:
- **Traditional MF**: $\mathbf{p}_u$, $\mathbf{q}_i$ directly represent user $u$ and item $i$
- **LightFM**: Build representations from features

#### Mathematical Formulation

**User representation** is the sum of embeddings for all user features:

$$\mathbf{p}_u = \sum_{j \in \mathcal{F}_u} \mathbf{e}_j$$

where $\mathcal{F}_u$ is the set of features describing user $u$. This includes:
- User ID itself (as a feature)
- Demographic features (age group, gender, location)
- Behavioral features (activity level, preferences)

**Item representation**:

$$\mathbf{q}_i = \sum_{k \in \mathcal{F}_i} \mathbf{e}_k$$

where $\mathcal{F}_i$ includes item ID and content features.

**Prediction**:

$$\hat{r}_{ui} = \sigma\left( \mathbf{p}_u^T \mathbf{q}_i \right) = \sigma\left( \sum_{j \in \mathcal{F}_u} \sum_{k \in \mathcal{F}_i} \mathbf{e}_j^T \mathbf{e}_k \right)$$

For regression tasks, can use identity instead of sigmoid $\sigma$.

#### Feature Representation

**Example**: Movie recommendation

**User features**:
```python
user_features = {
    'user_id:42',      # User ID as feature
    'age:25-34',       # Age bucket
    'gender:M',        # Gender
    'location:NYC',    # Location
    'premium:True'     # Account type
}
```

**Item features**:
```python
item_features = {
    'movie_id:100',    # Item ID as feature
    'genre:Action',    # Primary genre
    'genre:Sci-Fi',    # Secondary genre (multiple values OK)
    'year:2020',       # Release year
    'director:Nolan'   # Director
}
```

Each feature gets its own embedding vector $\mathbf{e}_j \in \mathbb{R}^k$.

#### Training Objective

For implicit feedback (BPR loss):

$$\mathcal{L} = -\sum_{(u,i,j) \in D_S} \log \sigma(\hat{r}_{ui} - \hat{r}_{uj}) + \lambda \sum_{\theta} \|\theta\|^2$$

where item $i$ is observed, $j$ is negative sample.

For explicit feedback (MSE):

$$\mathcal{L} = \sum_{(u,i,r) \in \mathcal{D}} (r_{ui} - \hat{r}_{ui})^2 + \lambda \sum_{\theta} \|\theta\|^2$$

#### Implementation

```python
from lightfm import LightFM
from lightfm.data import Dataset
import scipy.sparse as sp

# Build dataset
dataset = Dataset()
dataset.fit(
    users=user_ids,
    items=item_ids,
    user_features=all_user_features,  # All possible user feature values
    item_features=all_item_features   # All possible item feature values
)

# Build interaction matrix and feature matrices
(interactions, weights) = dataset.build_interactions(
    [(user_id, item_id, rating) for user_id, item_id, rating in ratings]
)

user_features_matrix = dataset.build_user_features(
    [(user_id, [feat1, feat2, ...]) for user_id, features in user_feature_map.items()]
)

item_features_matrix = dataset.build_item_features(
    [(item_id, [feat1, feat2, ...]) for item_id, features in item_feature_map.items()]
)

# Train model
model = LightFM(
    no_components=64,      # Embedding dimension
    loss='warp',           # Or 'bpr', 'logistic', 'warp-kos'
    learning_rate=0.05,
    item_alpha=1e-6,       # L2 penalty on item features
    user_alpha=1e-6        # L2 penalty on user features
)

model.fit(
    interactions,
    user_features=user_features_matrix,
    item_features=item_features_matrix,
    epochs=30,
    num_threads=4
)

# Predict
predictions = model.predict(
    user_ids,
    item_ids,
    user_features=user_features_matrix,
    item_features=item_features_matrix
)
```

#### Properties

| Aspect | Description |
|--------|-------------|
| **Complexity** | Moderate - similar to MF but with feature embeddings |
| **Cold Start** | Excellent - best-in-class for new users/items with features |
| **Interpretability** | Moderate - can examine feature embeddings |
| **Scalability** | Good - efficient implementation with HOGWILD! SGD |
| **Best For** | Cold start scenarios, metadata-rich domains, hybrid systems |

#### Key Advantages

1. **Unified Model**: Same framework handles warm-start (with ID features) and cold-start (without ID features)
2. **Feature Sharing**: Multiple users/items can share features (e.g., all movies with "genre:Action" share that feature's embedding)
3. **Flexible Feature Definition**: Easy to add new features without retraining from scratch
4. **Implicit Feedback Focus**: Designed for real-world recommendation scenarios

#### When to Use

- **Cold start is critical**: E-commerce with high item churn, news recommendation
- **Rich metadata available**: Movies, music, articles with detailed features
- **Implicit feedback**: Clicks, views, purchases rather than explicit ratings
- **Need for explanations**: Feature contributions are interpretable

---

## Comparison of Approaches

### Comprehensive Comparison Table

| Approach | Complexity | Cold Start | Interpretability | Training Time | Inference Time | Best Use Case |
|----------|------------|------------|------------------|---------------|----------------|---------------|
| **Factorization Machines** | Moderate | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Moderate | Fast | CTR prediction, sparse features |
| **SVD++ Extensions** | Moderate | ⭐⭐⭐ | ⭐⭐⭐⭐ | Moderate | Moderate | Explicit ratings + implicit feedback |
| **Regression-Based** | Low | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Fast | Very Fast | Small feature sets, simple relationships |
| **Content-Boosted CF** | Moderate | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Moderate | Fast | Very sparse data, strong content signal |
| **Hybrid MF** | Moderate | ⭐⭐⭐⭐ | ⭐⭐⭐ | Moderate | Fast | Balanced CF + content integration |
| **Neural CF with Features** | High | ⭐⭐⭐⭐⭐ | ⭐⭐ | Slow (GPU) | Moderate | Complex patterns, large datasets |
| **LightFM** | Moderate | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Fast | Fast | Cold start critical, implicit feedback |

**Legend**: ⭐ = Poor, ⭐⭐⭐ = Good, ⭐⭐⭐⭐⭐ = Excellent

### Decision Flowchart

```
Start: What's your primary concern?

├─ Cold start is critical?
│  ├─ YES → LightFM or Neural CF with Features
│  └─ NO → Continue
│
├─ Feature set size?
│  ├─ Small (< 10 features) → Regression-Based
│  ├─ Medium (10-100 features) → Hybrid MF or SVD++ Extensions
│  └─ Large (> 100 features) → Factorization Machines
│
├─ Data type?
│  ├─ Explicit ratings → SVD++ Extensions or Regression-Based
│  └─ Implicit feedback → LightFM or Neural CF
│
├─ Need interpretability?
│  ├─ YES → Regression-Based or Content-Boosted CF
│  └─ NO → Neural CF with Features (best accuracy)
│
├─ Computational resources?
│  ├─ Limited → Regression-Based or LightFM
│  └─ GPU available → Neural CF with Features
│
└─ Data sparsity level?
   ├─ Extremely sparse → Content-Boosted CF
   └─ Moderate → Any approach works
```

### Performance Characteristics

| Approach | RMSE Improvement* | Training Time | Memory Usage | Implementation Difficulty |
|----------|-------------------|---------------|--------------|---------------------------|
| Regression-Based | +5-10% | 1x | 1x | Easy |
| SVD++ Extensions | +10-15% | 1.5x | 1.2x | Moderate |
| Hybrid MF | +10-20% | 1.5x | 1.3x | Moderate |
| Content-Boosted CF | +10-15% | 2x | 1.5x | Moderate |
| Factorization Machines | +15-25% | 2x | 1.2x | Moderate |
| LightFM | +15-30% | 1.5x | 1.3x | Easy |
| Neural CF with Features | +20-35% | 5x | 2x | Hard |

*Compared to basic MF without features. Actual improvements vary by dataset and feature quality.

---

## Practical Considerations

### Feature Engineering Guidelines

#### Categorical Features

**One-Hot Encoding** (for small cardinality):
```python
# Gender: {M, F, Other}
gender_M = 1 if gender == 'M' else 0
gender_F = 1 if gender == 'F' else 0
gender_Other = 1 if gender == 'Other' else 0
```

**Embedding/Hashing** (for high cardinality):
```python
# Location: thousands of cities
# Use feature hashing or learned embeddings
location_hash = hash(location) % embedding_table_size
location_embedding = embedding_table[location_hash]
```

**Multi-Hot Encoding** (for multi-valued features):
```python
# Movie genres: can have multiple
genres = ['Action', 'Sci-Fi', 'Thriller']
genre_Action = 1
genre_SciFi = 1
genre_Thriller = 1
genre_Drama = 0
# ... (rest 0)
```

#### Continuous Features

**Normalization**:
```python
# Z-score normalization
age_normalized = (age - mean_age) / std_age

# Min-max scaling
year_scaled = (year - min_year) / (max_year - min_year)

# Log transformation for skewed distributions
views_log = np.log1p(num_views)  # log(1 + x) to handle 0
```

**Binning** (discretization):
```python
# Age buckets
age_bucket = pd.cut(age, bins=[0, 18, 25, 35, 50, 65, 100],
                    labels=['<18', '18-24', '25-34', '35-49', '50-64', '65+'])
```

#### Temporal Features

**Cyclical Encoding** (for periodic features):
```python
# Hour of day (0-23)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# Day of week (0-6)
day_sin = np.sin(2 * np.pi * day / 7)
day_cos = np.cos(2 * np.pi * day / 7)
```

**Recency Features**:
```python
# Time since last interaction
days_since_last_login = (current_date - last_login_date).days
recency_score = 1.0 / (1.0 + days_since_last_login)
```

#### Feature Selection

**Importance Analysis**:
1. Train simple model (e.g., regression-based MF)
2. Examine feature weights $|\alpha_j|$, $|\beta_k|$
3. Remove low-importance features
4. Retrain with reduced feature set

**Correlation Analysis**:
```python
# Remove highly correlated features
correlation_matrix = np.corrcoef(features.T)
# If features i and j have correlation > 0.9, keep only one
```

### Implementation Strategies

#### Start Simple, Add Complexity

1. **Baseline**: Basic MF without features
2. **Stage 1**: Add regression-based features (easiest)
3. **Stage 2**: If needed, try Hybrid MF or LightFM
4. **Stage 3**: For best performance, Neural CF with features

#### Feature Preprocessing Pipeline

```python
class FeaturePreprocessor:
    def __init__(self):
        self.categorical_encoders = {}
        self.continuous_scalers = {}
    
    def fit(self, df):
        # Fit encoders for categorical features
        for col in categorical_columns:
            encoder = LabelEncoder()
            encoder.fit(df[col])
            self.categorical_encoders[col] = encoder
        
        # Fit scalers for continuous features
        for col in continuous_columns:
            scaler = StandardScaler()
            scaler.fit(df[[col]])
            self.continuous_scalers[col] = scaler
    
    def transform(self, df):
        transformed = {}
        
        # Transform categorical
        for col, encoder in self.categorical_encoders.items():
            transformed[col] = encoder.transform(df[col])
        
        # Transform continuous
        for col, scaler in self.continuous_scalers.items():
            transformed[col] = scaler.transform(df[[col]]).flatten()
        
        return pd.DataFrame(transformed)
```

#### Handling Missing Features

**Strategies**:
1. **Imputation**: Fill with mean/median/mode
2. **Indicator variable**: Add binary "is_missing" feature
3. **Separate embedding**: Learn special embedding for missing values
4. **Feature-specific defaults**: Domain knowledge (e.g., age=30 as default)

```python
# Imputation with indicator
age_filled = age.fillna(age.median())
age_is_missing = age.isna().astype(int)
```

### Evaluation Considerations

#### Metrics by Scenario

**Cold Start Performance**:
- Evaluate separately on new users/items with no interaction history
- Metrics: RMSE, MAE on cold-start subset

```python
# Filter test set for cold start users
cold_start_users = test_users.difference(train_users)
cold_start_predictions = predictions[predictions['user'].isin(cold_start_users)]
cold_start_rmse = compute_rmse(cold_start_predictions)
```

**Feature Contribution Analysis**:
```python
# Ablation study
models = {
    'No features': train_mf(use_features=False),
    'User features only': train_mf(use_user_features=True, use_item_features=False),
    'Item features only': train_mf(use_user_features=False, use_item_features=True),
    'All features': train_mf(use_user_features=True, use_item_features=True)
}

for name, model in models.items():
    print(f"{name}: RMSE = {evaluate(model)}")
```

#### Cross-Validation Strategies

**Time-based split** (for temporal datasets):
```python
# Train on first 80% of time, test on last 20%
cutoff_date = ratings['timestamp'].quantile(0.8)
train = ratings[ratings['timestamp'] <= cutoff_date]
test = ratings[ratings['timestamp'] > cutoff_date]
```

**Cold-start specific split**:
```python
# Ensure test set has some new users/items
train_users = set(train['user'])
test_users_existing = test[test['user'].isin(train_users)]
test_users_cold = test[~test['user'].isin(train_users)]

# Evaluate both separately
existing_rmse = evaluate(model, test_users_existing)
cold_rmse = evaluate(model, test_users_cold)
```

### Scalability Considerations

#### Memory Optimization

**Sparse Feature Matrices**:
```python
from scipy.sparse import csr_matrix

# Store features as sparse matrix
# Only non-zero entries consume memory
feature_matrix = csr_matrix(one_hot_encoded_features)
```

**Feature Hashing**:
```python
from sklearn.feature_extraction import FeatureHasher

# Hash high-cardinality categorical features
hasher = FeatureHasher(n_features=1000, input_type='string')
hashed_features = hasher.transform(categorical_feature_strings)
```

#### Computational Optimization

**Mini-batch Training**:
```python
# Don't load all data at once
for epoch in range(n_epochs):
    for batch in data_loader:
        # Process batch
        loss = train_step(batch)
```

**Parallel Feature Computation**:
```python
from joblib import Parallel, delayed

# Compute feature representations in parallel
user_features = Parallel(n_jobs=-1)(
    delayed(compute_user_features)(user_id) 
    for user_id in user_ids
)
```

---

## Concrete Examples

### Example 1: Movie Recommendation

**Scenario**: Predict rating for User 42 on Movie 100

**User 42 Features**:
- Age: 28
- Gender: Male
- Location: New York
- Occupation: Engineer
- Past ratings: 50 movies (average rating: 3.8)

**Movie 100 Features**:
- Title: "Inception"
- Genres: Action, Sci-Fi, Thriller
- Release Year: 2010
- Director: Christopher Nolan
- Average Rating: 4.5
- Number of Ratings: 10,000

#### Approach 1: Regression-Based

```python
# Feature vectors
f_u = [28/100, 1, 0, ...]  # age normalized, gender (M=1, F=0), ...
g_i = [1, 1, 0, 2010, ...]  # genre_action=1, genre_scifi=1, ...

# Latent factors (learned from data)
p_u = [0.5, -0.2, 0.8, ...]  # k=10 dimensions
q_i = [0.3, 0.1, 0.7, ...]

# Feature weights (learned)
alpha = [0.02, 0.15, ...]  # weights for user features
beta = [-0.1, 0.3, 0.05, ...]  # weights for item features

# Prediction
r_pred = (
    3.5 +                    # global mean
    0.3 +                    # user bias
    0.5 +                    # item bias
    p_u @ q_i +              # MF term = 0.63
    alpha @ f_u +            # user feature term = 0.12
    beta @ g_i               # item feature term = 0.25
)
# r_pred ≈ 5.3 → clip to [1, 5] → 5.0
```

#### Approach 2: LightFM

```python
# User features as set
F_u = {
    'user_id:42',
    'age:25-34',
    'gender:M',
    'location:NYC',
    'occupation:Engineer'
}

# Item features as set
F_i = {
    'movie_id:100',
    'genre:Action',
    'genre:Sci-Fi',
    'genre:Thriller',
    'year:2010s',
    'director:Nolan',
    'popularity:High'
}

# Each feature has learned embedding (k=10)
e['user_id:42'] = [0.2, 0.5, -0.1, ...]
e['age:25-34'] = [0.1, 0.0, 0.2, ...]
e['gender:M'] = [0.05, 0.1, -0.05, ...]
# ... (and so on for all features)

# User representation (sum of feature embeddings)
p_u = sum([e[f] for f in F_u])
# p_u = [0.45, 0.7, 0.2, ...]

# Item representation
q_i = sum([e[f] for f in F_i])
# q_i = [0.3, 0.5, 0.3, ...]

# Prediction
r_pred = sigmoid(p_u @ q_i)
```

**Key Difference**: 
- Regression-based: Features added linearly to MF term
- LightFM: Features define the representations themselves

### Example 2: E-Commerce Product Recommendation

**Scenario**: Recommend products for a new user (cold start)

**New User Features**:
- Age: 35
- Gender: Female
- Location: San Francisco
- Device: Mobile
- No purchase history

**Product Features** (candidate):
- Category: Electronics → Camera
- Price: $500
- Brand: Sony
- Average Rating: 4.3 stars
- Popularity: 85th percentile

#### Approaches that Work

**❌ Basic MF**: Cannot make predictions (no user history)

**✅ Regression-Based with Features**:
```python
# Even without user ID in model, can predict using features
f_u = [35, 1, SF_encoding, mobile_encoding]
g_i = [camera_encoding, 500, Sony_encoding, 4.3, 0.85]

# Use learned feature weights
score = alpha @ f_u + beta @ g_i
# Can rank products based on feature scores
```

**✅ LightFM**:
```python
F_u = {'age:35-44', 'gender:F', 'location:SF', 'device:mobile'}
F_i = {'category:Electronics/Camera', 'price:$400-600', 'brand:Sony', ...}

p_u = sum([e[f] for f in F_u])  # Build representation from features
q_i = sum([e[f] for f in F_i])
score = p_u @ q_i
# Works perfectly for cold start!
```

**✅ Neural CF with Features**:
```python
# Forward pass
user_feat_emb = neural_net(f_u)  # Process features through network
item_feat_emb = neural_net(g_i)
score = mlp([user_feat_emb, item_feat_emb])
# Can make prediction even without user ID embedding
```

### Example 3: News Article Recommendation

**Scenario**: Real-time news recommendation (extreme cold start on items)

**Challenge**: New articles arrive every minute; no interaction history

**Article Features**:
- Text embedding (from BERT): 768-dimensional vector
- Category: Politics
- Published: 5 minutes ago
- Author: Known journalist (ID: 123)
- Trending Score: 0.65

**User Context**:
- Historical interests: [Politics: 0.8, Sports: 0.3, Tech: 0.5]
- Reading time: Morning (8 AM)
- Device: Mobile
- Session length: 3 articles read

#### Recommended Approach: Hybrid MF

```python
# User representation
p_u_CF = learned_latent_factors[user_id]  # From historical data
p_u_context = W_u @ [interest_politics, interest_sports, ..., hour, device]

p_u = p_u_CF + p_u_context

# Item representation (content-heavy for new articles)
q_i_CF = learned_latent_factors.get(article_id, zeros(k))  # Likely zeros for new article
q_i_content = W_i @ [bert_embedding, category_emb, author_emb, recency, trending]

q_i = q_i_CF + q_i_content  # Relies heavily on content for new articles

# Prediction
score = p_u @ q_i
# Content features enable immediate recommendations for new articles
```

---

## References

### Academic Papers

#### Factorization Machines
- Rendle, S. (2010). "Factorization Machines". *IEEE International Conference on Data Mining (ICDM)*. https://doi.org/10.1109/ICDM.2010.127
- Rendle, S. (2012). "Factorization Machines with libFM". *ACM Transactions on Intelligent Systems and Technology (TIST)*, 3(3), 1-22.

#### Hybrid and Feature-Rich Methods
- Pilászy, I., Zibriczky, D., & Tikk, D. (2010). "Fast ALS-based matrix factorization for explicit and implicit feedback datasets". *RecSys 2010*.
- Koren, Y. (2008). "Factorization meets the neighborhood: a multifaceted collaborative filtering model". *KDD 2008*. https://doi.org/10.1145/1401890.1401944
- Melville, P., Mooney, R. J., & Nagarajan, R. (2002). "Content-boosted collaborative filtering for improved recommendations". *AAAI 2002*.

#### Neural Methods
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). "Neural Collaborative Filtering". *WWW 2017*. https://arxiv.org/abs/1708.05031
- Strub, F., Mary, J., & Gaudel, R. (2016). "Hybrid Collaborative Filtering with Neural Networks". *RecSys 2016*. https://arxiv.org/abs/1603.00806
- Nguyen, T., & Takasu, A. (2018). "NPE: Neural Personalized Embedding for Collaborative Filtering". *IJCAI 2018*. https://www.ijcai.org/proceedings/2018/0219.pdf
- Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction". *IJCAI 2017*. https://arxiv.org/abs/1703.04247

#### LightFM
- Kula, M. (2015). "Metadata Embeddings for User and Item Cold-start Recommendations". *CBRecSys 2015*. https://arxiv.org/abs/1507.08439

### Software Libraries

#### Factorization Machines
- **xLearn**: https://github.com/aksnzhy/xlearn
  - Fast C++ implementation with Python bindings
  - Supports FM, FFM, Linear models
- **libFM**: http://www.libfm.org/
  - Original implementation by Steffen Rendle
- **fastFM**: https://github.com/ibayer/fastFM
  - Python/Cython implementation

#### LightFM
- **LightFM**: https://github.com/lyst/lightfm
  - Hybrid recommendation algorithm in Python
  - Excellent documentation and examples
  - Installation: `pip install lightfm`

#### Neural Networks
- **PyTorch**: https://pytorch.org/
- **TensorFlow**: https://www.tensorflow.org/
- **TensorFlow Recommenders**: https://www.tensorflow.org/recommenders

#### Traditional MF with Features
- **Surprise**: http://surpriselib.com/
  - Scikit-learn style recommender library
  - SVD++ implementation
- **Cornac**: https://github.com/PreferredAI/cornac
  - Multimodal recommendation library
  - Many algorithms including hybrid methods

### Datasets for Experimentation

**With Rich Features**:
- **MovieLens**: https://grouplens.org/datasets/movielens/
  - User demographics (age, gender, occupation)
  - Movie metadata (genres, year, titles)
- **Amazon Product Data**: https://jmcauley.ucsd.edu/data/amazon/
  - Product categories, descriptions, prices
  - User reviews with helpful votes
- **Last.fm**: http://ocelma.net/MusicRecommendationDataset/
  - User demographics, artist tags
  - Listening history with timestamps
- **Yelp Dataset**: https://www.yelp.com/dataset
  - Business features (category, location, price)
  - User features (review count, elite status)

### Related Documentation

- **Main Guide**: [../README.md](../README.md) - Comprehensive recommender systems guide
  - Lines 330-419: Matrix factorization fundamentals
  - Lines 420-453: Hybrid methods overview
- **Implementation Guide**: [README.md](README.md) - ALS and SGD implementations
- **Hybrid Systems**: [../hybrid_recommendation/README.md](../hybrid_recommendation/README.md) - Combining collaborative and content-based methods

---

## Summary

Incorporating user and item features into matrix factorization addresses critical limitations of pure collaborative filtering:

1. **For Simple Cases**: Start with **Regression-Based Integration** - easy to implement and interpret
2. **For Cold Start**: Use **LightFM** - purpose-built for this scenario with excellent performance
3. **For Rich Features**: **Factorization Machines** handle arbitrary feature interactions elegantly
4. **For Best Accuracy**: **Neural CF with Features** achieves state-of-the-art but requires more resources
5. **For Production Systems**: Consider **Hybrid MF** for good balance of performance and complexity

**Key Takeaway**: The choice depends on your specific constraints (cold start severity, feature richness, computational budget, interpretability needs). Start simple and add complexity only when needed.

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Part of**: Recommender Systems Guide - Matrix Factorization Module
