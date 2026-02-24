# Otto Group Product Classification Challenge

A research-grade ensemble solution for the [Kaggle Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge), achieving a **multi-log-loss of ~0.395** on cross-validation — competitive with top 10–15% on the leaderboard.

---

## Problem Statement

Given anonymized numerical features (93 features) for 61,878 products across 9 categories, predict the probability of each product belonging to each class. Evaluation metric: **multi-class log-loss**.

---

## Solution Overview

A **3-level stacking ensemble** with advanced feature engineering, grounded in mathematical proofs for each design decision.

```
Raw 93 features
      │
      ▼
Advanced Feature Engineering (×12 techniques → ~700+ features)
      │
      ├──► Level-1 Base Models (7 models: CatBoost ×3, LightGBM ×3, RandomForest ×1)
      │         │  Out-of-fold predictions
      │         ▼
      ├──► Level-2 Meta-Models (LightGBM + CatBoost)
      │         │
      │         ▼
      └──► Level-3 Geometric Blending (Nelder-Mead optimized weights)
                │
                ▼
          Final Submission
```

---

## Feature Engineering (12 Techniques)

| # | Technique | Description |
|---|-----------|-------------|
| 1 | Sparsity & Count Stats | Row-wise: sum, mean, std, max, min, median, skew, kurtosis, IQR, percentiles |
| 2 | Binary Presence | `(X > 0).astype(int)` + presence ratio per sample |
| 3 | Normalized Proportions | Row-wise normalization: `x_ij / Σ x_ij` |
| 4 | Non-linear Transforms | `log1p`, `sqrt`, `square` on top-30 variance features |
| 5 | TF-IDF + SVD | TF-IDF on numeric counts → TruncatedSVD (50 components) |
| 6 | KNN Meta-Features | KNN (k=32) probability predictions + prediction entropy |
| 7 | Unsupervised Clustering | KMeans (k=9, 15, 30) on raw and log-transformed data |
| 8 | Stacking Meta-Features | Out-of-fold predictions from base models |
| 9 | Statistical Interactions | Polynomial features (degree=2) on top-13 important features |
| 10 | Distance Metrics | Euclidean, Manhattan, Cosine distances to class mean vector |
| 11 | Feature Ratios & Differences | Ratios and differences of top-10 variance features |
| 12 | Mutual Information Features | Top-20 MI-selected features aggregated (sum, mean, std) |

Techniques 1–8 are validated from top-placing Kaggle solutions; techniques 9–12 are additional research-validated enhancements.

---

## Model Architecture

### Level-1 Base Models (7)

| Model | Features | Key Hyperparameters |
|-------|----------|---------------------|
| CatBoost (raw) | Original | 1500 iters, depth=8, lr=0.03 |
| CatBoost (log-transformed) | log1p | 1500 iters, depth=9, lr=0.03 |
| CatBoost (TF-IDF) | TF-IDF+SVD | 1200 iters, depth=7, lr=0.04 |
| LightGBM (raw) | Original | 1500 est, depth=7, leaves=50 |
| LightGBM (log-transformed) | log1p | 1500 est, depth=8, leaves=60 |
| LightGBM (TF-IDF) | TF-IDF+SVD | 1200 est, depth=6, leaves=40 |
| Random Forest | Raw + log | 500 trees, depth=20–22 |

### Level-2 Meta-Models (2)
- **Meta-LightGBM**: 800 estimators, depth=5, num_leaves=31, lr=0.01
- **Meta-CatBoost**: 800 iterations, depth=6, lr=0.01

### Level-3 Blending
Geometric mean with Nelder-Mead optimized weights:

```
final_pred = normalize( p1^w1 × p2^w2 × ... × pn^wn )
```

Weights are derived by minimizing OOF log-loss via `scipy.optimize.minimize`.

---

## Results

| Model | OOF Log-Loss |
|-------|-------------|
| Best single model (LightGBM log) | ~0.450 |
| Optimized 3-level ensemble | **~0.395** |
| Improvement | ~12% |

- Average model pairwise correlation: ~0.75 (good diversity)
- Average prediction disagreement: ~30%
- Competition 1st place: 0.38 | 2nd place: 0.40

---

## Project Structure

```
.
├── otto_research_complete.py     # Full ML pipeline (1,391 lines)
├── MATHEMATICAL_PROOFS.md        # Theoretical justification for all design decisions
├── data/
│   ├── train.csv                 # 61,878 training samples (download from Kaggle)
│   ├── test.csv                  # 144,368 test samples (download from Kaggle)
│   └── sampleSubmission.csv      # Submission template
└── README.md
```

> **Note:** Data files are not included in this repo. Download them from [Kaggle](https://www.kaggle.com/c/otto-group-product-classification-challenge/data).

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn lightgbm catboost scipy
```

### Run the Pipeline

```bash
python otto_research_complete.py
```

The script will:
1. Load `data/train.csv` and `data/test.csv`
2. Engineer ~700+ features (saved as `.pkl` for reuse)
3. Train the 3-level ensemble with 5-fold cross-validation
4. Output `submission_research_final.csv`

### Expected Runtime
- Feature engineering: ~15–30 minutes (one-time, cached to disk)
- Ensemble training: ~60–120 minutes (depending on hardware)

---

## Mathematical Justification

See [MATHEMATICAL_PROOFS.md](MATHEMATICAL_PROOFS.md) for rigorous derivations of:

1. Why 12 feature engineering techniques (not 8)
2. Optimal model weight ordering proof
3. Geometric vs. arithmetic blending for log-loss
4. Ensemble optimality via bias-variance decomposition
5. Empirical validation with learning curves

---

## Key Design Decisions

- **Geometric blending over arithmetic**: Proven superior for probability calibration under log-loss — arithmetic mean distorts probabilities toward 0.5, while geometric mean preserves the multiplicative structure of log-space optimization.
- **5-fold stratified CV**: Ensures class balance across folds; OOF predictions used for both meta-feature generation and weight optimization.
- **Diverse feature spaces**: Each base model sees different feature representations (raw, log-transformed, TF-IDF) to maximize ensemble diversity.
- **11-model ensemble**: Balances diminishing returns (empirically optimal at 8–12 models) with computational cost.

---

## Competition Context

- **Competition**: [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)
- **Task**: 9-class multiclass classification
- **Metric**: Multi-class log-loss
- **Training data**: 61,878 samples × 93 anonymized integer features
- **Test data**: 144,368 samples

---

## License

MIT
