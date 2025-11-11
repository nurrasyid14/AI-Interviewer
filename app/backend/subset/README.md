# 🎯 Subset Selection Module

Part of the **AI Interviewer** backend pipeline.  
Responsible for intelligently partitioning datasets into training, validation, and testing subsets — enabling **reliable model evaluation** and **bias-free learning**.

---

## 📦 Overview

The **Subset Module** provides multiple strategies for dividing your processed dataset into meaningful subsets for model validation.  
This ensures the AI interviewer’s predictions are **generalizable**, **robust**, and **statistically consistent** across diverse participants and responses.

---

## 🧩 Submodules

subset/
│
├── k_fold.py # Implements standard K-Fold cross-validation
├── leave_one_out.py # Implements Leave-One-Out validation for small datasets
├── cluster_subset.py # Cluster-based sampling for balanced subsets
└── subset_utils.py # Shared helper functions

pgsql
Copy code

---

### 1️⃣ `k_fold.py`

Performs **K-Fold Cross-Validation**, a classical and efficient strategy for model evaluation.

**Features:**
- Configurable `k` value (default: 5)  
- Automatically stratifies subsets when labels are available  
- Returns indices or ready-to-train splits  

**Usage Example:**
```python
from subset.k_fold import KFoldSubset

splitter = KFoldSubset(k=5, shuffle=True, random_state=42)
for train_idx, test_idx in splitter.split(X, y):
    print("Train:", train_idx, "Test:", test_idx)
```
### 2️⃣ leave_one_out.py
Implements Leave-One-Out Cross-Validation (LOOCV) — an exhaustive validation strategy ideal for small datasets or sensitive evaluation scenarios.

Features:

Trains on n-1 samples and tests on 1 each iteration

Provides the most granular model performance estimates

Useful in interview session-level analysis

Usage Example:

```python
Copy code
from subset.leave_one_out import LeaveOneOutSubset

splitter = LeaveOneOutSubset()
for train_idx, test_idx in splitter.split(X):
    print(f"Testing on sample: {test_idx}")
```

### 3️⃣ cluster_subset.py
Uses unsupervised clustering (e.g., K-Means) to build representative and balanced subsets.
Ideal for heterogeneous data such as multilingual interview transcripts or diverse demographic groups.

Features:

Groups data points by similarity in feature space

Ensures balanced representation across folds

Compatible with embeddings or TF-IDF features

Usage Example:

```python
Copy code
from subset.cluster_subset import ClusterSubset

subset = ClusterSubset(n_clusters=3)
train_set, test_set = subset.create_balanced_subsets(X_features)
```

## 4️⃣ subset_utils.py
Utility helpers for consistent subset management.

Includes:

Seed control for reproducibility

Index randomization utilities

Subset statistics reporting (size, label balance, etc.)

## 🧠 Conceptual Flow
```
Copy code
Preprocessed Dataset
     ↓
Feature Extraction
     ↓
Subset Selection (K-Fold / LOOCV / Cluster)
     ↓
Model Training & Evaluation
Each splitter ensures non-overlapping, statistically valid, and reproducible data partitions — key to trustworthy model insights.
```
## 🧰 Requirements
```
numpy
scikit-learn
pandas
```

⚙️ Example Integration
```python
from subset.k_fold import KFoldSubset
from feature_extracting.vectorizer import TFIDF
```
## Prepare features
```python
tfidf = TFIDF()
X = tfidf.fit_transform(docs)
```
## Generate folds
```python
kfold = KFoldSubset(k=5)
for train_idx, test_idx in kfold.split(X):
    print("Training:", len(train_idx), "Testing:", len(test_idx))
```

## 🧩 Integration Notes
This module feeds into:

ensemble_learning/ → for training and blending multiple models

visualizer/ → to display validation performance per fold

evaluator/ → to aggregate cross-validation metrics