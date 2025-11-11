# 🎨 Visualizer Module

A reusable and modular visualization suite for the **AI Interviewer** backend.  
Built for clarity, diagnostics, and presentation — from raw data inspection to model evaluation.

---

## 🧩 Overview

The **Visualizer** is designed as a *service module* that can plug into multiple stages:
- 🧹 Preprocessing & Linguistics
- 🧠 Feature Extraction
- 🧪 Subset Selection
- 🏗️ Ensemble Learning
- 📊 Reporting & Dashboard Integration

It provides a clean API that auto-routes visual tasks to their corresponding engines:
`data_visuals`, `model_visuals`, `evaluation_visuals`, and `linguistic_visuals`.

---

## 📁 Directory Structure
```
visualizer/
│
├── __init__.py
│
├── evaluation/                  # Developer-focused evaluation visuals
│   ├── __init__.py
│   ├── model_performance.py     # Accuracy, loss, learning curve visualizations
│   ├── confusion_matrix_plotter.py  # Classification confusion matrix
│   └── reliability_distribution.py  # Confidence, calibration, reliability plots
│
├── user_dashboard/              # End-user visual dashboard components
│   ├── __init__.py
│   ├── reliability_overview.py  # Reliability & performance summary
│   ├── sentiment_chart.py       # Emotion and polarity visualization
│   ├── relevance_heatmap.py     # Relevance & importance relationships
│   └── overall_dashboard.py     # Integrates multiple visuals into a single dashboard
│
├── shared_components/           # Shared visual utilities and exports
│   ├── __init__.py
│   ├── plotly_theme.py          # Global Plotly/Matplotlib theme manager
│   └── export_utils.py          # For exporting figures or embedding in reports
│
└── visualizer_pipeline.py       # Central routing/orchestration module

```

---

## ⚙️ Components

### 1️⃣ DataVisualizer (`data_visuals.py`)
**Purpose:** Inspect raw or processed datasets before training.

**Key Methods:**
- `show_summary(df)` → displays shape, null heatmap, and stats
- `plot_distribution(df, column)` → visualize single feature distributions
- `plot_correlation(df)` → correlation heatmap for numerical features

---

### 2️⃣ ModelVisualizer (`model_visuals.py`)
**Purpose:** Monitor model learning and architecture at runtime.

**Key Methods:**
- `plot_training(history)` → visualize loss/accuracy curves
- `plot_feature_importance(model, feature_names)` → bar chart of feature importance (if supported)

---

### 3️⃣ EvaluationVisualizer (`evaluation_visuals.py`)
**Purpose:** Evaluate classification or regression model performance.

**Key Methods:**
- `plot_confusion_matrix(y_true, y_pred)`
- `plot_classification_report(y_true, y_pred)`
- `plot_roc_curve(y_true, y_prob)`
- `plot_precision_recall(y_true, y_prob)`

---

### 4️⃣ LinguisticVisualizer (`linguistic_visuals.py`)
**Purpose:** Visualize linguistic, textual, or embedding-based results.

**Key Methods:**
- `plot_word_freq(tokens, top_n=20)`
- `plot_wordcloud(tokens)`
- `plot_embeddings(embeddings, labels=None, method="tsne")`
- `plot_pos_distribution(pos_tags)`

---

## 🔗 Central Interface — `visualizer_main.py`

The **Visualizer** class unifies all visualization utilities under a single entry point.

```python
from visualizer.visualizer_main import Visualizer

viz = Visualizer(style="seaborn-v0_8")

# Example: Visualize dataset
viz.show_data_summary(df)

# Example: Model training
viz.show_model_performance(history)

# Example: Evaluation
viz.show_evaluation_metrics(y_true, y_pred, labels=classes)

# Example: Embedding visualization
viz.show_embeddings(embeddings, labels, method="tsne")
```

🧠 Usage in Pipeline
```
Preprocessor
   ↓
Feature Extraction
   ↓
Subset Selection
   ↓
Model Training / Ensemble
   ↓
Visualizer  ←── Used throughout the flow
```

The Visualizer can be imported and used in any stage — for inspection, debugging, or live monitoring (e.g., during web dashboard updates).

🧰 Requirements
```bash
pip install matplotlib seaborn wordcloud scikit-learn numpy
```
Optional (for linguistic visualization):

```bash
pip install spacy
python -m spacy download en_core_web_md
```
🧪 Example
```python

import pandas as pd
from visualizer.visualizer_main import Visualizer
```
# Demo dataset
```py
df = pd.DataFrame({
    "score": [0.5, 0.7, 0.9, 0.3],
    "result": ["pass", "pass", "fail", "fail"]
})

viz = Visualizer()
viz.show_data_summary(df)
```
# Simulated evaluation
```py
y_true = ["pass", "pass", "fail", "fail"]
y_pred = ["pass", "fail", "fail", "pass"]
viz.show_evaluation_metrics(y_true, y_pred, labels=["fail", "pass"])
```
🧩 Integration Notes
Works natively with outputs from:

subset/ (for k-fold or LOO visual evaluation)

feature_extracting/ (for embedding or TF-IDF visualization)

ensemble_learning/ (for comparing model ensemble results)

Can be extended into interactive visual dashboards using:

Plotly

Dash

Streamlit