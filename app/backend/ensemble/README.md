# 🧠 Ensemble Block

The **Ensemble module** implements a unified interface for training, evaluating, and combining multiple models (Random Forest, Neural Network, SVM, Naive Bayes).  
It acts as the **core predictive engine** of the pipeline, supporting both standalone and ensemble modes.

## 📁 Structure
```
ensemble/
├── init.py
├── random_forest.py
├── neural_network.py
├── svm.py
├── naive_bayes.py
└── ensemble_controller.py
```

## ⚙️ Key Features
- Unified train/predict/save interface for all models.
- EnsembleController supports:
  - **Voting** — majority label from all models.
  - **Weighted averaging** — average probabilities based on assigned weights.
- Easy to integrate with higher-level pipeline (e.g., Evaluator, Visualizer).
- Compatible with scikit-learn’s dataset format (NumPy arrays or pandas DataFrames).

## 🚀 Example Usage
```python
from ensemble import EnsembleController
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

controller = EnsembleController(mode="weighted", weights={"rf":1.5,"nn":1.0,"svm":1.2,"nb":0.8})
controller.train_all(X_train, y_train)
acc = controller.evaluate(X_test, y_test, metric_fn=accuracy_score)

print(f"Accuracy: {acc:.3f}")
```

## 📦 Dependencies
```
scikit-learn

numpy

joblib
```