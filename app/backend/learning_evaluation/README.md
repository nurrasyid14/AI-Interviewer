# 🎓 Learning Evaluation Module

This block performs **quantitative and ethical evaluation** of AI models.  
It ensures that every trained model in the pipeline is *accurate, calibrated, and fair*.

---

## 📦 Overview
`learning_evaluation/` connects trained models from the `ensemble/` block to validation and visual analytics.
```
learning_evaluation/
├── init.py
├── metrics.py # Accuracy, F1, precision, recall
├── calibration.py # Reliability calibration
├── bias_analysis.py # Fairness & group disparity
└── evaluation_pipeline.py
```
---

## 🧩 Submodules

### 1️⃣ Metrics
`metrics.py` computes standard classification metrics with flexible averaging schemes.

```python
from learning_evaluation.metrics import MetricsEvaluator
eval = MetricsEvaluator()
eval.evaluate(y_true, y_pred)
```

### 2️⃣ Calibration
calibration.py improves probabilistic predictions and produces reliability curves.

```python
from learning_evaluation.calibration import ModelCalibrator
cal = ModelCalibrator(method="isotonic")
calibrated_model = cal.calibrate(model, X_train, y_train)
```

### 3️⃣ Bias Analysis
bias_analysis.py detects fairness disparities across demographic or linguistic groups.
```python
from learning_evaluation.bias_analysis import BiasAnalyzer
bias = BiasAnalyzer("gender")
bias.analyze(df, y_true_col="actual", y_pred_col="predicted")
```

### 4️⃣ Evaluation Pipeline
evaluation_pipeline.py orchestrates the entire evaluation flow:

Computes metrics

Generates calibration curves

Runs bias diagnostics (if sensitive attribute available)

Outputs structured evaluation reports for the visualizer/ module

## 🧠 Example Usage
```python
from ensemble import EnsembleController
from learning_evaluation import EvaluationPipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

controller = EnsembleController()
controller.train_all(X_train, y_train)

pipeline = EvaluationPipeline()
report = pipeline.evaluate_model(controller.models["rf"], X_test, y_test)

print(report["metrics"])
```


## 📦 Dependencies
```
scikit-learn

numpy

pandas
```