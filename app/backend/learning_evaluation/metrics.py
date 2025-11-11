# learning_evaluation/metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any

class MetricsEvaluator:
    """Computes standard model performance metrics."""

    def __init__(self, average: str = "weighted"):
        self.average = average

    def evaluate(self, y_true, y_pred) -> Dict[str, Any]:
        """Return a dictionary of basic classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=self.average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=self.average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=self.average, zero_division=0),
        }
