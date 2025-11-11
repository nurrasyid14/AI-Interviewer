# learning_evaluation/evaluation_pipeline.py
from typing import Dict, Any
import numpy as np
from .metrics import MetricsEvaluator
from .calibration import ModelCalibrator
from .bias_analysis import BiasAnalyzer

class EvaluationPipeline:
    """
    End-to-end evaluation pipeline that interfaces with EnsembleController and Visualizer.
    Handles accuracy, reliability, and bias diagnostics.
    """

    def __init__(self, sensitive_attr: str = None):
        self.metrics_eval = MetricsEvaluator()
        self.calibrator = ModelCalibrator()
        self.bias_analyzer = BiasAnalyzer(sensitive_attr) if sensitive_attr else None

    def evaluate_model(self, model, X_test, y_test, X_train=None, y_train=None) -> Dict[str, Any]:
        """Full evaluation including calibration and optional bias check."""
        results = {}

        # Predictions
        y_pred = model.predict(X_test)
        results["metrics"] = self.metrics_eval.evaluate(y_test, y_pred)

        # Probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1] if model.predict_proba(X_test).ndim > 1 else model.predict_proba(X_test)
            results["reliability"] = self.calibrator.reliability_curve(y_test, y_prob)

        # Bias Analysis
        if self.bias_analyzer and hasattr(X_test, "columns"):
            if self.bias_analyzer.sensitive_attr in X_test.columns:
                df_eval = X_test.copy()
                df_eval["y_true"] = y_test
                df_eval["y_pred"] = y_pred
                results["bias"] = self.bias_analyzer.analyze(df_eval, "y_true", "y_pred")

        return results
