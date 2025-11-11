# learning_evaluation/bias_analysis.py
import numpy as np
import pandas as pd
from typing import Dict, Any

class BiasAnalyzer:
    """
    Analyzes group-level fairness metrics.
    Example: performance disparity across genders, languages, or demographics.
    """

    def __init__(self, sensitive_attr: str):
        self.sensitive_attr = sensitive_attr

    def analyze(self, df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> Dict[str, Any]:
        """Compute group-level accuracy and disparity."""
        if self.sensitive_attr not in df.columns:
            raise ValueError(f"Sensitive attribute '{self.sensitive_attr}' not found in dataframe.")

        groups = df[self.sensitive_attr].unique()
        results = {}

        for g in groups:
            subset = df[df[self.sensitive_attr] == g]
            correct = (subset[y_true_col] == subset[y_pred_col]).mean()
            results[g] = round(correct, 4)

        disparity = np.ptp(list(results.values()))  # range between best & worst performing groups
        return {"group_accuracy": results, "disparity": disparity}
