# learning_evaluation/calibration.py
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from typing import Dict, Any

class ModelCalibrator:
    """Handles model probability calibration and reliability curve generation."""

    def __init__(self, method: str = "isotonic", cv: int = 5):
        self.method = method
        self.cv = cv

    def calibrate(self, model, X_train, y_train):
        """Calibrate a given probabilistic model using sklearn’s CalibratedClassifierCV."""
        calibrated = CalibratedClassifierCV(model, method=self.method, cv=self.cv)
        calibrated.fit(X_train, y_train)
        return calibrated

    def reliability_curve(self, y_true, y_prob, n_bins: int = 10) -> Dict[str, np.ndarray]:
        """Generate reliability curve data for plotting."""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        return {"true": prob_true, "pred": prob_pred}
