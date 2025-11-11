# ensemble/ensemble_controller.py
from typing import Dict, Any
import numpy as np
from .random_forest import RandomForestModel
from .neural_network import NeuralNetworkModel
from .svm import SVMModel
from .naive_bayes import NaiveBayesModel

class EnsembleController:
    """
    Central controller for ensemble-based predictions.
    Handles model training, voting, and weighted averaging.
    """

    def __init__(self, mode: str = "voting", weights: Dict[str, float] = None):
        self.mode = mode
        self.models = {
            "rf": RandomForestModel(),
            "nn": NeuralNetworkModel(),
            "svm": SVMModel(),
            "nb": NaiveBayesModel()
        }
        self.weights = weights or {m: 1.0 for m in self.models.keys()}

    def train_all(self, X, y):
        """Train all ensemble members."""
        for name, model in self.models.items():
            print(f"Training {name.upper()}...")
            model.train(X, y)

    def predict(self, X):
        """Predict using ensemble method."""
        if self.mode == "voting":
            return self._voting_predict(X)
        elif self.mode == "weighted":
            return self._weighted_predict(X)
        else:
            raise ValueError("Invalid ensemble mode.")

    def _voting_predict(self, X):
        preds = np.array([m.predict(X) for m in self.models.values()])
        # Majority vote per column
        final = [np.bincount(col).argmax() for col in preds.T]
        return np.array(final)

    def _weighted_predict(self, X):
        """Weighted probability averaging."""
        probas = []
        for name, model in self.models.items():
            p = model.predict_proba(X)
            probas.append(self.weights[name] * p)
        avg_proba = np.sum(probas, axis=0) / sum(self.weights.values())
        return np.argmax(avg_proba, axis=1)

    def predict_proba(self, X):
        """Return averaged probabilities."""
        probas = []
        for name, model in self.models.items():
            probas.append(self.weights[name] * model.predict_proba(X))
        return np.sum(probas, axis=0) / sum(self.weights.values())

    def evaluate(self, X, y, metric_fn):
        """Compute evaluation metric (e.g., accuracy_score)."""
        y_pred = self.predict(X)
        return metric_fn(y, y_pred)

    def save_all(self, base_path: str):
        """Save each trained model to base_path/model_name.pkl."""
        import os
        os.makedirs(base_path, exist_ok=True)
        for name, model in self.models.items():
            model.save(f"{base_path}/{name}.pkl")

    def load_all(self, base_path: str):
        """Load models from base_path/model_name.pkl."""
        import os
        for name, model in self.models.items():
            path = f"{base_path}/{name}.pkl"
            if os.path.exists(path):
                model.load(path)
            else:
                print(f"Model file {path} not found. Skipping load for {name}.")
